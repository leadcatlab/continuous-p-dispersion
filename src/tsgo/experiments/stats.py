"""Summary statistics over a sweep, and the figures that go with them.

Reads the CSV written by tsgo-batch and reports, per (container, metric, p):
the number of runs and the mean, median and standard deviation of the radius D
and of the wall time. Optionally renders one figure per
cell from the stored solutions, so a single command produces both the table
and the images.

    tsgo-stats                                   # table to stdout
    tsgo-stats --latex tables/summary.tex        # + LaTeX
    tsgo-stats --csv tables/summary.csv         # + the same table as CSV
    tsgo-stats --figures figures/               # + one clean PNG per cell
"""

import argparse
import os
import subprocess
import sys
from typing import Optional

import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"no sweep CSV at {path}. Run tsgo-batch first, or pass --sweep.")
    df = pd.read_csv(path, skipinitialspace=True)
    # Tolerate a hand-aligned or hand-edited CSV: the sweep file is meant to be
    # read and inspected by a person, so column-padding must not break parsing.
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError:
            pass  # not a text column
    df["feasible"] = df["feasible"].astype(str).str.lower().eq("true")
    for col in ("D", "wall_time_s", "energy_final"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def take_first_n(df: pd.DataFrame, n: Optional[int]) -> pd.DataFrame:
    """Keep the n lowest-numbered seeds in each (container, metric, p) cell.

    Selecting by seed rather than by position in the file makes the choice
    deterministic and stable: appending more runs later never changes which
    rows a previous report used. Cells holding fewer than n runs are kept
    whole and flagged by the n_runs column, so a short cell is visible in the
    table rather than silently padded.
    """
    if not n:
        return df
    return (df.sort_values(["container", "metric", "p", "seed"])
              .groupby(["container", "metric", "p"], group_keys=False)
              .head(n))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (container, metric, p).

    D statistics are taken over feasible runs only -- an infeasible run has no
    meaningful radius -- while n_runs and success_rate describe all attempts,
    so a cell that mostly fails is visible rather than hidden.
    """
    keys = ["container", "metric", "p"]
    ok = df[df.feasible]

    d = (ok.groupby(keys)
           .agg(n_ok=("D", "size"),
                D_min=("D", "min"), D_max=("D", "max"),
                D_mean=("D", "mean"), D_median=("D", "median"), D_std=("D", "std"),
                t_min=("wall_time_s", "min"), t_max=("wall_time_s", "max"),
                t_mean=("wall_time_s", "mean"), t_median=("wall_time_s", "median"),
                t_std=("wall_time_s", "std")))

    allruns = df.groupby(keys).agg(n_runs=("D", "size"), success_rate=("feasible", "mean"))
    out = allruns.join(d, how="left").reset_index()

    # std is undefined for a single run; report 0 rather than NaN
    for c in ("D_std", "t_std"):
        out[c] = out[c].fillna(0.0)
    out["D_cv"] = np.where(out.D_mean > 0, out.D_std / out.D_mean, np.nan)
    return out.sort_values(keys).reset_index(drop=True)


def monotonicity(stats: pd.DataFrame, tol: float = 1e-8) -> pd.DataFrame:
    """D(p) must be non-increasing: a feasible (p+1)-configuration restricts to
    a feasible p-configuration with at least the same radius."""
    rows = []
    for (c, m), g in stats.sort_values("p").groupby(["container", "metric"]):
        g = g.dropna(subset=["D_max"])
        ps, ds = g.p.values, g.D_max.values
        for i in range(len(ds) - 1):
            if ds[i + 1] > ds[i] + tol:
                rows.append({"container": c, "metric": m, "p": ps[i], "D_p": ds[i],
                             "p_next": ps[i + 1], "D_p_next": ds[i + 1],
                             "increase_pct": (ds[i + 1] - ds[i]) / ds[i] * 100})
    return pd.DataFrame(rows)


def render_figures(df: pd.DataFrame, stats: pd.DataFrame, out_dir: str, output_root: str,
                   elev: float, azim: float) -> None:
    """One clean figure per cell, from the stored solution of its best feasible seed.

    The seed is chosen from the sweep rows, not by scanning the solution files:
    infeasible runs store a file too, often with a larger D, so the largest D
    on disk was not necessarily a solution at all.
    """
    os.makedirs(out_dir, exist_ok=True)
    ok = df[df.feasible]
    made = 0
    for _, r in stats.iterrows():
        cell_rows = ok[(ok.container == r.container) & (ok.metric == r.metric) & (ok.p == r.p)]
        if cell_rows.empty:
            continue
        seed = int(cell_rows.loc[cell_rows["D"].idxmax(), "seed"])
        best = os.path.join(output_root, r.container, f"p{int(r.p)}", r.metric.lower(),
                            f"seed{seed}.npz")
        if not os.path.exists(best):
            continue
        dest = os.path.join(out_dir, f"{r.container}_p{int(r.p)}_{r.metric.lower()}.png")
        subprocess.run([sys.executable, "-m", "tsgo.experiments.render", best,
                        "--clean", "--elev", str(elev), "--azim", str(azim),
                        "--out", dest], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        made += 1
    print(f"[stats] wrote {made} figures -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summary table (and figures) for a sweep.")
    ap.add_argument("--sweep", default="results/sweep.csv", help="CSV written by tsgo-batch.")
    ap.add_argument("--output-root", dest="output_root", default="output",
                    help="Where the per-run solutions live.")
    ap.add_argument("--container", nargs="*", default=None, help="Restrict to these containers.")
    ap.add_argument("--metric", nargs="*", default=None, help="Restrict to these metrics.")
    ap.add_argument("--csv", default=None, help="Also write the table as a tidy CSV.")
    ap.add_argument("--latex", default=None, help="Also write a LaTeX table (booktabs).")
    ap.add_argument("--figures", default=None, help="Also render one clean PNG per cell into this directory.")
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--azim", type=float, default=135.0)
    ap.add_argument("--decimals", type=int, default=8, help="Decimals for D in the printed/LaTeX table.")
    ap.add_argument("--first-n", dest="first_n", type=int, default=None,
                    help="Use only the first N seeds per cell, so every row rests on the "
                         "same number of runs. Recommended for anything reported in the paper.")
    args = ap.parse_args()

    df = load(args.sweep)
    if args.container:
        df = df[df.container.isin(args.container)]
    if args.metric:
        df = df[df.metric.isin([m.upper() for m in args.metric])]
    if df.empty:
        raise SystemExit("no rows after filtering")

    if args.first_n:
        before = len(df)
        df = take_first_n(df, args.first_n)
        print(f"[stats] standardising on the first {args.first_n} seeds per cell "
              f"({before} rows -> {len(df)})")

    stats = summarise(df)

    if args.first_n:
        short = stats[stats.n_runs < args.first_n]
        if not short.empty:
            print(f"[stats] WARNING: {len(short)} cell(s) have fewer than {args.first_n} runs "
                  f"and are NOT comparable with the rest:")
            for _, r in short.iterrows():
                print(f"           {r.container:<13}{r.metric:<10}p={int(r.p):<4}"
                      f"{int(r.n_runs)} run(s)")

    pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 400)
    # The reported table; the CSV holds the same columns at full precision.
    cols = ["container", "metric", "p", "n_runs",
            "D_mean", "D_median", "D_std", "t_mean", "t_median", "t_std"]
    show = stats.copy()
    for c in ("D_mean", "D_median", "D_std"):
        show[c] = show[c].map(lambda v: f"{v:.{args.decimals}f}" if pd.notna(v) else "--")
    for c in ("t_mean", "t_median", "t_std"):
        show[c] = show[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "--")
    print(show[cols].to_string(index=False))

    viol = monotonicity(stats)
    print()
    if viol.empty:
        print("[stats] monotonicity: OK, D(p) is non-increasing everywhere")
    else:
        print(f"[stats] monotonicity: {len(viol)} violation(s) -- the smaller p is under-optimised")
        print(viol.to_string(index=False))

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        stats[cols].to_csv(args.csv, index=False)
        print(f"[stats] table -> {args.csv}")

    if args.latex:
        os.makedirs(os.path.dirname(os.path.abspath(args.latex)) or ".", exist_ok=True)
        # longtable: one row per cell runs to several pages.
        tex = (show[cols].to_latex(index=False, escape=True, na_rep="--", longtable=True,
                                   caption="Radius $D$ and wall time $t$ (seconds) by "
                                           "container, metric and $p$: mean, median and "
                                           "sample standard deviation over the cell's runs. "
                                           "$D$ statistics are over feasible runs.",
                                   label="tab:sweep_summary"))
        with open(args.latex, "w") as fh:
            fh.write(tex)
        print(f"[stats] LaTeX -> {args.latex}")

    if args.figures:
        render_figures(df, stats, args.figures, args.output_root, args.elev, args.azim)


if __name__ == "__main__":
    main()
