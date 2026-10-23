"""Per-cell stability summary over every run made in the sweep's configuration.

Reads the gold sweep (results/sweep.csv), which holds every run of every cell,
and writes one row per (container, metric, p) to results/summary_final.csv.

D statistics are over feasible runs; time statistics over all runs.
within1pct counts runs within 1% of the cell's best D, at_best those within a
relative 1e-6 of it. A cell rests on every seed it has, so n_runs varies.

    python -m tsgo.experiments.summary_final
"""

import argparse
import os

import pandas as pd

from tsgo.experiments.stats import load

KEYS = ["container", "metric", "p"]


def _cell(g: pd.DataFrame) -> pd.Series:
    D, t = g.loc[g.feasible, "D"], g.wall_time_s
    best = D.max()
    return pd.Series({
        "n_runs": len(g), "D_best": best, "D_mean": D.mean(), "D_median": D.median(),
        "D_std": D.std(), "D_min": D.min(), "t_mean": t.mean(), "t_median": t.median(),
        "t_std": t.std(), "t_min": t.min(), "t_max": t.max(),
        "D_cv": D.std() / D.mean(), "t_cv": t.std() / t.mean(),
        "within1pct": int((D >= 0.99 * best).sum()),
        "at_best": int((D >= best * (1 - 1e-6)).sum()),
    })


def build(sweep: str) -> pd.DataFrame:
    df = load(sweep)
    df["p"] = pd.to_numeric(df["p"]).astype(int)
    df["seed"] = pd.to_numeric(df["seed"]).astype(int)
    out = df.groupby(KEYS).apply(_cell).reset_index()
    for c in ("n_runs", "within1pct", "at_best"):
        out[c] = out[c].astype(int)
    return out.sort_values(KEYS).reset_index(drop=True)


def write_aligned(out: pd.DataFrame, path: str) -> None:
    """Column-padded CSV, as the sweep files are: readable by eye, and parsed
    with skipinitialspace=True plus stripping."""
    cols = list(out.columns)
    text = {c: [repr(v) if isinstance(v, float) else str(v) for v in out[c]] for c in cols}
    width = {c: max(len(c), *(len(s) for s in text[c])) for c in cols}
    right = {c for c in cols if out[c].dtype.kind == "i"}
    fmt = lambda c, s: s.rjust(width[c]) if c in right else s.ljust(width[c])
    lines = [", ".join(fmt(c, c) for c in cols)]
    lines += [", ".join(fmt(c, text[c][i]) for c in cols) for i in range(len(out))]
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-cell stability summary.")
    ap.add_argument("--sweep", default="results/sweep.csv")
    ap.add_argument("--out", default="results/summary_final.csv")
    args = ap.parse_args()
    out = build(args.sweep)
    write_aligned(out, args.out)
    print(f"[summary] {len(out)} cells, {int(out.n_runs.sum())} runs -> {args.out}")


if __name__ == "__main__":
    main()
