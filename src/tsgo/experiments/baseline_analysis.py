"""Analysis for paper/BASELINE_PROTOCOL.md, sections 5 and 6 (experiments A, B, E).

    python -m tsgo.experiments.baseline_analysis [--dir results/baseline]

Reads every finished trial, scores it by what it completed within its budget,
and writes CSV tables and a summary to <dir>/analysis/.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

SOLVERS = ("tsgo", "ipopt", "slsqp")
TIE = 1e-6


def load(d: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(os.path.join(d, "*", "*.json")):
        r = json.load(open(f))
        H = r["budget"]
        within = [(t, b) for t, b in r["trace"] if t <= H]
        rows.append({"solver": r["solver"], "container": r["container"], "metric": r["metric"],
                     "p": r["p"], "seed": r["seed"], "budget": H,
                     "D": within[-1][1] if within else 0.0,
                     "starts_in_budget": len(within), "trace": within,
                     "solves": r.get("solves", [])})
    return pd.DataFrame(rows)


def published():
    ref = json.load(open("src/tsgo/experiments/data/reference_D.json"))
    out = {}
    for c, md in ref.items():
        if c.startswith("_"):
            continue
        for m, spec in md.items():
            # Protocol section 5: the published value is the target wherever one
            # exists, proven optimal or best known.
            for p, v in spec["values"].items():
                out[(c, m, int(p))] = float(v)
    return out


def time_to(trace, target):
    for t, d in trace:
        if d >= target:
            return t
    return np.inf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/baseline")
    args = ap.parse_args()
    out = os.path.join(args.dir, "analysis")
    os.makedirs(out, exist_ok=True)

    # Gurobi runs on 6 of the 45 cells only (l2, convex), so it would skew
    # per-cell comparisons such as "fastest solver"; it is reported separately.
    df = load(args.dir)
    df = df[df.solver.isin(SOLVERS)]
    key = ["container", "metric", "p"]
    pub = published()

    # ---- B: quality at equal budget -----------------------------------------
    g = df.groupby(key + ["solver"]).D
    cell = pd.DataFrame({"best": g.max(), "median": g.median(), "q25": g.quantile(.25),
                         "q75": g.quantile(.75), "trials": g.size()}).unstack("solver")
    cell.to_csv(os.path.join(out, "B_cells.csv"))

    lines = ["# Baseline comparison: results", "",
             f"Trials read: {len(df)} of 1350. Scored by what each trial completed within its budget.", ""]
    from scipy.stats import wilcoxon
    lines += ["## B. Quality at equal wall time (best of 10, certified D)", "",
              "| against | TSGO better | tied | TSGO worse | Wilcoxon p (two-sided) | median TSGO/baseline |",
              "|---|---|---|---|---|---|"]
    for b in ("ipopt", "slsqp"):
        both = cell["best"][["tsgo", b]].dropna()
        r = both["tsgo"] / both[b].replace(0, np.nan)
        diff = (both["tsgo"] - both[b]) / both[["tsgo", b]].max(axis=1)
        win, loss = int((diff > TIE).sum()), int((diff < -TIE).sum())
        tie = len(diff) - win - loss
        try:
            pval = wilcoxon(both["tsgo"], both[b]).pvalue if win + loss else float("nan")
        except ValueError:
            pval = float("nan")
        lines.append(f"| {b} | {win} | {tie} | {loss} | {pval:.2g} | {np.nanmedian(r):.4f} |")
    lines.append("")
    for grp, sub in (("convex (cube, tetrahedron)", ["cube", "tetrahedron"]),
                     ("non-convex (h-box, star, cage)", ["h-box", "pointy-star", "cage"])):
        bb = cell["best"].loc[cell.index.get_level_values(0).isin(sub)]
        parts = []
        for b in ("ipopt", "slsqp"):
            diff = (bb["tsgo"] - bb[b]) / bb[["tsgo", b]].max(axis=1)
            parts.append(f"vs {b}: {int((diff > TIE).sum())} better / {int((diff.abs() <= TIE).sum())} tied / "
                         f"{int((diff < -TIE).sum())} worse")
        lines.append(f"- {grp}: " + "; ".join(parts))
    lines.append("")

    # ---- A: time to target ---------------------------------------------------
    best_any = df.groupby(key).D.max()
    tt_rows = []
    for (c, m, p), sub in df.groupby(key):
        Dref = pub.get((c, m, p), best_any[(c, m, p)])
        for frac in (0.99, 0.999):
            for s, ss in sub.groupby("solver"):
                times = [time_to(tr, frac * Dref) for tr in ss.trace]
                tt_rows.append({"container": c, "metric": m, "p": p, "solver": s, "target": frac,
                                "D_ref": Dref, "reached": int(np.isfinite(times).sum()),
                                "trials": len(times), "median_time": float(np.median(times))})
    tt = pd.DataFrame(tt_rows)
    tt.to_csv(os.path.join(out, "A_time_to_target.csv"), index=False)
    lines += ["## A. Time to target", "",
              "| target | solver | cells where most trials reach it | median of per-cell median time (s) |",
              "|---|---|---|---|"]
    for frac in (0.99, 0.999):
        for s in SOLVERS:
            x = tt[(tt.target == frac) & (tt.solver == s)]
            reached = x[np.isfinite(x.median_time)]
            lines.append(f"| {frac:.1%} | {s} | {len(reached)} of {len(x)} | "
                         f"{reached.median_time.median() if len(reached) else float('nan'):.1f} |")
    lines.append("")
    # performance profile at the 99.9% target
    prof = tt[tt.target == 0.999].pivot_table(index=["container", "metric", "p"], columns="solver",
                                               values="median_time")
    fastest = prof.min(axis=1)
    ratios = prof.div(fastest, axis=0)
    ratios.to_csv(os.path.join(out, "A_profile_ratios.csv"))
    lines += ["Performance profile at the 99.9% target: share of cells a solver reaches within "
              "tau times the fastest solver's median time.", "",
              "| solver | tau=1 | tau=2 | tau=10 | ever |", "|---|---|---|---|---|"]
    for s in SOLVERS:
        if s in ratios:
            r = ratios[s]
            lines.append(f"| {s} | {(r <= 1).mean():.0%} | {(r <= 2).mean():.0%} | "
                         f"{(r <= 10).mean():.0%} | {np.isfinite(r).mean():.0%} |")
    lines.append("")

    # ---- E: feasibility audit ------------------------------------------------
    e_rows = []
    # The local solvers only: Gurobi's incumbents carry no convergence flag.
    for _, r in df[df.solver.isin(["ipopt", "slsqp"])].iterrows():
        for s in r.solves:
            if s["converged"]:
                e_rows.append({"solver": r.solver, "container": r.container, "metric": r.metric,
                               "p": r.p, "reported": s["reported_D"], "certified": s["certified_D"]})
    e = pd.DataFrame(e_rows)
    if len(e):
        e["over_rel"] = (e.reported - e.certified) / e.reported.where(e.reported > 0)
        e.to_csv(os.path.join(out, "E_audit.csv"), index=False)
        lines += ["## E. Feasibility of reported radii (converged local solves)", "",
                  "| solver | converged solves | over-stated by >1e-6 | over-stated by >1% | "
                  "certified radius zero | worst over-statement |", "|---|---|---|---|---|---|"]
        for s, x in e.groupby("solver"):
            lines.append(f"| {s} | {len(x)} | {(x.over_rel > 1e-6).mean():.1%} | "
                         f"{(x.over_rel > 1e-2).mean():.1%} | {(x.certified <= 0).mean():.1%} | "
                         f"{x.over_rel.max():.2%} |")
        lines.append("")

    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
