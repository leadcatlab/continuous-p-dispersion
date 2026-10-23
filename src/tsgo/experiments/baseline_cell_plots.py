"""One figure per cell of the equal-budget comparison: best certified D against time.

Reads the trials run_baseline.sh writes (results/baseline/<solver>/*.json) and
writes plots/<container>/p<p>/<metric>/d_vs_time.png, mirroring output/.

Top panel: best certified D so far against elapsed time, one thin line per
seed and the median over seeds in bold, for every solver that ran the cell;
Gurobi's proven upper bound is dashed. Bottom panel: the same medians as the
relative gap to the cell's reference radius, on a log scale, so differences
of 1e-6 and 1e-2 are both visible. The reference is the best-known value where
one exists (Packomania for the cube under l2, the stacked configuration for
the tetrahedron at a tetrahedral number), otherwise the best radius any run of
the final sweep or of this comparison found. Only what finished within the
cell's budget is drawn, as in the analysis.

    python -m tsgo.experiments.baseline_cell_plots [--trials results/baseline] [--out plots]
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SOLVERS = {"tsgo": ("TSGO", "#4169e1"), "ipopt": ("IPOPT", "#841ae8"),
           "slsqp": ("SLSQP", "#9bb7f5"), "gurobi": ("Gurobi", "#333333")}
REF_JSON = os.path.join(os.path.dirname(__file__), "data", "reference_D.json")


def known_optimum(container: str, metric: str, p: int):
    """(value, label) of a published or closed-form radius, or None."""
    if metric != "EUCLIDEAN":
        return None
    if container == "tetrahedron":
        for k in range(2, 20):                       # tetrahedral numbers
            if k * (k + 1) * (k + 2) // 6 == p:
                return 1.0 / (2 * (k - 1) + 2 * math.sqrt(6)), "stacked optimum"
    if container == "cube":
        vals = json.load(open(REF_JSON)).get("cube", {}).get("EUCLIDEAN", {}).get("values", {})
        if str(p) in vals:
            return float(vals[str(p)]), "Packomania"
    return None


def at(trace, grid):
    """Best-so-far at each time of `grid`; NaN before the first entry."""
    out = np.full(len(grid), np.nan)
    if not trace:
        return out
    t = np.array([a for a, _ in trace]); d = np.array([b for _, b in trace])
    idx = np.searchsorted(t, grid, side="right") - 1
    ok = idx >= 0
    out[ok] = np.maximum.accumulate(d)[idx[ok]]
    return out


def median_curve(traces, grid):
    """Median over seeds; NaN wherever fewer than half have a value yet."""
    if not traces:
        return np.full(len(grid), np.nan)
    V = np.vstack([at(tr, grid) for tr in traces])
    have = np.sum(~np.isnan(V), axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(np.isnan(V), -np.inf, V), axis=0)
    med[have < len(traces) / 2] = np.nan
    return med


def load_trials(root: str):
    cells = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(root, "*", "*.json")):
        r = json.load(open(f))
        H = float(r["budget"])
        r["trace"] = [(t, d) for t, d in r["trace"] if t <= H]
        r["bound_trace"] = [(t, b) for t, b in r.get("bound_trace", []) if t <= H]
        cells[(r["container"], r["metric"], int(r["p"]))][r["solver"]].append(r)
    return cells


def plot_cell(key, by_solver, sweep_best, out_root):
    container, metric, p = key
    H = max(r["budget"] for rs in by_solver.values() for r in rs)
    found = max([sweep_best.get(key, 0.0)] + [d for rs in by_solver.values()
                                              for r in rs for _, d in r["trace"]])
    known = known_optimum(container, metric, p)
    ref, ref_label = (known if known else (found, "best found (all runs)"))
    t_min = min([t for rs in by_solver.values() for r in rs for t, _ in r["trace"][:1]] or [0.1])
    grid = np.geomspace(max(min(t_min, 1.0) * 0.9, 1e-2), H, 400)

    fig, (ax, gx) = plt.subplots(2, 1, figsize=(7.5, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
    lows = []
    for solver, (label, colour) in SOLVERS.items():
        rs = sorted(by_solver.get(solver, []), key=lambda r: r["seed"])
        if not rs:
            continue
        for r in rs:
            v = at(r["trace"], grid)
            ax.step(grid, v, where="post", color=colour, alpha=0.22, lw=0.8)
        med = median_curve([r["trace"] for r in rs], grid)
        ax.step(grid, med, where="post", color=colour, lw=2.2, label=f"{label} (median of {len(rs)})")
        # The top panel frames the endgame: its floor is the lowest median
        # from 2% of the budget on (the bottom panel shows the whole history).
        late = med[(grid >= 0.02 * H) & (med > 0)]
        if late.size and np.any(~np.isnan(late)):
            lows.append(np.nanmin(late))
        gap = np.clip((ref - med) / ref, 1e-10, None)
        gx.step(grid, gap, where="post", color=colour, lw=2.0)
        bounds = [r["bound_trace"] for r in rs if r.get("bound_trace")]
        if bounds:
            bmed = np.nanmedian(np.vstack([at([(t, -b) for t, b in bt], grid) for bt in bounds]), axis=0)
            final = -bmed[~np.isnan(bmed)][-1] if np.any(~np.isnan(bmed)) else float("nan")
            ax.step(grid, -bmed, where="post", color=colour, lw=1.4, ls="--",
                    label=f"{label} proven upper bound (median; final {final:.6f})")

    ax.axhline(ref, color="#555555", lw=1.0, ls=":", label=f"{ref_label}: {ref:.8f}")
    lo = min(lows) if lows else ref * 0.9
    span = max(ref - lo, ref * 1e-4)
    ax.set_ylim(max(0.0, lo - 0.08 * span), ref + 0.25 * span)
    ax.set_ylabel("best certified radius D")
    ax.set_title(f"{container}, {metric.lower()}, p = {p}   (budget {H:.0f} s per trial)")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.25)
    gx.set_xscale("log"); gx.set_yscale("log")
    gx.set_xlabel("elapsed time (s), one core")
    gx.set_ylabel("gap to reference\n(ref - D) / ref, floor 1e-10")
    gx.axvline(H, color="#999999", lw=0.8)
    gx.grid(alpha=0.25, which="both")
    gx.set_xlim(grid[0], H * 1.05)

    dest = os.path.join(out_root, container, f"p{p}", metric.lower(), "d_vs_time.png")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", default="results/baseline")
    ap.add_argument("--sweep", default="results/sweep.csv")
    ap.add_argument("--out", default="plots")
    a = ap.parse_args()
    cells = load_trials(a.trials)
    sweep = pd.read_csv(a.sweep)
    sweep_best = {(c, m, int(p)): float(d) for (c, m, p), d in
                  sweep[sweep.feasible].groupby(["container", "metric", "p"]).D.max().items()}
    for key in sorted(cells):
        print(plot_cell(key, cells[key], sweep_best, a.out))
    print(f"[plots] {len(cells)} cells -> {a.out}/")


if __name__ == "__main__":
    main()
