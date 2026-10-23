"""Figure and table for the Stoyan & Yaskov (2013) comparison (run_stoyan_yaskov.sh).

    python -m tsgo.experiments.stoyan_yaskov_plot [--csv results/stoyan_yaskov.csv] [--out paper/figures]

One panel per example, l2: the best certified radius over 10 seeds at each
sphere count tried, against the radius 1 at which they pack. A count is fitted
when some seed reaches radius 1; their published count is marked. Also prints
the numbers the paper's table quotes, including the time each count was first
fitted, from the progress trace each run file keeps.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsgo.container.stoyan_yaskov import PUBLISHED
from tsgo.experiments.plots import (AXIS, FS_LABEL, FS_NOTE, FS_TICK, FULL_IN, INK, INK_SOFT,
                                    SURFACE, _style)

FIT = "#2a78d6"      # count fitted at radius 1
MISS = "#a8a7a2"     # count not fitted
TOL = 1e-9


def first_fit_minutes(k: int, p: int) -> float:
    """Earliest time, over the seeds, at which a run held radius >= 1."""
    times = []
    for f in glob.glob(f"output/stoyan-yaskov-{k}/p{p}/euclidean/seed*.npz"):
        with np.load(f) as z:
            pr = z["progress"] if "progress" in z.files else np.zeros((0, 2))
        hit = pr[pr[:, 1] >= 1.0 - TOL, 0] if len(pr) else []
        if len(hit):
            times.append(float(hit[0]))
    return min(times) / 60 if times else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/stoyan_yaskov.csv")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df = df[(df.metric == "EUCLIDEAN") & (df.feasible.astype(str).str.lower() == "true")]

    fig, axes = plt.subplots(2, 4, figsize=(FULL_IN, 3.1), facecolor=SURFACE)
    rows = []
    for ax, k in zip(axes.flat, range(1, 9)):
        theirs, minutes = PUBLISHED[k]
        best = df[df.container == f"stoyan-yaskov-{k}"].groupby("p").D.max().sort_index()
        ours = int(best[best >= 1.0 - TOL].index.max())
        _style(ax)
        ax.axhline(1.0, color=INK_SOFT, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        ax.plot(best.index, best.values, color=AXIS, linewidth=1.0, zorder=2)
        fit = best >= 1.0 - TOL
        ax.scatter(best.index[fit], best.values[fit], s=14, color=FIT, zorder=3, linewidths=0)
        ax.scatter(best.index[~fit], best.values[~fit], s=14, color=MISS, zorder=3, linewidths=0)
        ax.scatter([theirs], [best[theirs]], s=46, facecolor="none", edgecolor=INK, linewidth=0.9,
                   zorder=4)
        ax.set_title(f"Example {k}: {theirs} $\\to$ {ours}", fontsize=FS_LABEL, color=INK, loc="left",
                     pad=3)
        ax.set_xticks(sorted({best.index.min(), theirs, ours, best.index.max()}))
        ax.tick_params(axis="x", labelsize=FS_TICK)
        rows.append({"example": k, "theirs": theirs, "ours": ours, "margin": ours - theirs,
                     "D_at_theirs": round(float(best[theirs]), 4),
                     "first_fit_at_theirs_min": round(first_fit_minutes(k, theirs), 2),
                     "their_minutes": minutes})
    for ax in axes[1]:
        ax.set_xlabel("spheres $p$", fontsize=FS_LABEL, color=INK_SOFT, labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("best radius", fontsize=FS_LABEL, color=INK_SOFT)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=FIT, markersize=4),
               plt.Line2D([], [], marker="o", linestyle="", color=MISS, markersize=4),
               plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                          markeredgecolor=INK, markersize=6),
               plt.Line2D([], [], color=INK_SOFT, linestyle=(0, (3, 2)), linewidth=0.8)]
    fig.legend(handles, ["fitted (radius $\\geq 1$)", "not fitted", "published count",
                         "radius 1"], loc="upper center", ncol=4, frameon=False,
               fontsize=FS_NOTE, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.93), h_pad=1.2, w_pad=1.0)
    path = os.path.join(args.out, "stoyan_yaskov.png")
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"[stoyan_yaskov_plot] wrote {path}")


if __name__ == "__main__":
    main()
