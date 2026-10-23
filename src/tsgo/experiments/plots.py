"""Charts built from results/sweep.csv.

Separate from make_figures.py, which renders 3-D configurations. Everything in
the manifest's "plots" section is produced here, so a chart in the manuscript
can be traced to the runs behind it -- the previous runtime figure was a
committed PNG that no code reproduced.

Colours come from a validated categorical palette, assigned to containers in a
fixed order that never depends on which containers a particular chart happens to
include. Three of the five slots fall below 3:1 contrast on the chart surface,
so every series carries a direct label at the end of its line: identity is never
left to colour alone, which is also what keeps the chart readable in greyscale
and for colour-vision deficiency.
"""

import os
from typing import Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical palette, light surface. Worst adjacent CVD dE 9.1,
# worst adjacent normal-vision dE 19.6. Slots are assigned by container, in this
# order, so a chart that drops a container does not repaint the others.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
AXIS = "#c9c8c4"

# Print geometry of the manuscript (revtex4-1, two columns): charts are drawn at
# the size they print, so their text is never scaled. Measured from the
# compiled paper: \columnwidth = 246 pt, \textwidth = 510 pt.
COLUMN_IN = 246 / 72.27
FULL_IN = 510 / 72.27
FS_TICK, FS_LABEL, FS_NOTE = 6.5, 7.5, 7.0

CONTAINER_ORDER = ["cube", "tetrahedron", "h-box", "pointy-star", "cage"]
SERIES = {
    "cube":        "#2a78d6",   # slot 1 blue
    "tetrahedron": "#eb6834",   # slot 2 orange
    "h-box":       "#1baf7a",   # slot 3 aqua
    "pointy-star": "#eda100",   # slot 4 yellow
    "cage":        "#e87ba4",   # slot 5 magenta
}
LABEL = {
    "cube": "Cube", "tetrahedron": "Tetrahedron", "h-box": "H-box",
    "pointy-star": "Star", "cage": "Cage",
}
METRIC_LABEL = {
    "EUCLIDEAN": r"Euclidean ($\ell_2$)",
    "TAXICAB": r"Taxicab ($\ell_1$)",
    "SUP_NORM": r"Sup norm ($\ell_\infty$)",
}


def _style(ax) -> None:
    """Recessive chrome: solid hairline grid a shade off the surface."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.35, linestyle="-", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, which="both", length=2.5, width=0.6, labelsize=FS_TICK)
    ax.set_axisbelow(True)


def runtime_chart(df: pd.DataFrame, metric: str, out_path: str,
                  containers: Optional[Sequence[str]] = None,
                  dpi: int = 300, figsize=(FULL_IN * 0.62, 2.6)) -> Dict[str, dict]:
    """Mean wall time per run against p, one line per container.

    Log-log, because the quantity of interest is how cost scales rather than its
    value at any one p: three orders of magnitude in time and two in p make a
    linear pair of axes unreadable at the bottom left, and on log-log the growth
    rate is the slope of the line.

    Returns the plotted series, so the caller can record provenance or emit the
    table view that the palette's contrast warning requires as an alternative to
    the direct labels used here.
    """
    d = df[(df["metric"] == metric)]
    if containers is None:
        containers = [c for c in CONTAINER_ORDER if c in set(d["container"])]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    plotted: Dict[str, dict] = {}
    for name in containers:
        sub = d[d["container"] == name]
        if sub.empty:
            continue
        g = (sub.groupby("p")["wall_time_s"]
                .agg(mean="mean", n="count")
                .reset_index()
                .sort_values("p"))
        colour = SERIES[name]
        ax.plot(g["p"], g["mean"], color=colour, linewidth=1.3, zorder=3,
                solid_capstyle="round")
        # A 2px surface ring keeps markers legible where two series cross.
        ax.plot(g["p"], g["mean"], linestyle="none", marker="o", markersize=3.0,
                markerfacecolor=colour, markeredgecolor=SURFACE,
                markeredgewidth=0.7, zorder=4)
        plotted[name] = {"p": g["p"].tolist(), "mean_s": g["mean"].round(3).tolist(),
                         "runs": g["n"].tolist()}

    # A legend rather than end-of-line labels: four lines finish within a
    # factor of three of each other at p = 50, too close to label legibly.
    handles = [plt.Line2D([], [], color=SERIES[n], linewidth=1.3, marker="o", markersize=3,
                          markeredgecolor=SURFACE, markeredgewidth=0.6) for n in plotted]
    ax.legend(handles, [LABEL[n] for n in plotted], fontsize=FS_NOTE, frameon=False,
              loc="upper left", handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Dispersion points $p$", fontsize=FS_LABEL, color=INK_SOFT, labelpad=3)
    ax.set_ylabel("Mean wall time per run (s)", fontsize=FS_LABEL, color=INK_SOFT, labelpad=3)

    ticks = [1, 2, 3, 5, 10, 20, 30, 50, 100]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1000"])


    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return plotted


REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "data", "reference_D.json")


def load_reference() -> dict:
    import json
    with open(REFERENCE_PATH) as fh:
        return json.load(fh)


def load_nlp_baseline(parts_dir: str = "results/nlp_parts") -> pd.DataFrame:
    """Best feasible radius per cell from the multi-start NLP baseline.

    L-BFGS-B and SLSQP are pooled: both are restarted from fresh random
    configurations until the cell's time budget expires, and the baseline's
    answer is the best feasible radius either of them reached.
    """
    import glob
    files = sorted(glob.glob(os.path.join(parts_dir, "*.csv")))
    if not files:
        return pd.DataFrame(columns=["container", "metric", "p", "D", "feasible"])
    df = pd.concat([pd.read_csv(f, skipinitialspace=True) for f in files],
                   ignore_index=True)
    df = df[pd.to_numeric(df["p"], errors="coerce").notna()]
    df["p"] = df["p"].astype(int)
    df["D"] = pd.to_numeric(df["D"], errors="coerce")
    df["feasible"] = df["feasible"].astype(str).str.strip().str.lower() == "true"
    return df


def comparison_chart(sweep: pd.DataFrame, container: str, metric: str,
                     out_path: str, nlp: Optional[pd.DataFrame] = None,
                     dpi: int = 300, figsize=(COLUMN_IN, 2.2)) -> Dict[str, dict]:
    """Our best radius against the published reference for this container.

    Two readings, one per kind of reference:
      cube (Packomania records, which we reach): the relative gap
        (D_ref - D) / D_ref in percent, over p. Zero is "matches the record".
      tetrahedron (Kazakov et al., which we exceed): the radii themselves --
        the benchmark, ours, and the optimum Gurobi proves -- over p, so the
        three can be compared directly.
    """
    ref = (load_reference().get(container, {}) or {}).get(metric) or {}
    values = {int(k): float(v) for k, v in ref.get("values", {}).items()}
    ours = (sweep[(sweep["container"] == container) & (sweep["metric"] == metric)
                  & (sweep["feasible"] == True)]                      # noqa: E712
            .groupby("p")["D"].max())
    ps = sorted(p for p in values if p in ours.index)
    gap = {p: (values[p] - float(ours[p])) / values[p] for p in ps}
    x = np.arange(len(ps))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.set_xticks(x, [str(p) for p in ps])
    ax.set_xlabel("Dispersion points $p$", fontsize=FS_LABEL, color=INK_SOFT, labelpad=3)

    if np.median(list(gap.values())) >= 0:          # we match the reference
        ys = [100.0 * gap[p] for p in ps]
        ax.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)
        ax.plot(x, ys, color=SERIES["cube"], linewidth=1.2, zorder=3)
        ax.plot(x, ys, linestyle="none", marker="o", markersize=3.4, zorder=4,
                markerfacecolor=SERIES["cube"], markeredgecolor=SURFACE, markeredgewidth=0.6)
        # Scaled to the gaps themselves: a fixed floor sized for gaps of 0.1%
        # flattens gaps of 0.0005% into the zero line.
        top = max(max(ys) * 1.3, 1e-6)
        ax.set_ylim(-0.08 * top, top)
        ax.set_ylabel(f"gap to {ref.get('label', 'reference')} (%)", fontsize=FS_LABEL,
                      color=INK_SOFT, labelpad=3)
        for xi, y, p in zip(x, ys, ps):            # label the gaps worth reading
            if y > 0.1 * top:
                ax.annotate(np.format_float_positional(y, precision=2, fractional=False, trim="-") + "%", (xi, y), xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=FS_NOTE - 0.5, color=INK)
    else:                                            # we exceed the reference
        opt = {}
        cert = "results/certificates/gurobi.csv"
        if os.path.exists(cert):
            c = pd.read_csv(cert)
            c = c[(c.container == container) & (c.status == "optimal")]
            opt = dict(zip(c.p.astype(int), c.gurobi_bound))
        bench = SERIES["tetrahedron"]                # slot 2 orange
        mine = SERIES["cube"]                        # slot 1 blue
        ax.plot(x, [values[p] for p in ps], linestyle="none", marker="o", markersize=4,
                markerfacecolor=bench, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3,
                label="Kazakov et al. (published)")
        ax.plot(x, [float(ours[p]) for p in ps], linestyle="none", marker="o", markersize=4,
                markerfacecolor=mine, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=4,
                label="This work")
        ox = [xi for xi, p in zip(x, ps) if p in opt]
        ax.plot(ox, [opt[p] for p in ps if p in opt], linestyle="none", marker="_",
                markersize=11, markeredgewidth=1.1, color=INK, zorder=5,
                label="Proven optimum (Gurobi)")
        ax.set_ylabel("radius $D$", fontsize=FS_LABEL, color=INK_SOFT, labelpad=3)
        ax.legend(fontsize=FS_NOTE, frameon=False, loc="upper right", handletextpad=0.3,
                  borderaxespad=0.2)

    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    ratio = [round(float(ours[p]) / values[p], 6) for p in ps]
    return {"This work": {"p": ps, "ratio": ratio, "gap": [gap[p] for p in ps]}}
