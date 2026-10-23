"""Figures for the equal-budget baseline comparison (paper/BASELINE_PROTOCOL.md).

    python -m tsgo.experiments.baseline_plots [--dir results/baseline] [--out paper/figures]

Three figures, all from the trial files and scored, as in the analysis, by what
each trial completed within its budget:

  baseline_quality.png   TSGO's best radius over each baseline's, every cell
  baseline_anytime.png   best radius against elapsed time on four hard cells
  baseline_outcomes.png  per solver: fastest / reaches later / never, at 99.9%

Solvers take slots 1-3 of the shared palette (the only three that validate on
every pair, in both modes): TSGO blue, IPOPT orange, SLSQP aqua. Aqua is below
3:1 contrast on the surface, so every series is also labelled directly and the
quality chart adds marker shape, for metric.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsgo.experiments.baseline_analysis import load, published
from tsgo.experiments.plots import (AXIS, COLUMN_IN, CONTAINER_ORDER, FS_LABEL, FS_NOTE,
                                    FS_TICK, FULL_IN, GRID, INK, INK_SOFT, LABEL, SURFACE,
                                    _style)

SOLVER_COLOR = {"tsgo": "#2a78d6", "ipopt": "#eb6834", "slsqp": "#1baf7a"}
SOLVER_LABEL = {"tsgo": "TSGO", "ipopt": "IPOPT", "slsqp": "SLSQP"}
METRIC_MARK = {"EUCLIDEAN": "o", "TAXICAB": "s", "SUP_NORM": "^"}
METRIC_SHORT = {"EUCLIDEAN": r"$\ell_2$", "TAXICAB": r"$\ell_1$", "SUP_NORM": r"$\ell_\infty$"}


def quality(df: pd.DataFrame, path: str) -> None:
    """TSGO's advantage over each baseline in every cell, as lollipops rising
    from 'tie'. One panel per container in a 3 + 2 grid; within a panel the
    cells run p = 10, 20, 40, and within each p the three metrics."""
    best = df.groupby(["container", "metric", "p", "solver"])["D"].max()
    fig, axes = plt.subplots(2, 3, figsize=(FULL_IN, 4.0), sharey=True, facecolor=SURFACE)
    ps, metrics = [10, 20, 40], list(METRIC_MARK)
    ticks = [1, 1.25, 1.5, 2, 3]
    for ax, c in zip(axes.flat, CONTAINER_ORDER):
        _style(ax)
        ax.grid(False, axis="x")
        ax.axhline(1.0, color=INK_SOFT, linewidth=0.8, zorder=1)
        xt, xl = [], []
        for gi, p in enumerate(ps):
            if gi:
                ax.axvline(gi * 4 - 1, color=GRID, linewidth=0.8, zorder=0)
            for mi, m in enumerate(metrics):
                x0 = gi * 4 + mi
                xt.append(x0); xl.append(METRIC_SHORT[m])
                for b, dx in (("ipopt", -0.18), ("slsqp", 0.18)):
                    r = best[(c, m, p, "tsgo")] / best[(c, m, p, b)]
                    ax.plot([x0 + dx, x0 + dx], [1.0, r], color=SOLVER_COLOR[b], linewidth=1.2,
                            zorder=2, solid_capstyle="butt")
                    ax.plot([x0 + dx], [r], marker="o", markersize=3.6, color=SOLVER_COLOR[b],
                            markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=3)
            ax.text(gi * 4 + 1, -0.2, f"$p = {p}$", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=FS_LABEL, color=INK_SOFT)
        ax.set_xticks(xt, xl, fontsize=FS_TICK)
        ax.set_xlim(-0.7, 10.7)
        ax.set_title(LABEL[c], fontsize=FS_LABEL + 0.5, color=INK, loc="left", pad=3)
        ax.set_yscale("log")
        ax.set_ylim(0.97, 3.3)
        ax.set_yticks(ticks, [f"+{100 * (t - 1):.0f}%" for t in ticks])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for ax in axes[:, 0]:
        ax.set_ylabel("TSGO's best radius,\nabove the baseline's", fontsize=FS_LABEL,
                      color=INK_SOFT)
    # The sixth slot carries the key.
    key = axes.flat[5]
    key.axis("off")
    for i, b in enumerate(("ipopt", "slsqp")):
        y = 0.78 - 0.16 * i
        key.plot([0.06, 0.06], [y - 0.05, y + 0.05], color=SOLVER_COLOR[b], linewidth=1.2,
                 transform=key.transAxes)
        key.plot([0.06], [y + 0.05], marker="o", markersize=3.6, color=SOLVER_COLOR[b],
                 transform=key.transAxes, markeredgecolor=SURFACE, markeredgewidth=0.5)
        key.text(0.12, y, f"against {SOLVER_LABEL[b]}", transform=key.transAxes,
                 fontsize=FS_LABEL, color=INK, va="center")
    key.text(0.02, 0.4, "Each stem is one cell: how much larger\nTSGO's best radius is than the\n"
             "baseline's (best of 10 trials, same\nwall time). On the line: a tie.",
             transform=key.transAxes, fontsize=FS_NOTE, color=INK_SOFT, va="top")
    fig.tight_layout(pad=0.5, h_pad=2.2, w_pad=1.0)
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def _median_curve(traces, grid):
    """Median over trials of best-so-far D on a time grid (step functions)."""
    vals = np.zeros((len(traces), len(grid)))
    for i, tr in enumerate(traces):
        if not tr:
            continue
        t = np.array([a for a, _ in tr]); d = np.array([b for _, b in tr])
        idx = np.searchsorted(t, grid, side="right") - 1
        vals[i] = np.where(idx >= 0, d[np.clip(idx, 0, None)], 0.0)
    return np.median(vals, axis=0), np.percentile(vals, 25, axis=0), np.percentile(vals, 75, axis=0)


def _spread(ends, gap):
    """End-of-line label positions, nudged apart so no two sit closer than
    `gap` (in data units); order is kept, so each label stays by its line."""
    ends = sorted(ends)
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - 1.0 if ys else 0.0
    if over > 0:                                # pushed off the top: shift the stack down
        ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    return [(y, e[1]) for y, e in zip(ys, ends)]


def anytime(df: pd.DataFrame, path: str, cells) -> None:
    """Median best radius against elapsed time on four hard cells, 2 x 2."""
    pub = published()
    fig, axes = plt.subplots(2, 2, figsize=(FULL_IN * 0.8, 3.9), facecolor=SURFACE)
    for ax, (c, m, p) in zip(axes.flat, cells):
        _style(ax)
        sub = df[(df.container == c) & (df.metric == m) & (df.p == p)]
        H = float(sub.budget.iloc[0])
        Dref = pub.get((c, m, p), sub.D.max())
        grid = np.linspace(0, H, 400)
        for s in ("ipopt", "slsqp", "tsgo"):
            med, _, _ = _median_curve(list(sub[sub.solver == s].trace), grid)
            ax.step(grid, med / Dref, where="post", color=SOLVER_COLOR[s], linewidth=1.4,
                    zorder=3, label=SOLVER_LABEL[s])
        ax.set_xlim(0, H)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{LABEL[c]}, {METRIC_SHORT[m]}, $p={p}$", fontsize=FS_LABEL + 0.5,
                     color=INK, loc="left", pad=3)
    for ax in axes[1]:
        ax.set_xlabel("elapsed time (s)", fontsize=FS_LABEL, color=INK_SOFT, labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("best $D$ / best known", fontsize=FS_LABEL, color=INK_SOFT)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    order = [labels.index(SOLVER_LABEL[s]) for s in ("tsgo", "slsqp", "ipopt")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
               ncol=3, frameon=False, fontsize=FS_LABEL, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94), h_pad=1.4, w_pad=1.6)
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def outcomes(ratios: pd.DataFrame, path: str) -> pd.DataFrame:
    """How each solver fares across the 45 cells at the 99.9% target: fastest
    to reach it, reaches it but not first, or does not reach it in the median
    trial. (The same information as a performance profile's two ends.)"""
    rows = []
    for s in ("tsgo", "slsqp", "ipopt"):
        r = ratios[s]
        fastest = int((r <= 1.0 + 1e-9).sum())
        reaches = int(np.isfinite(r).sum())
        rows.append((s, fastest, reaches - fastest, len(r) - reaches))
    t = pd.DataFrame(rows, columns=["solver", "fastest", "later", "never"]).set_index("solver")
    fig, ax = plt.subplots(figsize=(FULL_IN * 0.62, 1.55), facecolor=SURFACE)
    _style(ax)
    ax.grid(False, axis="y")
    segs = (("fastest", "fastest to the target", 1.0), ("later", "reaches it, not first", 0.45),
            ("never", "does not reach it", 0.0))
    for i, s in enumerate(t.index):
        left = 0
        for col, _, shade in segs:
            n = t.loc[s, col]
            colour = SOLVER_COLOR[s] if shade else "#e6e5e1"
            ax.barh(i, n, left=left, height=0.62, color=colour, alpha=shade if shade else 1.0,
                    edgecolor=SURFACE, linewidth=1.0, zorder=3)
            if n:
                ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=FS_NOTE,
                        color="white" if shade == 1.0 else INK, zorder=4)
            left += n
    ax.set_yticks(range(len(t)), [SOLVER_LABEL[s] for s in t.index], fontsize=FS_LABEL)
    ax.invert_yaxis()
    ax.set_xlim(0, 45)
    ax.set_xlabel("cells (of 45)", fontsize=FS_LABEL, color=INK_SOFT, labelpad=2)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return t


# End labels in panel (a) that would collide at the last p (SLSQP 1, IPOPT 0).
A_NUDGE = {"slsqp": 0.6, "ipopt": -0.6}


def reach(tt: pd.DataFrame, path: str, target: float = 0.999) -> pd.DataFrame:
    """Per p: (a) how many of its cells each solver's median trial brings to the
    target, and (b) the median time it takes in those cells. Time alone would
    hide that the baselines stop reaching the target as p grows, so it is drawn
    only where reached; panel (a) gives how many cells each point rests on."""
    tt = tt[np.isclose(tt.target, target)].copy()
    tt["hit"] = np.isfinite(tt.median_time)
    t = (tt.groupby(["p", "solver"])
           .agg(cells=("hit", "size"), reached=("hit", "sum"),
                time=("median_time", lambda s: s[np.isfinite(s)].median()))
           .reset_index())
    ps = sorted(t.p.unique())
    x = {p: i for i, p in enumerate(ps)}
    n_cells = int(t.cells.max())

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_IN * 0.78, 2.0), facecolor=SURFACE)
    for ax in (a, b):
        _style(ax)
        ax.set_xticks(range(len(ps)), [str(p) for p in ps], fontsize=FS_TICK)
        ax.set_xlim(-0.3, len(ps) - 1 + 0.75)
        ax.set_xlabel("$p$", fontsize=FS_LABEL, color=INK_SOFT, labelpad=2)
        ax.tick_params(axis="y", labelsize=FS_TICK)
    for s in ("tsgo", "slsqp", "ipopt"):
        g = t[t.solver == s].sort_values("p")
        xs = [x[p] for p in g.p]
        a.plot(xs, g.reached, color=SOLVER_COLOR[s], marker="o", markersize=4, linewidth=1.4,
               markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)
        a.text(xs[-1] + 0.12, g.reached.iloc[-1] + A_NUDGE.get(s, 0), SOLVER_LABEL[s],
               color=SOLVER_COLOR[s], fontsize=FS_NOTE, va="center", fontweight="bold")
        h = g[g.reached > 0]
        hx = [x[p] for p in h.p]
        b.plot(hx, h.time, color=SOLVER_COLOR[s], marker="o", markersize=4, linewidth=1.4,
               markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)
        if len(h):
            b.text(hx[-1] + 0.12, h.time.iloc[-1], SOLVER_LABEL[s], color=SOLVER_COLOR[s],
                   fontsize=FS_NOTE, va="center", fontweight="bold")
    a.set_ylim(0, n_cells + 0.5)
    a.set_yticks(range(0, n_cells + 1, 5))
    a.set_ylabel(f"cells reached (of {n_cells})", fontsize=FS_LABEL, color=INK_SOFT)
    a.set_title("(a) how often", fontsize=FS_LABEL, color=INK, loc="left")
    b.set_yscale("log")
    b.set_ylim(0.3, 2000)
    b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_ylabel("median time to target (s)", fontsize=FS_LABEL, color=INK_SOFT)
    b.set_title("(b) how fast, where reached", fontsize=FS_LABEL, color=INK, loc="left")
    fig.tight_layout(pad=0.4, w_pad=1.5)
    fig.savefig(path, dpi=300, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/baseline")
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    df = load(args.dir)
    quality(df, os.path.join(args.out, "baseline_quality.png"))
    anytime(df, os.path.join(args.out, "baseline_anytime.png"),
            [("cage", "TAXICAB", 40), ("pointy-star", "SUP_NORM", 40),
             ("h-box", "SUP_NORM", 40), ("cube", "EUCLIDEAN", 40)])
    ratios = pd.read_csv(os.path.join(args.dir, "analysis", "A_profile_ratios.csv"),
                         index_col=[0, 1, 2])
    t = outcomes(ratios, os.path.join(args.out, "baseline_outcomes.png"))
    print(t.to_string())
    tt = pd.read_csv(os.path.join(args.dir, "analysis", "A_time_to_target.csv"))
    print(reach(tt, os.path.join(args.out, "baseline_reach.png")).to_string())
    print("[baseline_plots] wrote baseline_quality, baseline_anytime, baseline_outcomes, baseline_reach")


if __name__ == "__main__":
    main()
