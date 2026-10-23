"""Packing figures for the Stoyan & Yaskov (2013) comparison.

    python -m tsgo.experiments.stoyan_yaskov_packings [--out paper/figures]

stoyan_yaskov_packings.png  each example at the most spheres we fit at radius
                            1, the best seed of that count; removed pieces
                            (holes, cones, prisms, dihedral cuts) in grey
stoyan_yaskov_ex6.png       example 6: a run caught in the thin wall under the
                            hole (ball in the wall in purple), and the same run
                            after the escape phase
                            (results/stoyan_yaskov_escape.csv); written only
                            once that file exists

Drawn with tsgo.plotting.raycast, so balls and obstacles occlude each other
correctly. Prints the run behind every panel.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsgo.container.stoyan_yaskov import PUBLISHED
from tsgo.experiments.make_figures import ball_colors
from tsgo.experiments.plots import FS_LABEL, FS_NOTE, FULL_IN, INK, INK_SOFT
from tsgo.experiments.render import load_solution
from tsgo.plotting.raycast import draw_edges, render

PALETTE = {"color_by": "height", "cmap": ["#9bb7f5", "#4169e1", "#841ae8"], "cmap_range": [0, 1]}
TOL = 1e-9
ELEV, AZIM = 24, -130
WALL = 0.93364          # inradius of example 6's bottom wall under the hole


def fitted(df: pd.DataFrame, k: int):
    """(p, seed, D): the most spheres fitted at radius 1, best seed of that count."""
    d = df[(df.container == f"stoyan-yaskov-{k}") & (df.D >= 1.0 - TOL)]
    p = int(d.p.max())
    r = d[d.p == p].sort_values("D").iloc[-1]
    return p, int(r.seed), float(r.D)


def panel(ax, path, colors=None, elev=ELEV, azim=AZIM, width=900):
    X, D, c, m, p = load_solution(path)
    img, cam, depth = render(X, D, c, colors if colors is not None else ball_colors(PALETTE, X),
                             elev=elev, azim=azim, width=width)
    ax.imshow(img, interpolation="lanczos")
    draw_edges(ax, c, cam, depth, lw=0.35)
    ax.set_xlim(0, cam["W"]); ax.set_ylim(cam["H"], 0)
    ax.axis("off")
    return X, D


def gallery(df: pd.DataFrame, out: str) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(FULL_IN, 3.9), facecolor="white")
    for ax, k in zip(axes.flat, range(1, 9)):
        p, seed, D = fitted(df, k)
        src = f"output/stoyan-yaskov-{k}/p{p}/euclidean/seed{seed}.npz"
        panel(ax, src)
        theirs = PUBLISHED[k][0]
        ax.set_title(f"Example {k}: {p} spheres", fontsize=FS_LABEL, color=INK, pad=1)
        ax.text(0.5, -0.02, f"published {theirs}", transform=ax.transAxes, ha="center", va="top",
                fontsize=FS_NOTE, color=INK_SOFT)
        print(f"  example {k}: p={p} seed={seed} D={D:.6f}  <- {src}")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.04, wspace=0.03, hspace=0.18)
    path = os.path.join(out, "stoyan_yaskov_packings.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def in_wall(X) -> np.ndarray:
    """Balls whose centre lies in example 6's bottom wall: directly under the
    tapered hole, a cone from (-10, 7, 3) through the rectangle y in [1, 9.43],
    z in [1, 6.83] on the face x = 9.39. At x its floor is z = 3 - 2 s and it
    spans y from 7 - 6 s to 7 + 2.43 s, with s = (x + 10) / 19.39."""
    X = np.asarray(X)
    s = (X[:, 0] + 10) / 19.39
    return (X[:, 2] < 3 - 2 * s) & (X[:, 1] > 7 - 6 * s) & (X[:, 1] < 7 + 2.43 * s)


def ex6(df: pd.DataFrame, esc: pd.DataFrame, out: str):
    trapped = esc[(esc.p == 30) & (np.abs(esc.D_before - WALL) < 1e-4) & (esc.D >= 1.0 - TOL)]
    if trapped.empty:
        print("  example 6: no trapped p = 30 run that the escape phase lifted to radius 1")
        return None
    seed = int(trapped.sort_values("D").iloc[-1].seed)
    before = f"output/stoyan-yaskov-6/p30/euclidean/seed{seed}.npz"
    after = f"output/variants/escape/stoyan-yaskov-6/p30/euclidean/seed{seed}.npz"
    panels = [(before, "p = 30, seed %d: stuck in the wall" % seed),
              (after, "the same run after the escape phase")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_IN * 0.8, 2.3), facecolor="white")
    for ax, (src, note) in zip(axes, panels):
        X, D, *_ = load_solution(src)
        cols = ["#9bb7f5"] * len(X)
        for i in np.flatnonzero(in_wall(X)):
            cols[i] = "#841ae8"
        panel(ax, src, colors=cols, elev=-15, azim=180)
        ax.set_title(f"radius {D:.5f}", fontsize=FS_LABEL, color=INK, pad=1)
        ax.text(0.5, -0.03, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=FS_NOTE, color=INK_SOFT)
        print(f"  example 6 panel: {src} D={D:.6f}, {int(in_wall(X).sum())} in the wall")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.09, wspace=0.04)
    path = os.path.join(out, "stoyan_yaskov_ex6.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/stoyan_yaskov.csv")
    ap.add_argument("--escape-csv", dest="escape_csv", default="results/stoyan_yaskov_escape.csv")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--only", choices=["gallery", "ex6"], default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df = df[(df.metric == "EUCLIDEAN") & (df.feasible.astype(str).str.lower() == "true")]
    if args.only in (None, "gallery"):
        print(f"[stoyan_yaskov_packings] wrote {gallery(df, args.out)}")
    if args.only in (None, "ex6") and os.path.exists(args.escape_csv):
        esc = pd.read_csv(args.escape_csv)
        esc = esc[esc.feasible.astype(str).str.lower() == "true"]
        path = ex6(df, esc, args.out)
        if path:
            print(f"[stoyan_yaskov_packings] wrote {path}")


if __name__ == "__main__":
    main()
