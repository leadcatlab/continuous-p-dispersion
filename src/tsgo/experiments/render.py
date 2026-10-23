"""Render a stored solution.

Takes either a .pkl or .npz written by tsgo-batch (or run-tsgo's save) and
draws the container with its spheres, without re-solving. Figures are
deterministic: sphere sampling uses a fixed seed and the camera is explicit.

    # look at a result
    tsgo-render output/cube/p10/euclidean/seed0.pkl --show

    # publication figure: no axes, no gridlines, no panes, transparent margin
    tsgo-render output/cube/p10/euclidean/seed0.pkl --clean --elev 22 --azim 135 \
        --out figures/cube_p10.png

    # a whole sweep at once
    tsgo-render "output/*/p*/euclidean/seed0.pkl" --clean --out-dir figures/
"""

import argparse
import glob
import os
import pickle
from typing import Optional, Tuple

import matplotlib
import numpy as np

from tsgo.container.sample_polyhedron import CONTAINERS


def load_solution(path: str):
    """Return (X, D, container, metric, p) from a solution file.

    .npz is the current format: one self-describing file per run, with the
    container stored by name and rehydrated here. .pkl is still read so that
    solutions saved by older runs, which embedded the Container object, keep
    working."""
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        container = CONTAINERS[str(d["container"])]()
        return d["X"], float(d["D"]), container, str(d["metric"]), int(d["p"])

    with open(path, "rb") as fh:
        d = pickle.load(fh)
    X = d.get("X_opt", d.get("optimized points"))
    D = d.get("D_opt", d.get("D"))
    container = d.get("container")
    if isinstance(container, str):
        container = CONTAINERS[container]()
    return np.asarray(X, dtype=float), float(D), container, d.get("metric", "EUCLIDEAN"), d.get("p", len(X))


def strip_chrome(ax) -> None:
    """Remove axes, ticks, gridlines and panes, leaving only the geometry.

    For figures that sit in the paper, where the surrounding matplotlib
    furniture is noise.
    """
    # Keep the aspect the plot was built with. set_box_aspect(None, ...)
    # below would otherwise reset it to Axes3D's 4:4:3 default, compressing z
    # by 25% and rendering spheres as visibly oblate ellipsoids.
    box_aspect = ax.get_box_aspect()

    ax.set_axis_off()
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticks([])
        axis.line.set_linewidth(0)
        axis.pane.set_visible(False)
        axis.pane.fill = False
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_title("")
    try:                       # matplotlib >= 3.6 keeps a little padding otherwise
        ax.set_box_aspect(box_aspect, zoom=1.25)
    except TypeError:
        ax.set_box_aspect(box_aspect)


def container_limits(container, pad: float = 0.0):
    """Axis limits from the container's own bounding box, as an equal cube.

    Figures meant to be compared side by side must share a scale, and the
    default limits are derived from the points and the radius, so two renders
    of the same container at different p or different D come out at different
    scales and the comparison is meaningless.
    """
    V = np.asarray([v.as_array() for v in container.vertices], dtype=float)
    lo, hi = V.min(axis=0) - pad, V.max(axis=0) + pad
    c = (lo + hi) / 2.0
    r = 0.5 * float(np.max(hi - lo))
    return (c[0] - r, c[0] + r, c[1] - r, c[1] + r, c[2] - r, c[2] + r)


def render_solution(X, D, container, metric, p, out_path: str, *,
                    clean: bool = True, elev: float = 20.0, azim: float = -60.0,
                    dpi: int = 300, transparent: bool = False, seed: int = 0,
                    colors=None, title=None, boundary_limits=None,
                    face_alpha: float = 0.08, sphere_alpha: float = 0.35) -> str:
    """Render one solution to a file and return the path.

    The body main() used to hold inline, so the figure pipeline can call it
    directly instead of shelling out to this command.
    """
    import matplotlib
    # Only claim the non-interactive backend if nothing has chosen one already.
    # This function writes files, but it shares a process with the viewer, and
    # forcing Agg here used to close the window path behind it.
    if matplotlib.get_backend().lower() not in ("agg",):
        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            pass
    import matplotlib.pyplot as plt
    from tsgo.plotting.visualizer import plot_result

    if title is None and not clean:
        title = f"{container.description}  p={p}  D={D:.8f}"
    ax = plot_result(X, D, container=container, show=False, show_normals=False,
                     metric_name=metric, seed=seed, colors=colors,
                     grid=not clean, title=None if clean else title,
                     boundary_limits=boundary_limits,
                     face_alpha=face_alpha, sphere_alpha=sphere_alpha)
    ax.view_init(elev=elev, azim=azim)
    if clean:
        strip_chrome(ax)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    ax.figure.savefig(out_path, dpi=dpi, bbox_inches="tight",
                      pad_inches=0.0 if clean else 0.1, transparent=transparent)
    plt.close(ax.figure)
    return out_path


def view() -> None:
    """Open stored solutions in an interactive 3-D window.

    The same renderer as tsgo-render with the window as the default rather than
    a file. Rotating a configuration is how you check one -- whether balls sit
    where you expect, whether any is against a wall it should not be -- and a
    static PNG at one camera angle answers a different question. tsgo-render
    defaults the other way because it exists to feed the figure pipeline.
    """
    main(default_show=True)


def main(default_show: bool = False) -> None:
    ap = argparse.ArgumentParser(
        description=("Open stored solutions in an interactive 3-D viewer."
                     if default_show else
                     "Render stored solutions to image files."))
    ap.add_argument("paths", nargs="+", help="Solution .pkl/.npz files, or globs.")
    ap.add_argument("--out", default=None, help="Output file (single input only).")
    ap.add_argument("--out-dir", dest="out_dir", default=None,
                    help="Output directory; names derive from the input path.")
    ap.add_argument("--clean", action="store_true",
                    help="Publication mode: no axes, ticks, gridlines or panes.")
    ap.add_argument("--elev", type=float, default=20.0, help="Camera elevation, degrees.")
    ap.add_argument("--azim", type=float, default=-60.0, help="Camera azimuth, degrees.")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--transparent", action="store_true", help="Transparent background.")
    ap.add_argument("--seed", type=int, default=0, help="Sphere-sampling seed; fixed for reproducibility.")
    ap.add_argument("--colors", default=None, help="Single colour or comma-separated list.")
    ap.add_argument("--title", default=None, help="Explicit title; default is auto, suppressed by --clean.")
    ap.add_argument("--show", action="store_true", default=default_show,
                    help="Open a window instead of saving.")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="Write a file instead of opening a window.")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tsgo.plotting.visualizer import plot_result

    files = [f for pat in args.paths for f in (glob.glob(pat) if any(c in pat for c in "*?[") else [pat])]
    if not files:
        raise SystemExit(f"no files matched: {args.paths}")

    for path in files:
        X, D, container, metric, p = load_solution(path)
        colors = args.colors.split(",") if args.colors and "," in args.colors else args.colors

        title = None if args.clean else (args.title or
                f"{container.description}  p={p}  D={D:.8f}")
        ax = plot_result(X, D, container=container, show=False, show_normals=False,
                         metric_name=metric, seed=args.seed, colors=colors,
                         grid=not args.clean, title=title)
        ax.view_init(elev=args.elev, azim=args.azim)
        if args.clean:
            strip_chrome(ax)

        if args.show:
            plt.show()
            continue

        if args.out and len(files) == 1:
            dest = args.out
        else:
            stem = os.path.splitext(path)[0].replace(os.sep, "_").lstrip("._")
            dest = os.path.join(args.out_dir or os.path.dirname(os.path.abspath(path)), stem + ".png")
        os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
        ax.figure.savefig(dest, dpi=args.dpi, bbox_inches="tight",
                          pad_inches=0.0 if args.clean else 0.1,
                          transparent=args.transparent)
        plt.close(ax.figure)
        print(f"{path} -> {dest}")


if __name__ == "__main__":
    main()
