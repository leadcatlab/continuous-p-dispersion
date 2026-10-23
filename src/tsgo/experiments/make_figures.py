"""Rebuild the paper's generated figures from the recorded results.

Reads paper/figures.manifest.json, which declares for each figure whether it
comes from a solver run or is hand-drawn artwork. For a generated figure the
best seed of the named (container, metric, p) cell is located in the sweep,
rendered without axes or gridlines, and written under the name the manuscript
already uses -- so no \\includegraphics path ever has to change.

A declared figure whose cell has no data is reported and, unless --keep-going
is given, fails the build. That is deliberate: the failure mode worth
preventing is a regenerated paper silently keeping a stale image from an
earlier version of the algorithm.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_sweep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError:
            pass
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["D"] = pd.to_numeric(df["D"], errors="coerce")
    df["feasible"] = df["feasible"].astype(str).str.lower().eq("true")
    return df


def best_seed(df: pd.DataFrame, container: str, metric: str, p: int):
    """Seed with the largest feasible D for a cell, or None if there is none."""
    cell = df[(df.container == container) & (df.metric == metric) &
              (df.p == p) & df.feasible]
    if cell.empty:
        return None
    row = cell.loc[cell["D"].idxmax()]
    return int(row["seed"]), float(row["D"])


def ball_colors(spec: dict, X):
    """Per-ball colours from the manifest entry, or None for the renderer's
    default. "colors": one colour for every ball. "color_by": "height" maps
    each ball's height to a sequential colormap ("cmap": a matplotlib name or a
    list of colour stops from bottom to top, restricted to "cmap_range"), so
    the vertical layering of a packing reads at a glance."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    if spec.get("color_by") == "height":
        cmap = spec.get("cmap", "YlOrRd")
        cmap = (LinearSegmentedColormap.from_list("stops", cmap)
                if isinstance(cmap, list) else plt.get_cmap(cmap))
        lo, hi = spec.get("cmap_range", (0.35, 0.85))
        z = np.asarray(X, dtype=float)[:, 2]
        t = (z - z.min()) / (np.ptp(z) or 1.0)
        return [cmap(lo + (hi - lo) * v) for v in t]
    if spec.get("colors"):
        return [spec["colors"]] * len(X)
    return None


def solution_path(root: str, container: str, metric: str, p: int, seed: int) -> str:
    return os.path.join(root, container, f"p{p}", metric.lower(), f"seed{seed}.npz")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the paper's generated figures.")
    ap.add_argument("--manifest", default="paper/figures.manifest.json")
    ap.add_argument("--sweep", default="results/sweep.csv")
    ap.add_argument("--output-root", dest="output_root", default="output")
    ap.add_argument("--figures", default="paper/figures")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--archive", default="results/figures",
                    help="keep a copy of each figure named for the run that "
                         "produced it; empty string disables")
    ap.add_argument("--provenance", default="paper/figures/PROVENANCE.json",
                    help="record which run each paper figure came from")
    ap.add_argument("--keep-going", action="store_true",
                    help="report missing cells instead of failing")
    ap.add_argument("--list", action="store_true", help="show what would be built")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    generated: Dict[str, dict] = manifest.get("generated", {})
    static: List[str] = manifest.get("static", [])
    pending: Dict[str, dict] = manifest.get("pending", {})

    df = load_sweep(args.sweep)
    os.makedirs(args.figures, exist_ok=True)

    built, missing = [], []
    prov: Dict[str, dict] = {}
    for name, spec in generated.items():
        container, metric, p = spec["container"], spec["metric"], int(spec["p"])
        # A cell outside the gold file's grid names its own rows and solution
        # tree; the rows are then read from there instead of the sweep.
        cell_df, out_root = df, args.output_root
        if spec.get("rows"):
            import glob as _glob
            files = sorted(_glob.glob(spec["rows"]))
            cell_df = (pd.concat([load_sweep(f) for f in files], ignore_index=True)
                       if files else df.iloc[0:0])
            out_root = spec.get("output_root", out_root)
        hit = best_seed(cell_df, container, metric, p)
        if hit is None:
            missing.append((name, f"no feasible run for {container}/{metric}/p={p}"))
            continue
        seed, D = hit
        src = solution_path(out_root, container, metric, p, seed)
        if not os.path.exists(src):
            missing.append((name, f"solution file absent: {src}"))
            continue
        dest = os.path.join(args.figures, f"{name}.png")
        if args.list:
            print(f"  {name:<24} <- {container}/{metric}/p={p} seed={seed} D={D:.8f}")
            built.append(name)
            continue

        from tsgo.experiments.render import (load_solution, render_solution,
                                             container_limits)
        X, Dv, cont, met, pv = load_solution(src)

        # The stored solution must be the run the sweep row names. Solution
        # files are keyed by (container, p, metric, seed) only, so a job run
        # under a different configuration -- the min/sum A/B test, say --
        # overwrites them, and 17 Euclidean rows currently disagree with the
        # file at their own path. Rendering one of those would put a figure in
        # the paper attributed to a run that did not produce it, silently.
        if abs(float(Dv) - float(D)) > 1e-10:
            missing.append((name, f"{src} holds D={float(Dv):.10f} but the sweep "
                                  f"row for seed {seed} says D={float(D):.10f}; "
                                  f"the file was overwritten by another job"))
            continue

        # "radius" draws the same configuration at a different D -- used for
        # the side-by-side against a published radius. "scale": "container"
        # pins the axes to the container's bounding box so paired figures
        # share a scale exactly, whatever their p or D.
        D_draw = float(spec["radius"]) if spec.get("radius") is not None else Dv
        lims = (container_limits(cont, pad=spec.get("scale_pad", 0.0))
                if spec.get("scale") == "container" else None)

        render_solution(X, D_draw, cont, met, pv, dest, clean=True,
                        elev=spec.get("elev", 20.0), azim=spec.get("azim", -60.0),
                        dpi=args.dpi, boundary_limits=lims, colors=ball_colors(spec, X),
                        face_alpha=spec.get("face_alpha", 0.08),
                        sphere_alpha=spec.get("sphere_alpha", 0.35))

        # Archive under a name that says which run it came from. The figure in
        # paper/ is overwritten in place so the manuscript needs no path
        # changes, which means the previous image survives only in git; the
        # archive keeps every rendering that has ever been used, and the seed
        # is chosen at build time as the best feasible run, so without this
        # there is no record of which run a published figure shows.
        if args.archive:
            os.makedirs(args.archive, exist_ok=True)
            arch = os.path.join(
                args.archive,
                f"{container}_{metric.lower()}_p{p}_seed{seed}.png")
            shutil.copy2(dest, arch)

        prov[name] = {
            "container": container, "metric": metric, "p": p, "seed": seed,
            "D": D, "D_drawn": D_draw, "scale": spec.get("scale", "auto"),
            "source": src,
            "archived_as": arch if args.archive else None,
            "sha256": _sha256(dest), "rendered_at": datetime.now().isoformat(timespec="seconds"),
        }
        print(f"[figures] {name}.png  <- {container}/{metric}/p={p} seed={seed} D={D:.8f}")
        built.append(name)

    # Charts from the sweep. Declared in the manifest's "plots" section, which
    # until now named figures that no code produced -- the committed PNGs could
    # not be traced to any run. A plot whose cells are absent is reported like a
    # missing render rather than silently leaving a stale image in place.
    plots: Dict[str, dict] = manifest.get("plots", {})
    for name, spec in plots.items():
        kind = spec.get("kind")
        metric = spec.get("metric", "EUCLIDEAN")

        if kind == "comparison":
            container = spec["container"]
            have = df[(df["metric"] == metric) & (df["container"] == container)]
            if have.empty:
                missing.append((name, f"no {metric} runs for {container}"))
                continue
            dest = os.path.join(args.figures, f"{name}.png")
            if args.list:
                print(f"  {name:<24} <- comparison/{container}/{metric}")
                built.append(name)
                continue
            from tsgo.experiments.plots import comparison_chart, load_nlp_baseline
            series = comparison_chart(df, container, metric, dest,
                                      nlp=load_nlp_baseline(), dpi=args.dpi)
            prov[name] = {"kind": "comparison", "container": container,
                          "metric": metric, "series": series,
                          "sha256": _sha256(dest),
                          "rendered_at": datetime.now().isoformat(timespec="seconds")}
            worst = min((min(v["ratio"]) for v in series.values()), default=float("nan"))
            print(f"[figures] {name}.png  <- comparison/{container}/{metric}, "
                  f"worst ratio {worst:.3f}")
            built.append(name)
            continue

        if kind != "runtime":
            continue
        have = df[df["metric"] == metric]
        if have.empty:
            missing.append((name, f"no {metric} runs in the sweep"))
            continue
        dest = os.path.join(args.figures, f"{name}.png")
        if args.list:
            cs = sorted(set(have["container"]))
            print(f"  {name:<24} <- runtime/{metric}: {', '.join(cs)}")
            built.append(name)
            continue
        from tsgo.experiments.plots import runtime_chart
        series = runtime_chart(have, metric, dest, containers=spec.get("containers"),
                               dpi=args.dpi)
        prov[name] = {
            "kind": "runtime", "metric": metric,
            "containers": {k: {"p": v["p"], "runs": v["runs"]} for k, v in series.items()},
            "sha256": _sha256(dest),
            "rendered_at": datetime.now().isoformat(timespec="seconds"),
        }
        cells = sum(len(v["p"]) for v in series.values())
        print(f"[figures] {name}.png  <- runtime/{metric}, "
              f"{len(series)} containers, {cells} cells")
        built.append(name)

    for name, why in missing:
        print(f"[figures] MISSING {name}: {why}", file=sys.stderr)

    for name, spec in pending.items():
        print(f"[figures] PENDING {name}: {spec.get('reason', 'not yet reproducible')} "
              f"-- committed image retained", file=sys.stderr)

    if prov and args.provenance and not args.list:
        os.makedirs(os.path.dirname(args.provenance) or ".", exist_ok=True)
        with open(args.provenance, "w") as fh:
            json.dump({"figures": prov,
                       "sweep": os.path.abspath(args.sweep),
                       "git_commit": _git_commit()}, fh, indent=2, sort_keys=True)
        print(f"[figures] provenance -> {args.provenance}")

    print(f"[figures] {len(built)} generated, {len(missing)} missing, "
          f"{len(pending)} pending, {len(static)} static (not rebuilt)")
    if missing and not args.keep_going and not args.list:
        raise SystemExit(f"{len(missing)} declared figure(s) could not be built. "
                         f"Fix the manifest or run the missing cells; "
                         f"--keep-going to proceed anyway.")


if __name__ == "__main__":
    main()
