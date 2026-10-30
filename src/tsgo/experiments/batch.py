"""Batch experiment runner.

Sweeps (container, metric, p, seed) and appends one CSV row per run. Designed
to be interrupted: rows are flushed as they complete and a rerun with the same
--out skips cells already present, so an overnight job can be stopped and
resumed freely.

Each run's full point configuration is written to its own .npz keyed by run_id,
so results can be re-verified and re-plotted without re-solving.

    tsgo-batch --containers cube h-box --p 1-30 --out results/sweep.csv
    tsgo-batch --containers cage --p 10,20,30 --metrics EUCLIDEAN SUP_NORM
    tsgo-batch --containers tetrahedron --p 5-15 --dry-run
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from tsgo.algorithm.tsgo_algorithm import (NoFeasibleConfiguration, default_Q, default_restart_patience,
                                           tsgo_algorithm_parallel)
from tsgo.container.sample_polyhedron import CONTAINERS
import tsgo.algorithm.energy_functions as ef
import tsgo.algorithm.optimization as _opt

FIELDS = [
    "run_id", "container", "metric", "p", "seed", "D", "energy_final", "feasible",
    "wall_time_s", "iterations", "restart_patience", "time_limit", "polish", "mbh", "processes", "D_init", "Q", "beta_max", "eps",
    "boundary_mode", "analytic_grad", "git_commit", "config_hash", "timestamp",
    "attempts",
]

# Default repeats per (container, metric, p). 50 gives usable spread statistics
# at every p; --seeds overrides it. Seeds are always 0..n-1 so a cell can be
# topped up later by raising n, and the earlier runs stay reproducible.
DEFAULT_SEEDS = 50


def seeds_for(p: int, n: int = DEFAULT_SEEDS) -> List[int]:
    return list(range(n))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def parse_p(spec: str) -> List[int]:
    """'1-30' or '10,20,30' or '5-10,20' -> sorted unique list."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


# Seeds sweep the starting radius as well as the starting configuration.
#
# The volume heuristic gives an estimate, not an answer, and Adjust Distance is
# not symmetric in its error. Starting below the achievable radius costs
# iterations; starting above it sends the run down the recovery path, which
# snaps D to the bound of whatever poor configuration it is holding and never
# climbs back. On the unit cube at p = 2 under the sup norm the difference
# between a good start and a bad one is 0.230 against 0.250, the latter exact.
#
# So the initial radius becomes a second multi-start dimension: a cell's seeds
# walk a geometric ladder of K rungs spanning [1/LAMBDA, LAMBDA] about the
# estimate. One spread parameter rather than a list of tuned values. LAMBDA is
# set from measurement -- across 40 non-Euclidean cells the achievable radius
# lies between 0.77 and 1.46 times the estimate, so 1.7 brackets it with margin.
#
# Every reported radius is already the best over a cell's seeds, so this costs
# no extra runs; it widens the multi-start from configurations alone to
# (configuration, initial radius) pairs.
#
# The top of the ladder is where the best answers come from -- rung 3 alone
# holds the cell best in 23 of 219 cells, up to 25% above anything rungs 0-2
# reach -- and also where runs end infeasible. So a run that ends infeasible is
# not discarded but restarted lower, in half-rung steps: from rung i it tries
# (a[i-1] + a[i]) / 2, then a[i-1], then the next half-rung down, and so on to
# rung 0. Only a run still infeasible at rung 0 is reported infeasible.
# Each restart keeps the seed, so it starts from the same configuration and
# differs only in radius; the reported wall time covers every attempt.
D_INIT_RUNGS = 5
D_INIT_LAMBDA = 1.7
D_INIT_LADDER = tuple(
    D_INIT_LAMBDA ** (2.0 * k / (D_INIT_RUNGS - 1) - 1.0) for k in range(D_INIT_RUNGS)
)


def descent_factors(rung: int) -> List[float]:
    """Ladder factors a run on `rung` tries, in order, until one is feasible:
    a[rung], then a half-rung and a full rung at a time down to a[0]."""
    a = D_INIT_LADDER
    out = [a[rung]]
    for i in range(rung, 0, -1):
        out += [(a[i - 1] + a[i]) / 2.0, a[i - 1]]
    return out


def d_init_descent(container_name: str, p: int, packing_density: float = 0.25,
                   metric: str = "EUCLIDEAN", seed: int = 0) -> List[float]:
    """Starting radii for the seed's rung and every fallback below it.

    Radii are clipped at the inradius, so near the top of the ladder two steps
    can clip to the same radius. Consecutive duplicates are dropped: the run is
    deterministic in (seed, radius), so repeating a radius repeats the failure.
    """
    container = CONTAINERS[container_name]()
    base = ef.initial_radius(container, p, packing_density, metric)
    r_in = ef.inradius(container, metric)
    out: List[float] = []
    for f in descent_factors(seed % len(D_INIT_LADDER)):
        d = float(min(base * f, r_in)) if r_in > 0.0 else float(base * f)
        if not out or d != out[-1]:
            out.append(d)
    return out


def d_init_for(container_name: str, p: int, packing_density: float = 0.25,
               metric: str = "EUCLIDEAN", seed: Optional[int] = None) -> float:
    """Initial radius for a run. Delegates to the shared definition so this
    runner and Config cannot disagree -- they previously held separate copies of
    the formula, and only one of them learned about the inradius cap.

    With a seed, the radius is placed on the rung of D_INIT_LADDER that seed
    selects; without one, the bare estimate is returned."""
    container = CONTAINERS[container_name]()
    base = ef.initial_radius(container, p, packing_density, metric)
    if seed is None:
        return base
    factor = D_INIT_LADDER[seed % len(D_INIT_LADDER)]
    r_in = ef.inradius(container, metric)
    scaled = base * factor
    # The inradius is a hard ceiling: no radius above it is feasible at any p,
    # so a rung that would exceed it is clipped rather than wasted.
    return float(min(scaled, r_in)) if r_in > 0.0 else float(scaled)


def load_done(path: str) -> set:
    """(container, metric, p, seed) cells already recorded."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add((row["container"], row["metric"], int(row["p"]), int(row["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def run_dir(root: str, container_name: str, p: int, metric: str) -> str:
    """output/<container>/p<p>/<metric>/ -- the layout Config already uses."""
    return os.path.join(root, container_name, f"p{p}", metric.lower())


# Below this a run's radius counts as collapsed, not found (see _attempt).
MIN_RADIUS = 1e-9


class AttemptError(Exception):
    """A run that failed for a reason other than infeasibility."""


def _attempt(container_name: str, metric: str, p: int, seed: int, D_init: float,
             args, dest: str, trace: Optional[list] = None):
    """One solver run from one starting radius, checked in this process.

    The solver's own feasibility test runs in the worker, on the container it
    optimised against; the check here is independent of it, on a fresh
    container with the id-keyed caches cleared, so nothing the solver cached
    can vouch for its own answer."""
    try:
        # The solver writes a crash-recovery dump of every run; it belongs in a
        # temporary directory, not beside the results (it used to leave
        # _scratch/crashs/*.pkl and *.txt inside output/).
        with tempfile.TemporaryDirectory(prefix="tsgo_") as scratch:
            X, D = tsgo_algorithm_parallel(
                p=p, D_init=D_init, container=CONTAINERS[container_name](), Q=args.Q,
                beta_max=args.beta_max, eps=args.eps, metric=metric,
                iterations=args.iterations, processes=args.processes,
                optimization_method=args.method,
                output_dir=scratch, verbose=False, seed=seed,
                restart_patience=_patience(args, p),
                polish=not getattr(args, "no_polish", False),
                time_limit=_time_limit(args),
                trace=trace,
                **_escape_kwargs(args),
            )
    except NoFeasibleConfiguration as exc:        # infeasible: the ladder may help
        return None, float("nan"), float("nan"), str(exc)[:120]
    except Exception as exc:                      # anything else: a lower start will not
        raise AttemptError(str(exc)[:120]) from exc
    ef.clear_caches()
    E = ef.E(X, D, CONTAINERS[container_name](), metric=metric)
    if not D > MIN_RADIUS:
        # A radius of zero has zero energy, so it passes the energy test while
        # being no solution at all. From the top rung the solver can shrink
        # the radius all the way there: 37 sweep runs (2026-10-06, pointy-star
        # and tetrahedron at small p, cage) recorded D = 0 as feasible and
        # never fell back to a lower start. Treat it as infeasible so they do.
        return None, float("nan"), float("nan"), f"collapsed to D = {D:.3g}"
    return X, D, E, ""


def _patience(args, p: int) -> Optional[int]:
    """Restart patience for p: the default rule unless given; 0 turns it off."""
    k = getattr(args, "restart_patience", None)
    if k is None:
        return default_restart_patience(p)
    return k if k > 0 else None


def _time_limit(args) -> Optional[float]:
    """Seconds per run; 0 (or absent) means no limit."""
    t = getattr(args, "time_limit", None)
    return float(t) if t else None


def _escape_kwargs(args) -> Dict:
    """The escape option for the solver; empty unless --escape-factor is given."""
    f = getattr(args, "escape_factor", None)
    return {"escape_factor": f} if f else {}


def run_cell(container_name: str, metric: str, p: int, seed: int, args,
             commit: str, config_hash: str) -> Dict:
    # A fixed --d-init is a single start; the fallback walks the ladder.
    starts = ([args.d_init] if args.d_init
              else d_init_descent(container_name, p, args.packing_density, metric, seed=seed))
    run_id = f"{container_name}_{metric.lower()}_p{p}_s{seed}"
    dest = run_dir(args.output_root, container_name, p, metric)

    t0 = time.perf_counter()
    trace = []                                    # (D_init, D, E, wall) per attempt
    # (seconds since the run began, radius it would return if stopped there),
    # one entry per start of every attempt: the run's anytime curve, from
    # which time-to-target is read.
    progress = []
    for D_init in starts:
        ta = time.perf_counter()
        tr: list = []
        try:
            X, D, E, failed = _attempt(container_name, metric, p, seed, D_init, args, dest, tr)
        except AttemptError as exc:
            # An error is not infeasibility; walking the ladder would only
            # repeat it and record the error as a feasibility failure.
            X, D, E, failed = None, float("nan"), float("nan"), "error: " + str(exc)
            trace.append((D_init, D, E, time.perf_counter() - ta))
            break
        trace.append((D_init, D, E, time.perf_counter() - ta))
        progress += [(ta - t0 + t, d) for t, d in tr]
        if X is not None and np.isfinite(E) and E < args.eps:
            break
    wall = time.perf_counter() - t0

    row = {
        "run_id": run_id, "container": container_name, "metric": metric, "p": p,
        "seed": seed, "D": f"{D:.12f}", "energy_final": f"{E:.6e}",
        "feasible": bool(np.isfinite(E) and E < args.eps), "wall_time_s": f"{wall:.3f}",
        "iterations": args.iterations,
        "restart_patience": _patience(args, p),
        "time_limit": _time_limit(args),
        "polish": not getattr(args, "no_polish", False),
        "mbh": ("legacy" if _opt.MBH_LEGACY else f"step={_opt.MBH_STEP_FACTOR}D;T={_opt.MBH_T}"),
        "processes": args.processes,
        "D_init": f"{D_init:.10f}", "Q": args.Q if args.Q is not None else default_Q(p), "beta_max": args.beta_max, "eps": args.eps,
        "boundary_mode": ef.BOUNDARY_MODE,
        "analytic_grad": os.environ.get("CPDP_ANALYTIC_GRAD", "1") != "0",
        "git_commit": commit, "config_hash": config_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attempts": len(trace),
    }
    # Only runs with the escape phase carry its fields, so a default row, and
    # every row of the published sweeps, keeps exactly the columns in FIELDS.
    for k, v in _escape_kwargs(args).items():
        row[k] = v

    npz = os.path.join(dest, f"seed{seed}.npz")
    if X is None and os.path.exists(npz):
        # The final attempt returned nothing; a file left by an earlier run of
        # this key would otherwise stand in for it.
        os.remove(npz)
    if X is not None:
        os.makedirs(dest, exist_ok=True)
        # One self-describing .npz per run: the solution, the attempt history
        # and, under "row", every field of the run's table row -- so the table
        # is rebuilt from these files alone (finalize_sweep.py) and nothing
        # else needs to be kept beside them. The container is recorded by name
        # and rehydrated on load. Written to a temporary name and renamed, so
        # an interrupted run never leaves a truncated file that a resumed
        # sweep would take for a finished one.
        tmp = os.path.join(dest, f".seed{seed}.npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh,
                X=X, D=D, energy=E, p=p, seed=seed,
                container=container_name, metric=metric,
                D_init=D_init, elapsed_time=wall,
                attempt_D_init=np.array([t[0] for t in trace]),
                attempt_D=np.array([t[1] for t in trace]),
                attempt_E=np.array([t[2] for t in trace]),
                attempt_wall=np.array([t[3] for t in trace]),
                progress=np.array(progress, dtype=float).reshape(-1, 2),
                iterations=args.iterations, processes=args.processes,
                git_commit=commit, config_hash=config_hash,
                timestamp=datetime.now(timezone.utc).isoformat(),
                row=json.dumps(row),
            )
        os.replace(tmp, npz)

    return row, failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a (container x metric x p x seed) sweep.")
    ap.add_argument("--containers", nargs="+", default=["cube"], choices=list(CONTAINERS),
                    help="Containers to sweep.")
    ap.add_argument("--p", default="1-10", help="p values: '1-30', '10,20,30', or '5-10,20'.")
    ap.add_argument("--metrics", nargs="+", default=["EUCLIDEAN"],
                    choices=["EUCLIDEAN", "TAXICAB", "SUP_NORM"])
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                    help="Repeats per (container, metric, p). Default {}.".format(DEFAULT_SEEDS))
    ap.add_argument("--out", default="results/sweep.csv", help="Append-only results CSV.")
    ap.add_argument("--output-root", dest="output_root", default="output",
                    help="Root of the results tree: <root>/<container>/p<p>/<metric>/seed<n>.{npz,pkl}")
    ap.add_argument("--iterations", type=int, default=41,
                    help="Cap on starts per run (first start + restarts at the best radius). Default 41.")
    ap.add_argument("--restart-patience", dest="restart_patience", type=int, default=None,
                    help="Stop after this many restarts in a row fail to raise the best radius. "
                         "Default ceil(3 ln p); 0 turns it off.")
    ap.add_argument("--time-limit", dest="time_limit", type=float, default=1800.0,
                    help="Wall-time cap per run in seconds; a start under way is finished. "
                         "Default 1800; 0 turns it off.")
    ap.add_argument("--no-polish", dest="no_polish", action="store_true",
                    help="Skip the final LP tightening of each run's best configuration.")
    ap.add_argument("--processes", type=int, default=1, help="Parallel workers per run.")
    ap.add_argument("--Q", type=int, default=None,
                    help="Tabu neighbourhood size. Default max(3, round(ln p)).")
    ap.add_argument("--beta-max", dest="beta_max", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--method", default="L-BFGS-B")
    ap.add_argument("--d-init", dest="d_init", type=float, default=None,
                    help="Fixed initial radius. Default: from --packing-density.")
    ap.add_argument("--packing-density", dest="packing_density", type=float, default=0.25,
                    help="Fraction of the container the initial-radius estimate fills; 0.25 is the value in the paper.")
    ap.add_argument("--boundary-mode", choices=["sum", "min"], default=None)
    ap.add_argument("--escape-factor", dest="escape_factor", type=float, default=None,
                    help="Opt-in escape: a restart that ends feasible but no better retries from its "
                         "configuration with tabu aimed at this multiple of the best radius (e.g. "
                         "1.02). Off by default; writes an extra escape_factor column, so use its "
                         "own --out and --output-root.")
    ap.add_argument("--dry-run", action="store_true", help="List the cells and exit.")
    args = ap.parse_args()

    if args.boundary_mode:
        ef.set_boundary_mode(args.boundary_mode)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    ps = parse_p(args.p)
    cfg = {k: v for k, v in sorted(vars(args).items())
           if k not in ("out", "output_root", "dry_run")}
    if not args.escape_factor:
        # Off, the escape options leave the config hash what it was before
        # they existed, so default runs still match the published sweeps.
        cfg = {k: v for k, v in cfg.items() if not k.startswith("escape_")}
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    commit = git_commit()

    cells: List[Tuple[str, str, int, int]] = [
        (c, m, p, s)
        for c in args.containers for m in args.metrics for p in ps
        for s in range(args.seeds)
    ]
    done = load_done(args.out)
    todo = [c for c in cells if c not in done]

    print(f"[batch] {len(cells)} cells, {len(done)} already in {args.out}, {len(todo)} to run")
    print(f"[batch] commit {commit}  config {config_hash}  boundary={ef.BOUNDARY_MODE}")
    if args.dry_run:
        for c in todo[:40]:
            print("   ", c)
        if len(todo) > 40:
            print(f"    ... and {len(todo) - 40} more")
        return
    if not todo:
        print("[batch] nothing to do")
        return

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    t_start = time.perf_counter()
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + list(_escape_kwargs(args)))
        if write_header:
            w.writeheader(); fh.flush()
        for i, (cname, metric, p, seed) in enumerate(todo, 1):
            row, failed = run_cell(cname, metric, p, seed, args, commit, config_hash)
            w.writerow(row); fh.flush()
            done_frac = i / len(todo)
            eta = (time.perf_counter() - t_start) / done_frac * (1 - done_frac)
            status = "FAILED " + failed if failed else f"D={row['D']} feasible={row['feasible']}"
            print(f"[{i}/{len(todo)}] {cname:<12} {metric:<9} p={p:<3} seed={seed:<2} "
                  f"{row['wall_time_s']:>8}s  {status}   eta {eta/60:.1f}m", flush=True)
    for root, dirs, _ in os.walk(args.output_root):
        for d in list(dirs):
            if d == "_scratch":
                import shutil; shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    print(f"[batch] done in {(time.perf_counter()-t_start)/60:.1f}m")
    print(f"[batch]   summary   -> {args.out}")
    print(f"[batch]   solutions -> {args.output_root}/<container>/p<p>/<metric>/seed<n>.{{npz,pkl}}")


if __name__ == "__main__":
    main()
