"""Differential-evolution baseline, run for real.

The comparison figures previously plotted a hardcoded array labelled
"Simulated DE results (degrading with p)", with the call to the solver
commented out. This runs the solver.

Encoding. The earlier formulation optimised 3p + 1 variables -- the points
plus the radius R -- and returned a flat 1e9 whenever a configuration was
infeasible. That gives differential evolution a cliff carrying no information
about how badly a candidate misses, and much of the apparent collapse at
larger p came from the encoding rather than from the method.

Here DE searches the points only, over the container's bounding box, and the
radius is read off exactly:

    maximise  f(X) = max_feasible_D(X)

which is the largest D for which X is feasible, 0 when a point lies outside
the container. That is the same quantity TSGO reports, it removes the extra
dimension and the cliff, and it makes the baseline as strong as the method
reasonably allows -- which is the point of a baseline.

The weaker original encoding stays available via --encoding penalty, so the
two can be reported side by side.
"""

import argparse
import csv
import os
import time
from typing import List

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist

import tsgo.algorithm.energy_functions as ef
from tsgo.container.sample_polyhedron import CONTAINERS

FIELDS = ["run_id", "container", "metric", "p", "seed", "D", "feasible",
          "wall_time_s", "encoding", "time_limit_s", "nfev", "timestamp"]


def _count_outside(X, container) -> int:
    """How many of the points lie outside the container."""
    from tsgo.container.point import Point
    return sum(0 if container.contains(Point(float(a), float(b), float(c))) else 1
               for a, b, c in np.asarray(X, dtype=float))


def _feasible_population(container, p: int, n: int, seed: int) -> np.ndarray:
    """Start DE from configurations that are already inside the container.

    Its default Latin-hypercube init samples the bounding box, which for a
    container filling little of it yields an all-infeasible population.
    """
    rng = np.random.default_rng(seed)
    return np.asarray([container.generate_initial_solution(p, rng).reshape(-1)
                       for _ in range(n)], dtype=float)


def _bbox(container):
    V = np.asarray([v.as_array() for v in container.vertices], dtype=float)
    return V.min(axis=0), V.max(axis=0)


def solve_de(container, p: int, metric: str, time_limit: float, seed: int,
             encoding: str = "exact", popsize: int = 20):
    """Run DE under a wall-clock budget. Returns (D, nfev, elapsed)."""
    lo, hi = _bbox(container)
    start = time.perf_counter()
    state = {"nfev": 0}

    if encoding == "exact":
        bounds = [(float(lo[k % 3]), float(hi[k % 3])) for k in range(3 * p)]

        def objective(x):
            state["nfev"] += 1
            X = x.reshape(p, 3)
            n_out = _count_outside(X, container)
            if n_out == 0:
                return -float(ef.max_feasible_D(X, container, metric))
            # Outside the container max_feasible_D is 0, so scoring infeasible
            # candidates with it leaves a flat plateau carrying no information
            # about how far off they are. For a container filling little of its
            # bounding box that plateau is almost the whole search space -- a
            # random point set lands entirely inside the pointy star with
            # probability 1e-14 -- and DE never gets started. Rank infeasible
            # candidates by how many points escaped, so there is a path back.
            return 1.0 + n_out / float(p)
    else:
        # The original formulation, kept for comparison: points plus radius,
        # with a constant penalty for any infeasible candidate.
        rmax = float(min(hi - lo)) / 2.0
        bounds = [(float(lo[k % 3]), float(hi[k % 3])) for k in range(3 * p)] + [(0.0, rmax)]

        def objective(x):
            state["nfev"] += 1
            X = x[:-1].reshape(p, 3)
            R = float(x[-1])
            if np.any(X < lo + R) or np.any(X > hi - R):
                return 1e9
            if p > 1 and float(np.min(pdist(X))) < 2 * R:
                return 1e9
            return -R

    def callback(xk, convergence=None):
        return (time.perf_counter() - start) > time_limit

    n_pop = max(popsize * len(bounds), 5)
    if encoding == "exact":
        init = _feasible_population(container, p, n_pop, seed)
    else:
        init = "latinhypercube"
    res = differential_evolution(objective, bounds, strategy="best1bin",
                                 popsize=popsize, seed=seed, polish=True,
                                 init=init, callback=callback, disp=False)
    elapsed = time.perf_counter() - start

    # Score whatever DE returned with the exact feasibility bound, so the
    # number is comparable to TSGO's however the search was encoded.
    x = np.asarray(res.x, dtype=float)
    X = (x if encoding == "exact" else x[:-1]).reshape(p, 3)
    D = float(ef.max_feasible_D(X, container, metric))
    return D, state["nfev"], elapsed


def budgets_from_sweep(path: str) -> dict:
    """Per-cell DE budget taken from TSGO's own worst-case time.

    Maps (container, metric, p) -> max wall time TSGO spent on that cell. A
    flat budget always invites the objection that the baseline was not given
    long enough; matching each cell to the slowest TSGO run on the same
    configuration removes the argument, and errs in the baseline's favour --
    DE gets TSGO's worst case, not its mean.
    """
    import pandas as pd
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        try:
            df[col] = df[col].str.strip()
        except AttributeError:
            pass
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["wall_time_s"] = pd.to_numeric(df["wall_time_s"], errors="coerce")
    g = df.groupby(["container", "metric", "p"])["wall_time_s"].max()
    return {(c, m, int(p)): float(v) for (c, m, p), v in g.items() if v == v}


def main() -> None:
    ap = argparse.ArgumentParser(description="Differential-evolution baseline.")
    ap.add_argument("--containers", nargs="+", default=["cube"])
    ap.add_argument("--p", default=None, help="'1-10' or '1,2,5'; omit with "
                                              "--time-limit-from to use every p TSGO ran")
    ap.add_argument("--metrics", nargs="+", default=["EUCLIDEAN"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--time-limit", dest="time_limit", type=float, default=60.0)
    ap.add_argument("--time-limit-from", dest="time_limit_from", default=None,
                    help="sweep CSV; budget each cell at TSGO's t_max for the "
                         "same (container, metric, p)")
    ap.add_argument("--time-limit-scale", dest="tl_scale", type=float, default=1.0,
                    help="multiply the matched budget, to test sensitivity to it")
    ap.add_argument("--time-limit-cap", dest="tl_cap", type=float, default=None,
                    help="upper bound on a scaled budget")
    ap.add_argument("--min-time-limit", dest="min_tl", type=float, default=1.0,
                    help="floor for a matched budget, so a sub-second TSGO cell "
                         "still gives DE a usable run")
    ap.add_argument("--encoding", choices=["exact", "penalty"], default="exact")
    ap.add_argument("--popsize", type=int, default=20)
    ap.add_argument("--out", default="results/de_baseline.csv")
    args = ap.parse_args()

    budgets = budgets_from_sweep(args.time_limit_from) if args.time_limit_from else None

    from tsgo.experiments.batch import parse_p
    if args.p:
        ps: List[int] = parse_p(args.p)
    elif budgets:
        ps = sorted({p for (c, m, p) in budgets
                     if c in args.containers and m in args.metrics})
    else:
        ap.error("--p is required unless --time-limit-from is given")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out) as fh:
            for row in csv.DictReader(fh):
                done.add((row["container"].strip(), row["metric"].strip(),
                          int(row["p"]), int(row["seed"])))

    cells = [(c, m, p, s) for c in args.containers for m in args.metrics
             for p in ps for s in range(args.seeds)]
    if budgets is not None:
        # Only cells TSGO actually ran have a matched budget; the grids differ
        # per container (the tetrahedron uses tetrahedral numbers), so this
        # silently keeps DE on exactly the configurations there is data for.
        cells = [c for c in cells if (c[0], c[1], c[2]) in budgets]
    todo = [c for c in cells if c not in done]
    print(f"[de] {len(cells)} cells, {len(done)} already in {args.out}, {len(todo)} to run")
    budget_desc = (f"matched to TSGO t_max from {args.time_limit_from}"
                   if budgets is not None else f"{args.time_limit}s flat")
    print(f"[de] encoding={args.encoding} budget={budget_desc} popsize={args.popsize}")

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            w.writeheader(); fh.flush()
        for i, (cname, metric, p, seed) in enumerate(todo, 1):
            container = CONTAINERS[cname]()
            if budgets is not None:
                tl = max(budgets[(cname, metric, p)] * args.tl_scale, args.min_tl)
                if args.tl_cap is not None:
                    tl = min(tl, args.tl_cap)
            else:
                tl = args.time_limit
            D, nfev, elapsed = solve_de(container, p, metric, tl,
                                        seed, args.encoding, args.popsize)
            w.writerow({
                "run_id": f"de_{cname}_{metric.lower()}_p{p}_s{seed}",
                "container": cname, "metric": metric, "p": p, "seed": seed,
                "D": f"{D:.12f}", "feasible": D > 0.0,
                "wall_time_s": f"{elapsed:.3f}", "encoding": args.encoding,
                "time_limit_s": f"{tl:.3f}", "nfev": nfev,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            fh.flush()
            print(f"[{i}/{len(todo)}] {cname:<12} {metric:<9} p={p:<3} seed={seed} "
                  f"{elapsed:7.1f}s  D={D:.10f}  nfev={nfev}", flush=True)
    print(f"[de] done -> {args.out}")


if __name__ == "__main__":
    main()
