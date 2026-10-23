"""Off-the-shelf NLP baselines: multi-start L-BFGS-B and SLSQP.

These are the comparison a referee would build. Differential evolution treats
the problem as an opaque box and beating it says little; these two use the
same formulation and the same gradients TSGO does, so what separates them from
TSGO is the global strategy -- the tabu search and basin hopping -- and
nothing else.

L-BFGS-B (box constrained). Handed the same SUMT penalty TSGO minimises
internally,

    phi(X, D) = mu * E(X, D) - D^2,

over an increasing mu, restarted from random feasible configurations until
the budget runs out. This is deliberately TSGO's own local machinery with the
global layer removed.

SLSQP (generally constrained). Handed the constrained program itself,

    maximise D   s.t.   ||c_i - c_j|| >= 2D  for i < j
                        clearance(c_i)  >= D  for all i,

so it tests the formulation rather than the restart strategy: a real
constrained solver against the penalty method. The pairwise constraints are
smooth; the clearance is the exact distance to the boundary, piecewise smooth,
and SLSQP takes its gradient by finite differences.

Both report D through max_feasible_D, the same exact bound TSGO reports, so
the numbers are comparable however the search was posed.
"""

import argparse
import csv
import os
import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize

import tsgo.algorithm.energy_functions as ef
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import CONTAINERS

FIELDS = ["run_id", "solver", "container", "metric", "p", "seed", "D", "feasible",
          "wall_time_s", "time_limit_s", "restarts", "nfev", "timestamp"]

MU_SCHEDULE = (10.0, 100.0, 1000.0, 1e4, 1e5)


def _bbox(container):
    V = np.asarray([v.as_array() for v in container.vertices], dtype=float)
    return V.min(axis=0), V.max(axis=0)


def _clearance(x: np.ndarray, container, metric: str) -> float:
    """Distance from one point to the boundary, 0 if it is outside."""
    return float(ef.max_feasible_D(np.asarray(x, dtype=float).reshape(1, 3),
                                   container, metric))


def solve_lbfgsb(container, p, metric, time_limit, seed, maxiter=200):
    """Multi-start L-BFGS-B on the SUMT penalty, with analytic gradients."""
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best_D, best_X, restarts, nfev = 0.0, None, 0, 0

    while time.perf_counter() - t0 < time_limit:
        X = container.generate_initial_solution(p, rng).reshape(p, 3)
        D = max(ef.max_feasible_D(X, container, metric), 1e-6)
        restarts += 1
        for mu in MU_SCHEDULE:
            if time.perf_counter() - t0 >= time_limit:
                break

            def fun(z):
                Xz, Dz = z[:-1].reshape(p, 3), float(z[-1])
                val, grad = ef.phi_and_grad(Xz, Dz, mu, container, metric)
                return val, grad

            z0 = np.append(X.reshape(-1), D)
            res = minimize(fun, z0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter})
            nfev += int(res.nfev)
            X, D = res.x[:-1].reshape(p, 3), float(res.x[-1])

        D_exact = ef.max_feasible_D(X, container, metric)
        if D_exact > best_D:
            best_D, best_X = float(D_exact), X.copy()

    return best_D, restarts, nfev, time.perf_counter() - t0


def solve_slsqp(container, p, metric, time_limit, seed, maxiter=200):
    """Multi-start SLSQP on the constrained program."""
    rng = np.random.default_rng(seed)
    lo, hi = _bbox(container)
    t0 = time.perf_counter()
    best_D, restarts, nfev = 0.0, 0, 0
    iu, ju = np.triu_indices(p, 1) if p > 1 else (np.array([], int), np.array([], int))

    def neg_D(z):
        return -float(z[-1])

    def grad_neg_D(z):
        g = np.zeros_like(z)
        g[-1] = -1.0
        return g

    def c_pairs(z):
        if p < 2:
            return np.array([1.0])
        X, D = z[:-1].reshape(p, 3), float(z[-1])
        d = ef._dists_point_to_rows_pairwise(X[iu], X[ju], metric)
        return d - 2.0 * D

    def c_walls(z):
        X, D = z[:-1].reshape(p, 3), float(z[-1])
        return np.array([_clearance(X[i], container, metric) - D for i in range(p)])

    cons = [{"type": "ineq", "fun": c_pairs}, {"type": "ineq", "fun": c_walls}]
    bounds = [(float(lo[k % 3]), float(hi[k % 3])) for k in range(3 * p)] + \
             [(0.0, float(np.max(hi - lo)) / 2.0)]

    while time.perf_counter() - t0 < time_limit:
        X = container.generate_initial_solution(p, rng).reshape(p, 3)
        D = max(ef.max_feasible_D(X, container, metric), 1e-6)
        restarts += 1
        res = minimize(neg_D, np.append(X.reshape(-1), D), jac=grad_neg_D,
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-10})
        nfev += int(res.nfev)
        D_exact = ef.max_feasible_D(res.x[:-1].reshape(p, 3), container, metric)
        if D_exact > best_D:
            best_D = float(D_exact)

    return best_D, restarts, nfev, time.perf_counter() - t0


SOLVERS = {"lbfgsb": solve_lbfgsb, "slsqp": solve_slsqp}


def budgets_from_sweep(path: str) -> dict:
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
    ap = argparse.ArgumentParser(description="Multi-start NLP baselines.")
    ap.add_argument("--solvers", nargs="+", default=["lbfgsb", "slsqp"],
                    choices=sorted(SOLVERS))
    ap.add_argument("--containers", nargs="+", default=["cube"])
    ap.add_argument("--p", required=True)
    ap.add_argument("--metrics", nargs="+", default=["EUCLIDEAN"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--time-limit-from", dest="tl_from", default="results/sweep.csv")
    ap.add_argument("--time-limit-scale", dest="tl_scale", type=float, default=10.0,
                    help="multiple of TSGO's t_max to grant (default 10x)")
    ap.add_argument("--time-limit-cap", dest="tl_cap", type=float, default=None)
    ap.add_argument("--min-time-limit", dest="min_tl", type=float, default=5.0)
    ap.add_argument("--out", default="results/nlp_baseline.csv")
    args = ap.parse_args()

    from tsgo.experiments.batch import parse_p
    ps: List[int] = parse_p(args.p)
    budgets = budgets_from_sweep(args.tl_from)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        with open(args.out) as fh:
            for row in csv.DictReader(fh):
                done.add((row["solver"].strip(), row["container"].strip(),
                          row["metric"].strip(), int(row["p"]), int(row["seed"])))

    cells = [(s, c, m, p, sd) for s in args.solvers for c in args.containers
             for m in args.metrics for p in ps for sd in range(args.seeds)
             if (c, m, p) in budgets]
    todo = [c for c in cells if c not in done]
    print(f"[nlp] {len(cells)} cells, {len(done)} already in {args.out}, {len(todo)} to run")
    print(f"[nlp] budget = {args.tl_scale}x TSGO t_max"
          + (f", capped at {args.tl_cap}s" if args.tl_cap else ""))

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            w.writeheader(); fh.flush()
        for i, (solver, cname, metric, p, seed) in enumerate(todo, 1):
            container = CONTAINERS[cname]()
            tl = max(budgets[(cname, metric, p)] * args.tl_scale, args.min_tl)
            if args.tl_cap:
                tl = min(tl, args.tl_cap)
            D, restarts, nfev, wall = SOLVERS[solver](container, p, metric, tl, seed)
            w.writerow({
                "run_id": f"{solver}_{cname}_{metric.lower()}_p{p}_s{seed}",
                "solver": solver, "container": cname, "metric": metric, "p": p,
                "seed": seed, "D": f"{D:.12f}", "feasible": D > 0.0,
                "wall_time_s": f"{wall:.3f}", "time_limit_s": f"{tl:.3f}",
                "restarts": restarts, "nfev": nfev,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            fh.flush()
            print(f"[{i}/{len(todo)}] {solver:<7} {cname:<12} p={p:<3} seed={seed} "
                  f"{wall:7.1f}s  D={D:.10f}  restarts={restarts}", flush=True)
    print(f"[nlp] done -> {args.out}")


if __name__ == "__main__":
    main()
