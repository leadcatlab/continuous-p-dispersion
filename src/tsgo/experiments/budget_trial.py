"""One trial of the equal-budget comparison (paper/BASELINE_PROTOCOL.md).

    python -m tsgo.experiments.budget_trial SOLVER CONTAINER METRIC P SEED BUDGET OUTDIR

SOLVER is tsgo, ipopt, slsqp or gurobi. Writes OUTDIR/SOLVER/<container>_<metric>_p<p>_s<seed>.json
with the anytime trace [(seconds, best certified D so far)], the final best
certified D, and per-solve records for the baselines. Written atomically, so a
file that exists is a finished trial.

Every solver may overrun the budget by finishing the step under way when it
ends; the trace carries the times, and the analysis counts only what finished
within the budget.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np


def tsgo_trial(name: str, p: int, metric: str, seed: int, budget: float) -> dict:
    """TSGO as in the final sweep -- ladder start, half-rung fallback, Q and the
    final LP tightening at their defaults -- with restarts at the best radius
    continuing until the budget ends rather than stopping on patience."""
    import tsgo.algorithm.energy_functions as ef
    from tsgo.algorithm.tsgo_algorithm import _tsgo_algorithm
    from tsgo.container.sample_polyhedron import CONTAINERS
    from tsgo.experiments.batch import d_init_descent

    # The same per-worker seed tsgo_algorithm_parallel derives, so the first
    # start is the one the sweep made for this seed.
    worker_seed = int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0])
    t0 = time.perf_counter()
    trace, best, attempts, feasible = [], 0.0, 0, False
    for D_init in d_init_descent(name, p, 0.25, metric, seed=seed):
        remaining = budget - (time.perf_counter() - t0)
        if remaining <= 0:
            break
        attempts += 1
        offset, tr = time.perf_counter() - t0, []
        # The solver's crash dump goes to a throwaway directory, as in the sweep.
        with tempfile.TemporaryDirectory() as scratch:
            X, D, E = _tsgo_algorithm(p=p, D_init=D_init, container=CONTAINERS[name](),
                                      iterations=10 ** 6, Q=None, beta_max=10, eps=1e-8,
                                      metric=metric, optimization_method="L-BFGS-B",
                                      output_dir=scratch, verbose=False, seed=worker_seed,
                                      time_limit=remaining, trace=tr, restart_patience=None)
        for t, d in tr:
            best = max(best, d)
            trace.append((offset + t, best))
        ef.clear_caches()
        if ef.E(X, D, CONTAINERS[name](), metric=metric) < 1e-8:
            feasible = True
            break
    return {"trace": trace, "best": best, "wall": time.perf_counter() - t0,
            "starts": len(trace), "attempts": attempts, "feasible": feasible}


def gurobi_trial(name: str, p: int, metric: str, seed: int, budget: float) -> dict:
    """Gurobi's spatial branch-and-bound on the l2 program for a convex container
    (tsgo.experiments.gurobi_bound), started cold, on one thread. Every
    incumbent it finds is scored by max_feasible_D, the exact routine that
    certifies the other solvers: Gurobi accepts constraint violations up to its
    feasibility tolerance, so its own objective can overstate the radius. The
    proven upper bound is traced too."""
    from gurobipy import GRB
    import tsgo.algorithm.energy_functions as ef
    from tsgo.container.sample_polyhedron import CONTAINERS
    from tsgo.experiments.gurobi_bound import build

    if metric != "EUCLIDEAN":
        raise ValueError("the Gurobi model is the l2 program only")
    container = CONTAINERS[name]()
    m, c, D = build(container, p)
    # The budget is enforced in the callback on perf_counter, the clock every
    # other solver uses, which stops while the machine sleeps; Gurobi's own
    # TimeLimit runs on wall time and is only a backstop here.
    m.Params.TimeLimit = 10 * budget
    m.Params.Threads = 1
    m.Params.Seed = seed
    m.Params.MIPGap = 1e-6
    t0 = time.perf_counter()
    state = {"best": 0.0, "trace": [], "bound": [], "solves": []}
    cvars = [c[i, k] for i in range(p) for k in range(3)]

    def cb(model, where):
        if time.perf_counter() - t0 >= budget:
            model.terminate()
            return
        if where == GRB.Callback.MIPSOL:
            X = np.asarray(model.cbGetSolution(cvars), dtype=float).reshape(p, 3)
            cert = float(ef.max_feasible_D(X, container, metric))
            t = time.perf_counter() - t0
            state["best"] = max(state["best"], cert)
            state["trace"].append((t, state["best"]))
            state["solves"].append({"t": t, "reported_D": float(model.cbGetSolution(D)),
                                    "certified_D": cert})
        elif where == GRB.Callback.MIP:
            bnd = float(model.cbGet(GRB.Callback.MIP_OBJBND))
            t = time.perf_counter() - t0
            if not state["bound"] or (bnd < state["bound"][-1][1] - 1e-12
                                      and t - state["bound"][-1][0] >= 1.0):
                state["bound"].append((t, bnd))

    m.optimize(cb)
    wall = time.perf_counter() - t0
    if m.SolCount:
        state["bound"].append((wall, float(m.ObjBound)))
    return {"trace": state["trace"], "bound_trace": state["bound"], "solves": state["solves"],
            "best": state["best"], "wall": wall, "starts": len(state["solves"]),
            "status": {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
                       GRB.INTERRUPTED: "budget"}.get(m.Status, str(m.Status))}


def main() -> None:
    solver, name, metric, p, seed, budget, outdir = sys.argv[1:8]
    p, seed, budget = int(p), int(seed), float(budget)
    dest = os.path.join(outdir, solver, f"{name}_{metric}_p{p}_s{seed}.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if solver == "tsgo":
        res = tsgo_trial(name, p, metric, seed, budget)
    elif solver == "gurobi":
        res = gurobi_trial(name, p, metric, seed, budget)
    else:
        from tsgo.container.sample_polyhedron import CONTAINERS
        from tsgo.experiments.cop_baselines import run_trial
        res = run_trial(solver, CONTAINERS[name](), name, p, metric, seed, budget)
    res.update({"solver": solver, "container": name, "metric": metric, "p": p, "seed": seed,
                "budget": budget})
    tmp = dest + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, dest)
    print(f"{solver:7s}{name:12s}{metric:10s}p={p:<3d}s={seed} best={res['best']:.8f} "
          f"starts={res['starts']} wall={res['wall']:.0f}s budget={budget:.0f}s", flush=True)


if __name__ == "__main__":
    main()
