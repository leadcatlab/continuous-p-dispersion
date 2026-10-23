"""TSGO against Gurobi on the convex containers.

Runs both on the same (container, p) cells and reports, per cell, the
heuristic's best D over its seeds, Gurobi's incumbent and certified upper
bound, the resulting optimality gap, and the wall time of each.

    tsgo-compare --containers cube tetrahedron --p 2-8 --seeds 5 --time-limit 300
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from typing import List

import numpy as np

from tsgo.algorithm.tsgo_algorithm import tsgo_algorithm_parallel
from tsgo.container.sample_polyhedron import CONTAINERS
from tsgo.experiments.batch import d_init_for, git_commit, parse_p
from tsgo.experiments.gurobi_bound import CONVEX_CONTAINERS, solve as gurobi_solve
import tsgo.algorithm.energy_functions as ef

FIELDS = ["container", "p", "seeds", "D_tsgo_best", "D_tsgo_mean", "D_tsgo_std",
          "t_tsgo_total_s", "t_tsgo_mean_s", "D_gurobi", "D_gurobi_bound",
          "gurobi_gap", "gurobi_status", "t_gurobi_s", "rel_err_vs_bound",
          "certified_optimal", "git_commit", "timestamp"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare TSGO with Gurobi's certified bound.")
    ap.add_argument("--containers", nargs="+", default=list(CONVEX_CONTAINERS),
                    choices=list(CONVEX_CONTAINERS),
                    help="Only convex containers admit the half-space formulation.")
    ap.add_argument("--p", default="2-8")
    ap.add_argument("--metric", default="EUCLIDEAN", help="Gurobi's model is Euclidean only.")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--time-limit", dest="time_limit", type=float, default=600.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--packing-density", dest="packing_density", type=float, default=0.25)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--warm-start", dest="warm_start", action="store_true",
                    help="Seed Gurobi with the heuristic's configuration.")
    ap.add_argument("--out", default="results/compare.csv")
    args = ap.parse_args()

    if args.metric.upper() != "EUCLIDEAN":
        raise SystemExit("The Gurobi model encodes squared Euclidean distances; use --metric EUCLIDEAN.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    commit = git_commit()
    new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0

    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader(); fh.flush()

        for cname in args.containers:
            for p in parse_p(args.p):
                container = CONTAINERS[cname]()
                D_init = d_init_for(cname, p, args.packing_density)

                Ds, ts, best_X = [], [], None
                for seed in range(args.seeds):
                    t0 = time.perf_counter()
                    try:
                        X, D = tsgo_algorithm_parallel(
                            p=p, D_init=D_init, container=CONTAINERS[cname](), Q=3,
                            beta_max=10, eps=1e-8, metric="EUCLIDEAN",
                            iterations=args.iterations, processes=1,
                            optimization_method="L-BFGS-B",
                            output_dir=os.path.join("output", "_compare_scratch"),
                            verbose=False, seed=seed)
                    except Exception:
                        continue
                    finally:
                        ts.append(time.perf_counter() - t0)
                    Ds.append(D)
                    if best_X is None or D >= max(Ds):
                        best_X = X

                if not Ds:
                    print(f"  {cname} p={p}: TSGO found nothing feasible; skipping")
                    continue
                D_best = float(np.max(Ds))

                g = gurobi_solve(container, p, time_limit=args.time_limit,
                                 threads=args.threads,
                                 warm_start=best_X if args.warm_start else None)

                bound = g["D_bound"]
                rel = (bound - D_best) / bound * 100 if np.isfinite(bound) and bound > 0 else float("nan")
                certified = bool(g["status"] == "optimal")

                row = {
                    "container": cname, "p": p, "seeds": args.seeds,
                    "D_tsgo_best": f"{D_best:.10f}",
                    "D_tsgo_mean": f"{np.mean(Ds):.10f}",
                    "D_tsgo_std": f"{np.std(Ds, ddof=1) if len(Ds) > 1 else 0.0:.3e}",
                    "t_tsgo_total_s": f"{np.sum(ts):.3f}", "t_tsgo_mean_s": f"{np.mean(ts):.3f}",
                    "D_gurobi": f"{g['D_incumbent']:.10f}", "D_gurobi_bound": f"{bound:.10f}",
                    "gurobi_gap": f"{g['gap']:.3e}", "gurobi_status": g["status"],
                    "t_gurobi_s": f"{g['wall_s']:.3f}", "rel_err_vs_bound": f"{rel:.6f}",
                    "certified_optimal": certified, "git_commit": commit,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                w.writerow(row); fh.flush()
                print(f"{cname:<12} p={p:<3} TSGO {D_best:.8f} ({np.sum(ts):6.1f}s)   "
                      f"Gurobi {g['D_incumbent']:.8f} bound {bound:.8f} "
                      f"[{g['status']}, {g['wall_s']:.1f}s]   gap to bound {rel:.4f}%", flush=True)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
