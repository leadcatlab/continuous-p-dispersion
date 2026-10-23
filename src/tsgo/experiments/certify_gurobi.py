"""Experiment C of paper/BASELINE_PROTOCOL.md: optimality certificates.

For each convex container under l2 and each p, Gurobi's spatial branch-and-bound
maximises D with a one-hour limit, warm-started from TSGO's best configuration
in the final sweep. Its proven upper bound brackets the optimum: the gap
(bound - TSGO) / bound is how far TSGO's answer can be from optimal. Rows are
appended to results/certificates/gurobi.csv; a rerun skips finished cells.

    python -m tsgo.experiments.certify_gurobi
"""
import csv
import os

import numpy as np
import pandas as pd

from tsgo.container.sample_polyhedron import CONTAINERS
from tsgo.experiments.gurobi_bound import solve

OUT = "results/certificates/gurobi.csv"
FIELDS = ["container", "p", "status", "tsgo_best", "gurobi_incumbent", "gurobi_bound",
          "gap_tsgo_to_bound", "wall_s", "time_limit_s"]


def main() -> None:
    sweep = pd.read_csv("results/sweep.csv")
    ok = sweep[sweep.feasible & (sweep.metric == "EUCLIDEAN")]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        done = {(r["container"], int(r["p"])) for r in csv.DictReader(open(OUT))}
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader(); fh.flush()
        for name in ("cube", "tetrahedron"):
            for p in list(range(1, 11)) + [20]:
                if (name, p) in done:
                    continue
                cell = ok[(ok.container == name) & (ok.p == p)]
                best_row = cell.loc[cell.D.idxmax()]
                X = np.load(f"output/{name}/p{p}/euclidean/seed{int(best_row.seed)}.npz")["X"]
                # Gurobi's symmetry cut orders the points by x; order the warm start to match.
                X = X[np.argsort(X[:, 0])]
                r = solve(CONTAINERS[name](), p, time_limit=3600.0, threads=1, warm_start=X)
                tsgo = float(best_row.D)
                w.writerow({"container": name, "p": p, "status": r["status"], "tsgo_best": f"{tsgo:.10f}",
                            "gurobi_incumbent": f"{r['D_incumbent']:.10f}",
                            "gurobi_bound": f"{r['D_bound']:.10f}",
                            "gap_tsgo_to_bound": f"{(r['D_bound'] - tsgo) / r['D_bound']:.3e}",
                            "wall_s": f"{r['wall_s']:.1f}", "time_limit_s": 3600})
                fh.flush()
                print(f"{name:12s} p={p:<3d} {r['status']:10s} tsgo={tsgo:.8f} bound={r['D_bound']:.8f} "
                      f"gap={(r['D_bound'] - tsgo) / r['D_bound']:.2e} {r['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
