"""Experiment C of paper/BASELINE_PROTOCOL.md: optimality certificates.

For each convex container under l2 and each p, Gurobi's spatial branch-and-bound
maximises D with a one-hour limit, warm-started from TSGO's best configuration
in the final sweep. Its proven upper bound brackets the optimum: the gap
(bound - TSGO) / bound is how far TSGO's answer can be from optimal. Rows are
appended to supplement/certificates/gurobi.csv; a rerun skips finished cells.

    python -m tsgo.experiments.certify_gurobi                    # every cell, in turn
    python -m tsgo.experiments.certify_gurobi --cell cube 10     # one cell, to its own part file
    python -m tsgo.experiments.certify_gurobi --merge            # the part files into OUT

One cell per job lets the 22 cells run side by side (slurm/submit_all.sh all);
each writes supplement/certificates/parts/<container>_p<p>.csv, and --merge
gathers them into OUT in the usual order.
"""

import argparse
import csv
import glob
import os

import numpy as np
import pandas as pd

from tsgo import paths
from tsgo.container.sample_polyhedron import CONTAINERS
from tsgo.experiments.gurobi_bound import solve

OUT = 'supplement/certificates/gurobi.csv'
FIELDS = [
    'container',
    'p',
    'status',
    'tsgo_best',
    'gurobi_incumbent',
    'gurobi_bound',
    'gap_tsgo_to_bound',
    'wall_s',
    'time_limit_s',
]


PARTS = 'supplement/certificates/parts'
CELLS = [(name, p) for name in ('cube', 'tetrahedron') for p in [*range(1, 11), 20]]


def merge() -> None:
    rows = []
    for f in glob.glob(os.path.join(PARTS, '*.csv')):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    order = {c: i for i, c in enumerate(CELLS)}
    rows.sort(key=lambda r: order[(r['container'], int(r['p']))])
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    missing = [c for c in CELLS if c not in {(r['container'], int(r['p'])) for r in rows}]
    print(
        f'[certify] {OUT}: {len(rows)} of {len(CELLS)} cells'
        + (f', missing {missing}' if missing else '')
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', nargs=2, metavar=('CONTAINER', 'P'))
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    if a.merge:
        merge()
        return
    cells = [(a.cell[0], int(a.cell[1]))] if a.cell else CELLS
    out = os.path.join(PARTS, f'{cells[0][0]}_p{cells[0][1]}.csv') if a.cell else OUT
    run(cells, out)


def run(cells, out) -> None:
    sweep = pd.read_csv('supplement/tables/sweep.csv')
    ok = sweep[sweep.feasible & (sweep.metric == 'EUCLIDEAN')]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        with open(out) as fh:
            done = {(r['container'], int(r['p'])) for r in csv.DictReader(fh)}
    new_file = not os.path.exists(out)
    with open(out, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
            fh.flush()
        for name, p in cells:
            if (name, p) in done:
                continue
            cell = ok[(ok.container == name) & (ok.p == p)]
            best_row = cell.loc[cell.D.idxmax()]
            X = np.load(paths.run_file(paths.RUNS, name, 'euclidean', p, best_row.seed))['X']
            # Gurobi's symmetry cut orders the points by x; order the warm start to match.
            X = X[np.argsort(X[:, 0])]
            r = solve(CONTAINERS[name](), p, time_limit=3600.0, threads=1, warm_start=X)
            tsgo = float(best_row.D)
            w.writerow(
                {
                    'container': name,
                    'p': p,
                    'status': r['status'],
                    'tsgo_best': f'{tsgo:.10f}',
                    'gurobi_incumbent': f'{r["D_incumbent"]:.10f}',
                    'gurobi_bound': f'{r["D_bound"]:.10f}',
                    'gap_tsgo_to_bound': f'{(r["D_bound"] - tsgo) / r["D_bound"]:.3e}',
                    'wall_s': f'{r["wall_s"]:.1f}',
                    'time_limit_s': 3600,
                }
            )
            fh.flush()
            print(
                f'{name:12s} p={p:<3d} {r["status"]:10s} tsgo={tsgo:.8f} bound={r["D_bound"]:.8f} '
                f'gap={(r["D_bound"] - tsgo) / r["D_bound"]:.2e} {r["wall_s"]:.0f}s',
                flush=True,
            )


if __name__ == '__main__':
    main()
