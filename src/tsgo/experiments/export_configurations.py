"""Write the centres of every reported configuration for independent checking.

    python -m tsgo.experiments.export_configurations [--dest supplement]

For each cell the paper reports, the best feasible seed (the one the tables and
figures quote) is written as one CSV under <dest>/configurations/:

    sweep                     <c>/<metric>/p<p>.csv             supplement/tables/sweep.csv
    Stoyan & Yaskov           stoyan-yaskov-<k>/<metric>/p<p>.csv  supplement/tables/stoyan_yaskov.csv
                                                                supplement/tables/stoyan_yaskov_ext.csv
    H-box, l-inf, p = 56      h-box/sup_norm/p56.csv            supplement/tables/hbox_sup_56.csv
    extra seeds (example 3, l2, p = 88)
                              variants/extra-seeds/stoyan-yaskov-3/euclidean/p88.csv
                                                                supplement/tables/stoyan_yaskov_extra.csv

Every run has the escape phase on (the 2026-10 cluster suite), so it has no
variant tree of its own; extra seeds of a cell, made with the same solver,
are kept apart from the single-run tables.

Each file starts with '#' lines naming the container, its face list (relative
to the file, copied to <dest>/containers/), the metric, the claimed radius and
the run it came from, then x,y,z per centre at full double precision (repr
round-trips exactly). supplement/verify_packing.py checks any of them without
the solver. The run file must hold the radius its table row names; a mismatch
stops the export rather than publish centres for the wrong run.
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd

from tsgo import paths
from tsgo.container import sample_polyhedron

GEOMETRY = {
    'cube': 'cube.json',
    'h-box': 'h_box.json',
    'tetrahedron': 'tetrahedron.json',
    'pointy-star': 'pointy_star.json',
    'cage': 'ribbed_cage.json',
    **{f'stoyan-yaskov-{k}': f'stoyan_yaskov_{k}.json' for k in range(1, 9)},
}
SOURCES = [  # (study, table, results tree, subdirectory under configurations/)
    ('sweep', 'supplement/tables/sweep.csv', paths.RUNS, ''),
    ('stoyan-yaskov', 'supplement/tables/stoyan_yaskov.csv', paths.RUNS, ''),
    ('stoyan-yaskov', 'supplement/tables/stoyan_yaskov_ext.csv', paths.RUNS, ''),
    ('h-box-l-inf-56', 'supplement/tables/hbox_sup_56.csv', paths.RUNS, ''),
    # Extra seeds of one Stoyan & Yaskov cell (example 3, l2, p = 88), same solver.
    (
        'extra-seeds',
        'supplement/tables/stoyan_yaskov_extra.csv',
        'supplement/runs/variants/extra-seeds',
        'variants/extra-seeds',
    ),
]
# Tables that also list rows from other studies keep only their own cell.
ONLY = {'supplement/tables/hbox_sup_56.csv': ('h-box', 'SUP_NORM', 56)}


def best_rows(table: str) -> pd.DataFrame:
    df = pd.read_csv(table)
    df = df[df.feasible.astype(str).str.lower() == 'true']
    return df.loc[df.groupby(['container', 'metric', 'p']).D.idxmax()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='supplement')
    args = ap.parse_args()
    data = os.path.join(os.path.dirname(sample_polyhedron.__file__), 'data')
    cdir = os.path.join(args.dest, 'containers')
    os.makedirs(cdir, exist_ok=True)
    for name in GEOMETRY.values():
        shutil.copy2(os.path.join(data, name), os.path.join(cdir, name))

    index = []
    for study, table, root, sub in SOURCES:
        if not os.path.exists(table):
            print(f'[export] {table} absent, skipped')
            continue
        rows = best_rows(table)
        if table in ONLY:
            c0, m0, p0 = ONLY[table]
            rows = rows[(rows.container == c0) & (rows.metric == m0) & (rows.p == p0)]
        for _, r in rows.iterrows():
            c, m, p, seed = r.container, r.metric, int(r.p), int(r.seed)
            src = paths.run_file(root, c, m, p, seed)
            with np.load(src, allow_pickle=True) as z:
                X, D = np.asarray(z['X'], float).reshape(-1, 3), float(z['D'])
                commit = str(z['git_commit'])
            if abs(D - float(r.D)) > 1e-10:
                raise SystemExit(f'{src} holds D={D!r}, its row says {r.D!r}')
            out_dir = os.path.join(args.dest, 'configurations', sub, c, m.lower())
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f'p{p}.csv')
            geom = os.path.relpath(os.path.join(cdir, GEOMETRY[c]), out_dir)
            with open(path, 'w') as fh:
                fh.write(
                    f'# container: {c}\n# geometry: {geom}\n# metric: {m}\n# p: {p}\n'
                    f'# radius: {D!r}\n# study: {study}\n'
                    f'# source: {src} (seed {seed}, commit {commit})\n'
                )
                fh.write('i,x,y,z\n')
                for i, x in enumerate(X):
                    fh.write(f'{i},{float(x[0])!r},{float(x[1])!r},{float(x[2])!r}\n')
            index.append(
                {
                    'study': study,
                    'container': c,
                    'metric': m,
                    'p': p,
                    'radius': D,
                    'seed': seed,
                    'file': os.path.relpath(path, args.dest),
                }
            )
    idx = pd.DataFrame(index)
    idx.to_csv(os.path.join(args.dest, 'configurations.csv'), index=False)
    print(
        f'[export] {len(idx)} configurations -> {args.dest}/configurations/, '
        f'index {args.dest}/configurations.csv'
    )
    print(idx.groupby('study').size().to_string())


if __name__ == '__main__':
    main()
