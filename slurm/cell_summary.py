#!/usr/bin/env python
"""supplement/tables/cell_summary.csv: feasibility, numerical stability and timing per cell.

    python slurm/cell_summary.py [--jobs 8]

One row per (study, container, metric, p), from the finalized tables and the
run files under supplement/runs/. Every run's final configuration is re-certified by
supplement/verify_packing.py, the standalone numpy checker that shares no code
with the solver: it recomputes from the centres alone the largest radius r the
configuration supports, and the margin (r - D) / D must be non-negative (the
solver reports D = B(X)(1 - 1e-9), so it sits near +1e-9).

Columns
  runs, infeasible            runs in the cell, and those the solver marked infeasible
  fallback_runs               runs that needed a lower D_init (more than one attempt)
  D_best, D_median, D_worst   radius over the cell's seeds (feasible runs)
  spread_rel                  (D_best - D_worst) / D_best
  seeds_at_best               seeds within 1e-6 (relative) of D_best
  energy_max                  largest final residual E(X, D) (feasible means < 1e-8)
  verify_margin_min           smallest (r_verify - D) / D over the cell's runs
  verify_fail                 runs with r_verify < D (relative 1e-12 slack), or a centre outside
  monotone_violation          D_best exceeds the smallest D_best at a smaller p of the same series
  wall_mean_s, wall_median_s, wall_max_s, runs_over_50min, core_hours
  D_prev, change_pct          best radius previously published (commit f76300c) and the change
"""

import argparse
import io
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'supplement'))
import verify_packing as vp  # noqa: E402  (the standalone checker, not a package)

DATA = os.path.join(ROOT, 'src', 'tsgo', 'container', 'data')
GEOMETRY = {
    'cube': 'cube.json',
    'h-box': 'h_box.json',
    'tetrahedron': 'tetrahedron.json',
    'pointy-star': 'pointy_star.json',
    'cage': 'ribbed_cage.json',
    **{f'stoyan-yaskov-{k}': f'stoyan_yaskov_{k}.json' for k in range(1, 9)},
}
STUDIES = [
    ('main', 'supplement/tables/sweep.csv'),
    ('h-box-linf-56', 'supplement/tables/hbox_sup_56.csv'),
    ('stoyan-yaskov', 'supplement/tables/stoyan_yaskov.csv'),
    ('stoyan-yaskov-ext', 'supplement/tables/stoyan_yaskov_ext.csv'),
]
_FACES = {}


def verify(args):
    container, metric, p, seed = args
    if container not in _FACES:
        _FACES[container] = vp.load_container(os.path.join(DATA, GEOMETRY[container]))
    path = os.path.join(
        ROOT, 'supplement', 'runs', container, metric.lower(), f'p{p}', f'seed{seed}.npz'
    )
    with np.load(path, allow_pickle=True) as z:
        X, D = np.asarray(z['X'], float).reshape(-1, 3), float(z['D'])
    r, _, _, outside = vp.certified_radius(X, _FACES[container], metric)
    return (container, metric, p, seed), (float(r) - D) / D, int(outside)


def previous(commit='f76300c'):
    try:
        text = subprocess.run(
            ['git', 'show', f'{commit}:supplement/configurations.csv'],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout
    except subprocess.CalledProcessError:
        return pd.DataFrame(columns=['container', 'metric', 'p', 'D_prev'])
    df = pd.read_csv(io.StringIO(text))
    df = df[df.study != 'escape'].rename(columns={'radius': 'D_prev'})
    return df[['container', 'metric', 'p', 'D_prev']]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--out', default='supplement/tables/cell_summary.csv')
    a = ap.parse_args()
    os.chdir(ROOT)

    runs = []
    for study, table in STUDIES:
        df = pd.read_csv(table)
        df['study'] = study
        runs.append(df)
    runs = pd.concat(runs, ignore_index=True)
    runs['p'] = runs.p.astype(int)
    runs['feasible'] = runs.feasible.astype(str).str.lower() == 'true'
    runs['wall'] = runs.wall_time_s.astype(float)

    keys = list(zip(runs.container, runs.metric, runs.p, runs.seed.astype(int), strict=False))
    with ProcessPoolExecutor(a.jobs) as ex:
        checked = {k: (m, o) for k, m, o in ex.map(verify, keys, chunksize=16)}
    runs['verify_margin'] = [checked[k][0] for k in keys]
    runs['outside'] = [checked[k][1] for k in keys]
    runs['verify_bad'] = (runs.verify_margin < -1e-12) | (runs.outside > 0)

    rows = []
    for (study, c, m, p), g in runs.groupby(['study', 'container', 'metric', 'p'], sort=False):
        ok = g[g.feasible]
        best = ok.D.max() if len(ok) else np.nan
        rows.append(
            {
                'study': study,
                'container': c,
                'metric': m,
                'p': p,
                'runs': len(g),
                'infeasible': int((~g.feasible).sum()),
                'fallback_runs': int((g.attempts > 1).sum()),
                'D_best': best,
                'D_median': ok.D.median(),
                'D_worst': ok.D.min(),
                'spread_rel': (best - ok.D.min()) / best if len(ok) else np.nan,
                'seeds_at_best': int((best * (1 - 1e-6) <= ok.D).sum()),
                'energy_max': g.energy_final.astype(float).max(),
                'verify_margin_min': g.verify_margin.min(),
                'verify_fail': int(g.verify_bad.sum()),
                'wall_mean_s': g.wall.mean(),
                'wall_median_s': g.wall.median(),
                'wall_max_s': g.wall.max(),
                'runs_over_50min': int((g.wall > 3000).sum()),
                'core_hours': g.wall.sum() / 3600,
            }
        )
    out = pd.DataFrame(rows)

    # D cannot rise with p: a configuration of p + 1 points contains one of p.
    out['monotone_violation'] = False
    for _, g in out.groupby(['study', 'container', 'metric']):
        g = g.sort_values('p')
        prior = g.D_best.cummin().shift(1)
        bad = g.D_best > prior * (1 + 1e-7)
        out.loc[g.index[bad.fillna(False)], 'monotone_violation'] = True

    out = out.merge(previous(), on=['container', 'metric', 'p'], how='left')
    out['change_pct'] = 100 * (out.D_best - out.D_prev) / out.D_prev
    order = {'cube': 0, 'tetrahedron': 1, 'h-box': 2, 'pointy-star': 3, 'cage': 4}
    out = out.sort_values(
        ['study', 'container', 'metric', 'p'],
        key=lambda s: s.map(order).fillna(s) if s.name == 'container' else s,
    )
    cols = [
        'study',
        'container',
        'metric',
        'p',
        'runs',
        'infeasible',
        'fallback_runs',
        'D_best',
        'D_median',
        'D_worst',
        'spread_rel',
        'seeds_at_best',
        'energy_max',
        'verify_margin_min',
        'verify_fail',
        'monotone_violation',
        'wall_mean_s',
        'wall_median_s',
        'wall_max_s',
        'runs_over_50min',
        'core_hours',
        'D_prev',
        'change_pct',
    ]
    out[cols].to_csv(a.out, index=False, float_format='%.10g')

    print(f'[summary] {a.out}: {len(out)} cells, {len(runs)} runs')
    print(f'  infeasible runs     : {int((~runs.feasible).sum())}')
    print(f'  fallback runs       : {int((runs.attempts > 1).sum())}')
    print(
        f'  verify failures     : {int(runs.verify_bad.sum())}  '
        f'(margin min {runs.verify_margin.min():.3e}, max {runs.verify_margin.max():.3e})'
    )
    print(f'  energy max          : {runs.energy_final.astype(float).max():.3e}')
    print(f'  monotone violations : {int(out.monotone_violation.sum())}')
    print(
        f'  wall median / max   : {runs.wall.median():.0f} s / {runs.wall.max():.0f} s; '
        f'over 50 min: {int((runs.wall > 3000).sum())}; core-hours {runs.wall.sum() / 3600:.0f}'
    )
    print(
        f'  vs previous         : higher {int((out.change_pct > 1e-4).sum())}, '
        f'tied {int((out.change_pct.abs() <= 1e-4).sum())}, '
        f'lower {int((out.change_pct < -1e-4).sum())} (of {int(out.change_pct.notna().sum())}, tie 1e-6 rel)'
    )


if __name__ == '__main__':
    main()
