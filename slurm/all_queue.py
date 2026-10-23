#!/usr/bin/env python
"""jobs/queues/all.queue: the sweep, the extra seeds and the benchmark in one queue
(slurm/submit_all.sh all).

Reads the sweep's lines ("weight command", as submit_all.sh's sweep_lines prints
them) on stdin and adds the 32 extra seeds of Stoyan & Yaskov example 3 (l2,
p = 88) and every benchmark trial. The benchmark runs beside the sweep instead
of after it, so its budgets cannot come from the sweep it runs beside: they are
the previous suite's (PREVIOUS/benchmark/budgets.csv), which the current solver
reproduces radius for radius. A recorded deviation; see BASELINE_PROTOCOL.md.

Lines are ordered longest first by what they are expected to take: a run by
its own wall time in the previous suite (PREVIOUS/tables), a trial by its
budget. Longest-first on true durations packs the cores tightly, and the
handful of runs of several hours start at once rather than last.

    ./slurm/submit_all.sh sweep_lines | python slurm/all_queue.py PREVIOUS ESCAPE
"""

import glob
import os
import shutil
import sys

import pandas as pd

previous, escape = sys.argv[1], sys.argv[2]
KEY = ['container', 'metric', 'p', 'seed']

walls = pd.concat(
    [pd.read_csv(f) for f in glob.glob(os.path.join(previous, 'tables', '*.csv'))
     if {'wall_time_s', *KEY} <= set(pd.read_csv(f, nrows=0).columns)],
    ignore_index=True,
)
walls = walls.drop_duplicates(KEY, keep='last').set_index(KEY).wall_time_s
fallback = float(walls.median())


def expected(c, m, p, s) -> float:
    try:
        return float(walls.loc[(c, m, int(p), int(s))])
    except KeyError:
        return fallback


lines = []
for raw in sys.stdin:
    cmd = raw.split(' ', 1)[1].rstrip('\n')
    c, m, p, s = cmd.split('cell_run ', 1)[1].split()[:4]
    lines.append((expected(c, m, p, s), cmd))

extra_root = 'supplement/runs/variants/extra-seeds'
for s in range(10, 42):
    cmd = (f'python -m tsgo.experiments.cell_run stoyan-yaskov-3 EUCLIDEAN 88 {s} '
           f'--output-root {extra_root} --escape-factor {escape}')
    lines.append((expected('stoyan-yaskov-3', 'EUCLIDEAN', 88, s), cmd))

budgets = pd.read_csv(os.path.join(previous, 'benchmark', 'budgets.csv'))
n_trials = 0
for r in budgets.itertuples():
    for s in range(10):
        for solver in ('tsgo', 'ipopt', 'slsqp', 'gurobi'):
            if solver == 'gurobi' and (r.metric != 'EUCLIDEAN' or r.container not in ('cube', 'tetrahedron')):
                continue
            out = f'jobs/baseline/{solver}/{r.container}_{r.metric}_p{r.p}_s{s}.json'
            tail = f' {escape}' if solver == 'tsgo' else ''
            lines.append((float(r.budget_s),
                          f'[ -f {out} ] || python -m tsgo.experiments.budget_trial {solver} '
                          f'{r.container} {r.metric} {r.p} {s} {r.budget_s:.1f} jobs/baseline{tail}'))
            n_trials += 1

lines.sort(key=lambda t: -t[0])
os.makedirs('jobs/queues', exist_ok=True)
with open('jobs/queues/all.queue', 'w') as fh:
    fh.writelines(cmd + '\n' for _, cmd in lines)
os.makedirs('supplement/benchmark', exist_ok=True)
shutil.copyfile(os.path.join(previous, 'benchmark', 'budgets.csv'), 'supplement/benchmark/budgets.csv')
hours = sum(w for w, _ in lines) / 3600
print(f'[all_queue] jobs/queues/all.queue: {len(lines)} lines ({n_trials} benchmark trials), '
      f'~{hours:.0f} core-hours expected, longest {lines[0][0] / 3600:.1f} h', file=sys.stderr)
