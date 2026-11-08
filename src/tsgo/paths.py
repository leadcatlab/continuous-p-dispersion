"""Where results live, relative to the repository root.

Every final result is under supplement/ (see supplement/README.md):

    supplement/runs/<container>/<metric>/p<p>/seed<k>.npz   one file per run
    supplement/configurations/<container>/<metric>/p<p>.csv best run of each cell
    supplement/tables/          sweep.csv (the gold file) and the other run tables
    supplement/figures/         figures built from them
    supplement/benchmark/       equal-budget benchmark: analysis, budgets, per-cell plots
    supplement/certificates/    Gurobi certificates

Job state that is only needed while a suite runs (queues, SLURM logs, the raw
benchmark trials, ad-hoc solver runs) goes under jobs/, which is not tracked.
"""

import os

SUPPLEMENT = 'supplement'
RUNS = os.path.join(SUPPLEMENT, 'runs')
TABLES = os.path.join(SUPPLEMENT, 'tables')
SWEEP = os.path.join(TABLES, 'sweep.csv')
FIGURES = os.path.join(SUPPLEMENT, 'figures')
BENCHMARK = os.path.join(SUPPLEMENT, 'benchmark')
CERTIFICATES = os.path.join(SUPPLEMENT, 'certificates')

JOBS = 'jobs'
TRIALS = os.path.join(JOBS, 'baseline')  # raw benchmark trials, <solver>/*.json


def table(name: str) -> str:
    """supplement/tables/<name>."""
    return os.path.join(TABLES, name)


def run_dir(root: str, container: str, metric: str, p: int) -> str:
    """<root>/<container>/<metric>/p<p>/, the folder holding a cell's runs."""
    return os.path.join(root, container, metric.lower(), f'p{int(p)}')


def run_file(root: str, container: str, metric: str, p: int, seed: int) -> str:
    """<root>/<container>/<metric>/p<p>/seed<k>.npz."""
    return os.path.join(run_dir(root, container, metric, p), f'seed{int(seed)}.npz')
