#!/usr/bin/env python
"""supplement/REPORT.md: the cluster suite in one page (slurm/report.sbatch).

    python slurm/make_report.py [--previous f76300c]

Every section reads only finished files and is skipped, with a note, when they
are absent. "Previous" is the best radius of each cell as published before this
suite (supplement/configurations.csv at --previous: the Mac sweeps, escape phase
off except in the seven cells listed there under study "escape").
"""

import argparse
import io
import os
import subprocess
from pathlib import Path

import pandas as pd

R = 'supplement'  # REPORT.md sits beside tables/, figures/, benchmark/
TIE = 1e-6


def table(df: pd.DataFrame, floatfmt: str = '.6f') -> str:
    def fmt(v):
        if isinstance(v, float):
            return '' if pd.isna(v) else format(v, floatfmt)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def read(name: str):
    path = os.path.join(R, name if '/' in name else os.path.join('tables', name))
    return pd.read_csv(path) if os.path.exists(path) else None


def previous(commit: str) -> pd.DataFrame:
    try:
        text = subprocess.run(
            ['git', 'show', f'{commit}:supplement/configurations.csv'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return pd.DataFrame(columns=['container', 'metric', 'p', 'prev_D'])
    df = pd.read_csv(io.StringIO(text))
    df = df[df.study != 'escape']
    return df.rename(columns={'radius': 'prev_D'})[['container', 'metric', 'p', 'prev_D']]


def best(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.feasible.astype(str).str.lower() == 'true']
    g = ok.groupby(['container', 'metric', 'p'])
    return pd.DataFrame(
        {
            'best_D': g.D.max(),
            'median_D': g.D.median(),
            'feasible': g.size(),
            'mean_wall_s': g.wall_time_s.mean().round().astype(int),
        }
    ).reset_index()


def study_summary(name: str, df: pd.DataFrame) -> dict:
    wall = df.wall_time_s.astype(float)
    return {
        'study': name,
        'cells': df.groupby(['container', 'metric', 'p']).ngroups,
        'runs': len(df),
        'feasible': int((df.feasible.astype(str).str.lower() == 'true').sum()),
        'core_hours': round(wall.sum() / 3600, 1),
        'median_wall_s': round(wall.median(), 1),
        'max_wall_s': round(wall.max(), 1),
        'runs_past_30_min': int((wall > 1800).sum()),
    }


def vs_previous(b: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    m = b.merge(prev, on=['container', 'metric', 'p'], how='left')
    m['change_%'] = 100 * (m.best_D - m.prev_D) / m.prev_D
    return m


def counts(m: pd.DataFrame) -> str:
    c = m['change_%'].dropna() / 100
    return (
        f'higher in **{int((c > TIE).sum())}**, tied in **{int((c.abs() <= TIE).sum())}**, '
        f'lower in **{int((c < -TIE).sum())}** of {len(c)} cells (tie: within {TIE:g} relative)'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--previous', default='f76300c')
    args = ap.parse_args()
    prev = previous(args.previous)
    commit = (
        Path(R, 'COMMIT').read_text().strip() if os.path.exists(os.path.join(R, 'COMMIT')) else '?'
    )
    out = [
        '# Continuous p-dispersion: cluster results suite',
        '',
        f'Solver commit `{commit}`, every run with the escape phase on (factor 1.02, '
        f'15-min phase limit), one core per run on IllinoisComputes (AMD EPYC 7713). '
        f'Built by `slurm/submit_all.sh`; regenerate this page with `python slurm/make_report.py`.',
        f'"Previous" is each cell\'s best radius as published before this suite (commit '
        f'`{args.previous}`, Apple M2 Pro, escape phase off). `D` is the radius throughout.',
        '',
    ]

    studies = {
        'Main grid': read('sweep.csv'),
        'H-box, l-inf, p = 56': read('hbox_sup_56.csv'),
        'Stoyan & Yaskov': read('stoyan_yaskov.csv'),
        'Stoyan & Yaskov, extension': read('stoyan_yaskov_ext.csv'),
    }
    present = {k: v for k, v in studies.items() if v is not None}
    out += ['## 1. Runs', '']
    if present:
        out += [table(pd.DataFrame([study_summary(k, v) for k, v in present.items()]), '.1f'), '']
    else:
        out += ['_No finalized tables yet._', '']

    cs = read('cell_summary.csv')
    out += ['## 2. Feasibility, numerical stability and timing', '']
    if cs is not None:
        g = cs.groupby('study', sort=False)
        agg = pd.DataFrame(
            {
                'cells': g.size(),
                'runs': g.runs.sum(),
                'infeasible': g.infeasible.sum(),
                'fallback_runs': g.fallback_runs.sum(),
                'verify_fail': g.verify_fail.sum(),
                'verify_margin_min': g.verify_margin_min.min(),
                'energy_max': g.energy_max.max(),
                'monotone_violations': g.monotone_violation.sum(),
                'runs_over_50min': g.runs_over_50min.sum(),
                'wall_max_s': g.wall_max_s.max().round(),
                'core_hours': g.core_hours.sum().round(1),
            }
        ).reset_index()
        out += [
            'Every final configuration is re-certified by `supplement/verify_packing.py`, '
            'the standalone checker that shares no code with the solver: `verify_margin` is '
            '(r - D) / D, where r is the radius it recomputes from the centres alone; the solver '
            'reports D = B(X)(1 - 1e-9), so the margin should sit at +1e-9. Per cell: '
            '`supplement/tables/cell_summary.csv`.',
            '',
            table(agg, '.3g'),
            '',
        ]
        mono = cs[cs.monotone_violation.astype(bool)]
        if len(mono):
            out += [
                'Cells whose best radius exceeds that of a smaller p in the same series '
                '(impossible at the optimum, so the smaller p is under-searched):',
                '',
                table(mono[['study', 'container', 'metric', 'p', 'D_best']], '.6f'),
                '',
            ]
        slow = cs[cs.runs_over_50min > 0].sort_values('wall_max_s', ascending=False).head(10)
        if len(slow):
            out += [
                'Slowest cells. A run over 50 minutes is one whose first start, which has no '
                'deadline, ran long: seeds on the two top rungs of the D_init ladder start above '
                'the optimum, and Tabu Search there runs its full patience.',
                '',
                table(
                    slow[
                        [
                            'study',
                            'container',
                            'metric',
                            'p',
                            'runs_over_50min',
                            'wall_median_s',
                            'wall_max_s',
                        ]
                    ],
                    '.0f',
                ),
                '',
            ]
    else:
        out += ['_No cell summary yet (`python slurm/cell_summary.py`)._', '']

    sweep = studies['Main grid']
    if sweep is not None:
        m = vs_previous(best(sweep), prev)
        out += ['## 3. Main grid: best of 20 seeds', '', f'Against previous: {counts(m)}.', '']
        for c in ['cube', 'tetrahedron', 'h-box', 'pointy-star', 'cage']:
            for metric in ['EUCLIDEAN', 'TAXICAB', 'SUP_NORM']:
                cell = m[(m.container == c) & (m.metric == metric)].sort_values('p')
                if cell.empty:
                    continue
                out += [
                    f'### {c}, {metric}',
                    '',
                    table(
                        cell[
                            [
                                'p',
                                'best_D',
                                'median_D',
                                'prev_D',
                                'change_%',
                                'feasible',
                                'mean_wall_s',
                            ]
                        ].rename(columns={'feasible': 'feasible_runs'})
                    ),
                    '',
                ]
        lower = m[m['change_%'] < -100 * TIE].sort_values('change_%')
        if len(lower):
            out += [
                '### Cells below the previous best',
                '',
                table(lower[['container', 'metric', 'p', 'best_D', 'prev_D', 'change_%']]),
                '',
            ]

    h = studies['H-box, l-inf, p = 56']
    if h is not None:
        out += [
            '## 4. H-box, l-inf, p = 56',
            '',
            table(
                vs_previous(best(h), prev)[
                    ['p', 'best_D', 'median_D', 'prev_D', 'change_%', 'feasible']
                ]
            ),
            '',
        ]

    sy = (
        pd.concat(
            [
                d
                for d in (studies['Stoyan & Yaskov'], studies['Stoyan & Yaskov, extension'])
                if d is not None
            ]
        )
        if (studies['Stoyan & Yaskov'] is not None)
        else None
    )
    if sy is not None:
        b = vs_previous(best(sy), prev)
        b['fits'] = b.best_D >= 1 - 1e-9
        fit = (
            b[b.fits]
            .groupby(['container', 'metric'])
            .p.max()
            .rename('largest_count_fitted')
            .reset_index()
        )
        extra = read('stoyan_yaskov_extra.csv')
        extra_note = []
        if extra is not None:
            eb = best(extra)
            eb = eb[eb.best_D >= 1 - 1e-9]
            both = pd.concat(
                [b[b.fits][['container', 'metric', 'p']], eb[['container', 'metric', 'p']]]
            )
            fit = fit.merge(
                both.groupby(['container', 'metric'])
                .p.max()
                .rename('with_extra_seeds')
                .reset_index(),
                on=['container', 'metric'],
                how='outer',
            )
            ok = extra[extra.feasible.astype(str).str.lower() == 'true']
            for (c, m, p), g in ok.groupby(['container', 'metric', 'p']):
                hits = int((g.D >= 1 - 1e-9).sum())
                extra_note.append(
                    f'Extra seeds (declared extension, same solver, `supplement/tables/stoyan_yaskov_extra.csv`): '
                    f'{c} {m} p = {p}, seeds {int(g.seed.min())}-{int(g.seed.max())}: {hits} of '
                    f'{len(g)} reach radius >= 1, best {g.D.max():.6f}.'
                )
        out += [
            '## 5. Stoyan & Yaskov',
            '',
            f'A count is fitted when a seed reaches radius >= 1. Against previous: {counts(b)}.',
            '',
            table(fit),
            '',
            *[x for n in extra_note for x in (n, '')],
            table(
                b.sort_values(['container', 'metric', 'p'])[
                    ['container', 'metric', 'p', 'best_D', 'prev_D', 'change_%', 'feasible']
                ]
            ),
            '',
        ]

    summ = os.path.join(R, 'benchmark', 'analysis', 'summary.md')
    out += ['## 6. Equal-budget benchmark', '']
    if os.path.exists(summ):
        out += [
            'TSGO runs with the escape phase (budget_trial.py ESCAPE_FACTOR); IPOPT, SLSQP and '
            'Gurobi as in `paper/BASELINE_PROTOCOL.md`. Budgets per cell in '
            '`supplement/benchmark/budgets.csv`; per-cell plots in `supplement/benchmark/plots/`.',
            '',
            '\n'.join(
                '##' + line if line.startswith('#') else line
                for line in Path(summ).read_text().splitlines()
            ),
            '',
        ]
        for f in ('baseline_quality.png', 'baseline_anytime.png', 'baseline_outcomes.png'):
            if os.path.exists(os.path.join(R, 'figures', f)):
                out += [f'![{f}](figures/{f})', '']
    else:
        out += ['_Benchmark not finished._', '']

    cert = read('certificates/gurobi.csv')
    out += ['## 7. Gurobi optimality certificates (l2, convex containers)', '']
    out += [table(cert, '.4g'), ''] if cert is not None else ['_Not finished._', '']

    figs = (
        sorted(f for f in os.listdir(os.path.join(R, 'figures')) if f.endswith('.png'))
        if os.path.isdir(os.path.join(R, 'figures'))
        else []
    )
    if figs:
        out += [
            '## 8. Figures',
            '',
            "In `supplement/figures/`, built with the manuscript's figure "
            'code from these results (paper/ is untouched):',
            '',
        ]
        out += [f'- [{f}](figures/{f})' for f in figs] + ['']
    Path(R, 'REPORT.md').write_text('\n'.join(out))
    print(f'[report] {R}/REPORT.md')


if __name__ == '__main__':
    main()
