"""comparison_chart draws our radii: a lint fix once made it select no rows."""

import pandas as pd

from tsgo.experiments.plots import comparison_chart


def test_comparison_chart_uses_the_feasible_runs(tmp_path):
    # Regression: `sweep['feasible'] == True` became `sweep['feasible'] is True`
    # under ruff's E712 fix, which is False for a Series, so the chart came out
    # empty and every series had no points.
    sweep = pd.DataFrame(
        {
            'container': ['cube'] * 4,
            'metric': ['EUCLIDEAN'] * 4,
            'p': [1, 1, 2, 2],
            'D': [0.4999999995, 0.3, 0.3169872978, 0.2],
            'feasible': [True, True, True, False],
        }
    )
    series = comparison_chart(
        sweep, 'cube', 'EUCLIDEAN', str(tmp_path / 'c.png'), nlp=pd.DataFrame()
    )
    assert any(len(v.get('ratio', [])) for v in series.values())
    assert (tmp_path / 'c.png').exists()
