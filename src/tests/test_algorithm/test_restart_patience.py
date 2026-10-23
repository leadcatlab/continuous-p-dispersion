"""restart_patience stops the restart loop once restarts stop paying off."""
import tsgo.algorithm.energy_functions as ef
import pytest

from tsgo.algorithm.tsgo_algorithm import _tsgo_algorithm, default_restart_patience
from tsgo.container.sample_polyhedron import CONTAINERS


def _run(tmp_path, **kw):
    trace = []
    C = CONTAINERS["cube"]()
    _, D, E = _tsgo_algorithm(p=2, D_init=ef.initial_radius(C, 2, 0.25, "EUCLIDEAN"), container=C,
                              iterations=8, Q=2, beta_max=2, eps=1e-8, metric="EUCLIDEAN",
                              optimization_method="L-BFGS-B", output_dir=str(tmp_path),
                              verbose=False, seed=0, trace=trace, **kw)
    return D, E, trace


def test_patience_stops_after_k_restarts_without_gain(tmp_path):
    # Two balls reach the optimum on the first start, so no restart can
    # improve on it: the run is the first start plus `patience` restarts.
    D, E, trace = _run(tmp_path, restart_patience=2)
    assert E < 1e-8
    assert len(trace) == 3


def test_without_patience_every_start_runs(tmp_path):
    _, _, trace = _run(tmp_path)
    assert len(trace) == 8


@pytest.mark.parametrize("p, K", [(1, 1), (2, 3), (10, 7), (20, 9), (40, 12), (60, 13), (100, 14)])
def test_default_patience_is_ceil_3_ln_p(p, K):
    assert default_restart_patience(p) == K


def test_batch_resolves_patience_and_time_limit():
    from types import SimpleNamespace
    from tsgo.experiments.batch import _patience, _time_limit
    assert _patience(SimpleNamespace(restart_patience=None), 100) == 14   # default rule
    assert _patience(SimpleNamespace(restart_patience=5), 100) == 5       # explicit
    assert _patience(SimpleNamespace(restart_patience=0), 100) is None    # off
    assert _time_limit(SimpleNamespace(time_limit=1800.0)) == 1800.0
    assert _time_limit(SimpleNamespace(time_limit=0)) is None
