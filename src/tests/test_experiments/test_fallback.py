"""The half-rung fallback: a run that ends infeasible restarts lower."""
import types

import numpy as np
import pytest

import tsgo.experiments.batch as batch
from tsgo.experiments.batch import D_INIT_LADDER, d_init_descent, descent_factors


def test_descent_walks_half_a_rung_at_a_time_to_the_bottom():
    a = D_INIT_LADDER
    assert descent_factors(0) == [a[0]]
    assert descent_factors(2) == [a[2], (a[1] + a[2]) / 2, a[1], (a[0] + a[1]) / 2, a[0]]
    top = descent_factors(len(a) - 1)
    assert len(top) == 2 * (len(a) - 1) + 1
    assert all(x > y for x, y in zip(top, top[1:]))


@pytest.mark.parametrize("container,p,metric", [
    ("cube", 1, "EUCLIDEAN"), ("cage", 1, "SUP_NORM"), ("tetrahedron", 20, "TAXICAB"),
])
def test_descent_radii_start_on_the_rung_and_end_on_rung_zero(container, p, metric):
    for seed in range(len(D_INIT_LADDER)):
        starts = d_init_descent(container, p, 0.25, metric, seed=seed)
        assert starts[0] == pytest.approx(batch.d_init_for(container, p, 0.25, metric, seed=seed))
        assert starts[-1] == pytest.approx(batch.d_init_for(container, p, 0.25, metric, seed=0))
        # clipping at the inradius can merge steps, never repeat one
        assert all(x > y for x, y in zip(starts, starts[1:]))


def _args(tmp_path):
    return types.SimpleNamespace(
        d_init=None, packing_density=0.25, output_root=str(tmp_path), eps=1e-8,
        iterations=1, processes=1, Q=3, beta_max=10, method="L-BFGS-B")


def _fake_attempt(threshold, calls):
    """Feasible at or below `threshold`, infeasible above it."""
    def attempt(container_name, metric, p, seed, D_init, args, dest, trace=None):
        calls.append(D_init)
        X = np.zeros((p, 3))
        return X, 0.5 * D_init, (0.0 if D_init <= threshold else 1e-2), ""
    return attempt


def test_an_infeasible_run_restarts_lower_until_feasible(tmp_path, monkeypatch):
    starts = d_init_descent("cube", 10, 0.25, "EUCLIDEAN", seed=4)
    calls = []
    monkeypatch.setattr(batch, "_attempt", _fake_attempt(starts[3], calls))
    row, _ = batch.run_cell("cube", "EUCLIDEAN", 10, 4, _args(tmp_path), "c", "h")
    assert calls == starts[:4]
    assert row["attempts"] == 4
    assert row["feasible"] is True
    assert float(row["D_init"]) == pytest.approx(starts[3])
    z = np.load(tmp_path / "cube" / "p10" / "euclidean" / "seed4.npz")
    assert list(z["attempt_D_init"]) == pytest.approx(starts[:4])


def test_a_feasible_run_is_not_retried(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "_attempt", _fake_attempt(np.inf, calls))
    row, _ = batch.run_cell("cube", "EUCLIDEAN", 10, 3, _args(tmp_path), "c", "h")
    assert len(calls) == 1 and row["attempts"] == 1


def test_a_run_infeasible_at_rung_zero_is_reported_infeasible(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "_attempt", _fake_attempt(0.0, calls))
    row, _ = batch.run_cell("cube", "EUCLIDEAN", 10, 4, _args(tmp_path), "c", "h")
    assert calls == d_init_descent("cube", 10, 0.25, "EUCLIDEAN", seed=4)
    assert row["feasible"] is False
    assert float(row["D_init"]) == pytest.approx(calls[-1])


def test_an_error_stops_the_descent_rather_than_walking_it(tmp_path, monkeypatch):
    calls = []

    def broken(container_name, metric, p, seed, D_init, args, dest, trace=None):
        calls.append(D_init)
        raise batch.AttemptError("boom")

    monkeypatch.setattr(batch, "_attempt", broken)
    row, failed = batch.run_cell("cube", "EUCLIDEAN", 10, 4, _args(tmp_path), "c", "h")
    assert len(calls) == 1 and row["attempts"] == 1
    assert row["feasible"] is False and failed.startswith("error:")


def test_a_run_that_collapses_to_zero_radius_counts_as_infeasible(tmp_path, monkeypatch):
    # Zero radius has zero energy, so it used to pass as a feasible answer and
    # stop the descent; it must instead fall through to the next start.
    monkeypatch.setattr(batch, "tsgo_algorithm_parallel",
                        lambda **kw: (np.zeros((kw["p"], 3)), 0.0))
    X, D, E, failed = batch._attempt("cube", "EUCLIDEAN", 4, 0, 0.1, _args(tmp_path), str(tmp_path))
    assert X is None and "collapsed" in failed
