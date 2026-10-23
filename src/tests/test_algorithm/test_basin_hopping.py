"""Basin hopping is monotonic with a step in units of D, unless legacy is asked for."""
import numpy as np
import pytest

import tsgo.algorithm.optimization as opt
from tsgo.container.sample_polyhedron import CONTAINERS


def _captured(monkeypatch):
    seen = {}
    real = opt.basinhopping

    def spy(**kw):
        seen.update(kw)
        return real(**kw)
    monkeypatch.setattr(opt, "basinhopping", spy)
    return seen


def test_default_is_monotonic_with_step_in_units_of_D(monkeypatch):
    seen = _captured(monkeypatch)
    monkeypatch.setattr(opt, "MBH_LEGACY", False)
    C = CONTAINERS["cube"]()
    X = C.generate_initial_solution(4, np.random.default_rng(0))
    out = opt.monotonic_basin_hopping(X, 0.2, C, iterations=2, seed=0)
    assert seen["T"] == 0.0
    assert seen["stepsize"] == pytest.approx(opt.MBH_STEP_FACTOR * 0.2)
    assert out.shape == X.shape


def test_legacy_restores_scipy_defaults(monkeypatch):
    seen = _captured(monkeypatch)
    monkeypatch.setattr(opt, "MBH_LEGACY", True)
    C = CONTAINERS["cube"]()
    X = C.generate_initial_solution(4, np.random.default_rng(0))
    opt.monotonic_basin_hopping(X, 0.2, C, iterations=2, seed=0)
    assert "T" not in seen and "stepsize" not in seen
