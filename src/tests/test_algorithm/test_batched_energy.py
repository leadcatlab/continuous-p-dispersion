"""The batched energy must equal the per-point loop to the last bit.

Not approximately: L-BFGS amplifies a one-ulp difference in E until same-seed
runs end at different optima, so anything short of exact equality breaks
reproducing a result from its seed.
"""
import numpy as np
import pytest

import tsgo.algorithm.energy_functions as ef
import tsgo.container.container as cc
from tsgo.container.sample_polyhedron import CONTAINERS


@pytest.fixture(params=[False, True], ids=["rays", "grid"])
def grid(request, monkeypatch):
    """Run each test with and without the inside/outside grid."""
    monkeypatch.setattr(cc, "USE_GRID", request.param)
    return request.param


def _configs(container, p, rng):
    """Interior, partly escaped, and near-vertex configurations."""
    bb = container._bounding_box()
    lo = np.array([bb["min_x"], bb["min_y"], bb["min_z"]])
    hi = np.array([bb["max_x"], bb["max_y"], bb["max_z"]])
    V = np.array([v.as_array() for v in container.vertices])
    k = max(1, p // 4)

    inside = container.generate_initial_solution(p, rng)
    escaped = inside.copy()
    escaped[:k] = lo - 0.05 * (hi - lo) + rng.random((k, 3)) * 1.1 * (hi - lo)
    near = inside.copy()
    near[:k] = (V[rng.integers(len(V), size=k)] * (1 - 1e-3) + 1e-3 * near[:k]
                + 1e-9 * rng.standard_normal((k, 3)))
    return inside, escaped, near


@pytest.mark.parametrize("mode", ["sum", "min"])
@pytest.mark.parametrize("metric", ["EUCLIDEAN", "TAXICAB", "SUP_NORM"])
@pytest.mark.parametrize("name", sorted(CONTAINERS))
def test_batched_energy_matches_loop_exactly(name, metric, mode, monkeypatch, grid):
    monkeypatch.setattr(ef, "BOUNDARY_MODE", mode)
    container = CONTAINERS[name]()
    rng = np.random.default_rng(0)
    container.set_ray_directions(rng)
    for p in (1, 7, 20):
        D0 = ef.initial_radius(container, p, 0.25, metric)
        for X in _configs(container, p, rng):
            for D in (D0, 0.3 * D0):
                # Small steps after the first reuse the pair list; the
                # large one at the end forces a rebuild.
                for step in (0.0, 1e-4, 1e-3, 0.3):
                    Y = X + step * D * rng.standard_normal(X.shape)
                    e0, g0, d0 = ef._E_and_grad_loop(Y, D, container, metric)
                    e1, g1, d1 = ef._E_and_grad_batched(Y, D, container, metric)
                    assert e1 == e0
                    assert d1 == d0
                    assert np.array_equal(g1, g0)
                    assert (ef._container_penalty_batched(Y, D, container, metric)
                            == ef._container_penalty_loop(Y, D, container, metric))


@pytest.mark.parametrize("name", sorted(CONTAINERS))
def test_contains_many_matches_contains(name, grid):
    container = CONTAINERS[name]()
    rng = np.random.default_rng(1)
    container.set_ray_directions(rng)
    bb = container._bounding_box()
    lo = np.array([bb["min_x"], bb["min_y"], bb["min_z"]])
    hi = np.array([bb["max_x"], bb["max_y"], bb["max_z"]])
    X = lo - 0.1 * (hi - lo) + rng.random((2000, 3)) * 1.2 * (hi - lo)
    V = np.array([v.as_array() for v in container.vertices])
    X = np.vstack([X, V, V + 1e-9 * rng.standard_normal(V.shape)])
    expected = container._contains_many_raycast(X)
    assert np.array_equal(container.contains_many(X), expected)
    container._contains_cache.clear()
    assert np.array_equal([container.contains(x) for x in X], expected)
