"""The final LP tightening reaches the local maximum and never loses radius."""
import numpy as np
import pytest

import tsgo.algorithm.energy_functions as ef
from tsgo.algorithm.polish import polish
from tsgo.container.sample_polyhedron import CONTAINERS


def test_two_balls_reach_the_exact_optimum():
    # Optimum: centres on the main diagonal, r = sqrt(3) / (2 + 2 sqrt(3)).
    r = np.sqrt(3.0) / (2.0 + 2.0 * np.sqrt(3.0))
    X = np.array([[-0.5 + r] * 3, [0.5 - r] * 3]) * 0.97           # shrunk inward
    X += 1e-3 * np.random.default_rng(0).standard_normal(X.shape)   # and knocked off the diagonal
    C = CONTAINERS["cube"]()
    assert ef.max_feasible_D(X, C, "EUCLIDEAN") < r * 0.99
    _, D = polish(X, C, "EUCLIDEAN")
    assert D == pytest.approx(r, rel=1e-8)
    assert ef.max_feasible_D(_, C, "EUCLIDEAN") >= D


@pytest.mark.parametrize("metric", ["EUCLIDEAN", "TAXICAB", "SUP_NORM"])
@pytest.mark.parametrize("name", ["cube", "h-box", "pointy-star"])
def test_never_lowers_the_certified_radius(name, metric):
    C = CONTAINERS[name]()
    X = C.generate_initial_solution(8, np.random.default_rng(1))
    D0 = ef.max_feasible_D(X, C, metric)
    X1, D1 = polish(X, C, metric, max_iter=30)
    assert D1 >= D0 * (1.0 - 1e-9)
    assert ef.max_feasible_D(X1, C, metric) >= D1
