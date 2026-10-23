"""The constrained-program baselines and the pieces they rely on."""
import numpy as np
import pytest

import tsgo.algorithm.energy_functions as ef
from tsgo.container.sample_polyhedron import CONTAINERS
from tsgo.experiments.cop_baselines import Program

NAMES = ["cube", "tetrahedron", "h-box", "pointy-star", "cage"]
METRICS = ["EUCLIDEAN", "TAXICAB", "SUP_NORM"]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("metric", METRICS)
def test_signed_clearance_is_the_exact_bound_inside_and_negative_outside(name, metric):
    C = CONTAINERS[name]()
    rng = np.random.default_rng(0)
    for x in C.generate_initial_solution(20, rng):
        s, _ = ef.signed_clearance_and_grad(x, C, metric)
        assert s == pytest.approx(ef.max_feasible_D(x[None], C, metric), abs=1e-12)
    V = np.array([v.as_array() for v in C.vertices])
    outside = V.max(axis=0) + 0.1 * (V.max(axis=0) - V.min(axis=0))
    assert ef.signed_clearance_and_grad(outside, C, metric)[0] < 0


@pytest.mark.parametrize("name", ["cube", "cage"])
@pytest.mark.parametrize("metric", METRICS)
def test_program_jacobian_matches_finite_differences(name, metric):
    P = Program(CONTAINERS[name](), name, 5, metric)
    z = P.start(np.random.default_rng(1))
    J, h = P.jacobian(z), 1e-7 * P.extent
    fd = np.column_stack([(P.constraints(z + h * e) - P.constraints(z - h * e)) / (2 * h)
                          for e in np.eye(P.n)])
    assert np.abs(J - fd).max() < 1e-5


def test_tsgo_trace_records_one_entry_per_start_without_changing_the_run():
    from tsgo.algorithm.tsgo_algorithm import _tsgo_algorithm
    kw = dict(p=5, D_init=0.2, iterations=3, Q=3, beta_max=10, eps=1e-8, metric="EUCLIDEAN",
              optimization_method="L-BFGS-B", output_dir="/tmp/tsgo_trace_test", verbose=False, seed=7)
    X1, D1, _ = _tsgo_algorithm(container=CONTAINERS["cube"](), **kw)
    tr = []
    X2, D2, _ = _tsgo_algorithm(container=CONTAINERS["cube"](), trace=tr, **kw)
    assert D1 == D2 and np.array_equal(X1, X2)
    # Each entry is what the run would return if stopped there -- the tightened
    # best so far -- so the last is exactly what it did return. (Tightening two
    # incumbents can land in different local maxima, so entries need not rise.)
    assert len(tr) == 3 and all(d > 0 for _, d in tr) and tr[-1][1] == D2
