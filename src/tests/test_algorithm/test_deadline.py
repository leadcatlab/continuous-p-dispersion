"""The time limit holds inside a start: Tabu Search stops at its deadline."""

import time

import numpy as np

import tsgo.algorithm.energy_functions as ef
from tsgo.algorithm.tsgo_algorithm import _tsgo_algorithm, escape_restarts, tabu_search
from tsgo.container.sample_polyhedron import CONTAINERS


def _tabu(deadline):
    C = CONTAINERS['cube']()
    rng = np.random.default_rng(0)
    X0 = C.generate_initial_solution(20, rng)
    # A radius no configuration of 20 points supports: the search never
    # reaches E < eps and, without a deadline, runs its full patience.
    t0 = time.perf_counter()
    X = tabu_search(
        X0,
        D=0.3,
        container=C,
        Q=3,
        beta_max=10,
        eps=1e-8,
        metric='EUCLIDEAN',
        optimization_method='L-BFGS-B',
        verbose=False,
        rng=rng,
        deadline=deadline,
    )
    return X, time.perf_counter() - t0


def test_tabu_search_stops_at_a_past_deadline():
    X, wall = _tabu(deadline=time.perf_counter())
    # Only the initial local optimization runs.
    assert X.shape == (20, 3)
    assert wall < 5.0


def test_run_ends_near_its_time_limit(tmp_path):
    C = CONTAINERS['cube']()
    t0 = time.perf_counter()
    # A first start at a feasible radius ends quickly; the restarts at the
    # best radius are what the limit has to stop.
    D_init = ef.initial_radius(C, 20, 0.25, 'EUCLIDEAN')
    _, D, E = _tsgo_algorithm(
        p=20,
        D_init=D_init,
        container=C,
        iterations=1000,
        Q=3,
        beta_max=10,
        eps=1e-8,
        metric='EUCLIDEAN',
        optimization_method='L-BFGS-B',
        output_dir=str(tmp_path),
        verbose=False,
        seed=0,
        time_limit=3.0,
    )
    wall = time.perf_counter() - t0
    # Within one local optimization of the limit, plus Adjust Distance and the polish.
    assert wall < 3.0 + 30.0
    assert D > 0.0 and E < 1e-8


def test_escape_phase_stops_at_its_time_limit():
    C = CONTAINERS['cube']()
    C.set_ray_directions(np.random.default_rng(0))
    X = C.generate_initial_solution(20, np.random.default_rng(0))
    D = float(ef.max_feasible_D(X, C, 'EUCLIDEAN'))
    seeds = np.random.SeedSequence(0).spawn(100)
    t0 = time.perf_counter()
    _, D_out, _ = escape_restarts(
        X,
        D,
        20,
        C,
        'EUCLIDEAN',
        seeds,
        1.5,
        patience=10**9,
        time_limit=2.0,
        Q=3,
        beta_max=10,
        eps=1e-8,
        optimization_method='L-BFGS-B',
    )
    assert time.perf_counter() - t0 < 2.0 + 30.0
    assert D_out >= D
