from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from datetime import datetime
import os
import time

import numpy as np
from typing import Dict, Tuple, Optional, Any, Union

from tsgo.configuration import Config
from tsgo.algorithm.optimization import (
    local_optimization,
    local_optimization_phi,
    monotonic_basin_hopping as mbh,
)
from tsgo.algorithm.polish import polish as polish_configuration
from tsgo.container.container import Container
from tsgo.container.sample_polyhedron import CONTAINERS
import tsgo.algorithm.energy_functions as ef
from tsgo.utils.io import get_datetime, save_to_file
from tsgo.utils import tabu_list as tl

def default_Q(p: int) -> int:
    """Tabu neighbourhood size used when none is given: max(3, round(ln p)).

    Each tabu iteration tries Q x Q moves, each followed by a local
    optimisation, so Q trades search breadth against restarts per unit time.
    At a fixed budget Q = 5 beat 3 and 6 on cube l2 p = 100, and Q = 4 beat
    3 in 12 of 24 paired seeds (5 the other way, 7 ties) on the tetrahedron
    and H-box at p = 60 across all three metrics. The rule leaves Q = 3 for
    p <= 33, so small-p behaviour is unchanged; Q = 4 for 34..90, 5 for
    91..244.
    """
    return max(3, int(round(math.log(p)))) if p > 1 else 3


def default_restart_patience(p: int) -> int:
    """Restarts in a row without improvement before a run stops: ceil(3 ln p).

    Chosen by replaying 160 long runs (cube, tetrahedron, H-box, pointy star;
    l2 and l1; p = 10..100; 4 seeds; capped at 41 starts / 30 min) against
    every stopping rule: constant K, a ln p, a sqrt p and a p. ceil(3 ln p)
    (7, 9, 12, 13, 14 at p = 10, 20, 40, 60, 100) used 81% of the capped CPU
    with mean loss 0.003%; only one cell of 40 lost more than 0.01% on its
    best of four seeds (h-box l1 p = 20, 0.06%). Constant K = 12 matched it at
    83%; rules linear in p were worst, starving small p. At large p the time
    cap binds first: p = 100 fits ~18 starts in 30 min and is still improving.
    """
    return max(1, int(math.ceil(3.0 * math.log(p)))) if p > 1 else 1


# A restart counts as an improvement for restart_patience only if it raises the
# best radius by more than this (relative); smaller gains are numerical noise
# around the same optimum.
RESTART_RTOL = 1e-7

# Escape phase (opt-in, off in every published sweep): restarts aimed above the
# best radius, run after the normal restarts have stopped.
ESCAPE_FACTOR = 1.02


def escape_restarts(
    X_opt: np.ndarray,
    D_opt: float,
    p: int,
    container: Container,
    metric: str,
    seeds,
    factor: float,
    patience: int,
    time_limit: Optional[float],
    Q: int,
    beta_max: int,
    eps: float,
    optimization_method: str,
    verbose: bool = False,
    on_start=None,
) -> Tuple[np.ndarray, float, int]:
    """Restarts whose tabu target is factor * D_opt rather than D_opt.

    A normal restart runs tabu at D_opt, and tabu only moves points while the
    energy is positive. A start that is already feasible at D_opt therefore
    returns unmoved, however poor its layout: in Stoyan & Yaskov's example 6
    one sphere left in the thin wall under the hole caps the radius at that
    wall's inradius, 0.93364, and every later restart that lands a point there
    is feasible at once and never pushes it out. Aiming above D_opt makes such
    a start infeasible, so tabu must move it.

    Each start draws a fresh configuration from its own stream in `seeds`, runs
    tabu at factor * D_opt and, if that target is reached, grows the radius
    with adjust_distance. Only a strict gain (RESTART_RTOL) is kept, so the
    phase never lowers D_opt. It stops after `patience` starts in a row
    without a gain, after len(seeds) starts, or once `time_limit` seconds have
    passed (a start under way is finished). `on_start(X_opt, D_opt)` is called
    after every start. Returns (X_opt, D_opt, starts made).
    """
    vprint = print if verbose else lambda *args, **kwargs: None
    t0 = time.perf_counter()
    stalled = starts = 0
    for ss in seeds:
        if time_limit is not None and time.perf_counter() - t0 >= time_limit:
            break
        if stalled >= patience:
            break
        rng = np.random.default_rng(ss)
        target = D_opt * factor
        X = container.generate_initial_solution(p, rng)
        X = tabu_search(X, D=target, container=container, beta_max=beta_max, Q=Q,
                        optimization_method=optimization_method, metric=metric, eps=eps,
                        verbose=verbose, rng=rng)
        improved = False
        if ef.E(X, target, container=container, metric=metric) < eps:
            X, D = adjust_distance(X0=X, D0=target, container=container, metric=metric,
                                   optimization_method=optimization_method, eps=eps,
                                   verbose=verbose)
            if D > D_opt * (1.0 + RESTART_RTOL) and ef.E(X, D, container=container, metric=metric) < eps:
                X_opt, D_opt, improved = X, D, True
        starts += 1
        stalled = 0 if improved else stalled + 1
        vprint(f"[ESCAPE] start {starts}: target={target:.6f} best={D_opt:.6f}")
        if on_start is not None:
            on_start(X_opt, D_opt)
    return X_opt, D_opt, starts


def escape_seeds(seed: int, iterations: int, n: int):
    """The streams escape_restarts uses in a run with this worker seed.

    _tsgo_algorithm spawns, from SeedSequence(seed), one stream per start and
    then one for the ray directions; the escape streams are the next n
    children. Spawning is stateful, so the normal part of a run is unchanged
    by the phase, and a continuation from a saved run (escape.py) draws the
    same streams the run itself would have drawn. Also returns the ray stream,
    which a continuation must reuse so the energy is the run's own."""
    ss = np.random.SeedSequence(seed)
    ss.spawn(max(iterations, 1))
    rays = ss.spawn(1)[0]
    return ss.spawn(n), rays


def _tsgo_algorithm(
    p: int, 
    D_init: float,
    container: Container,
    iterations: int,
    Q: Optional[int],
    beta_max: int,
    eps: float,
    metric: str,
    optimization_method: str,
    output_dir: str,
    verbose: bool,
    seed: Optional[int] = None,
    time_limit: Optional[float] = None,
    trace: Optional[list] = None,
    restart_patience: Optional[int] = None,
    polish: bool = True,
    escape_factor: Optional[float] = None,
    escape_patience: Optional[int] = None,
    escape_time_limit: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """One TSGO run: a start at D_init, then restarts at the best radius so far.

    By default the run makes `iterations` starts in all. With `time_limit`
    (seconds), restarts continue until that much time has passed, capped at
    `iterations`; a start already under way is finished. With `trace`, a list,
    each completed start appends (seconds since the run began, certified best
    radius so far) -- the anytime record the baseline comparison needs. With
    `restart_patience` K, restarts stop once K in a row have failed to raise
    the best radius by more than RESTART_RTOL (relative), still capped at
    `iterations`: small p converges in a few restarts while large p keeps
    improving for many more, so a fixed count is either wasted or too short.
    None of the options above changes what a default run does.

    With `polish` (the default), the best configuration is tightened at the
    end by sequential LP (tsgo.algorithm.polish): SUMT stops a few parts in
    1e5 short of the configuration's local maximum, and the LP closes that.
    It runs once, on the final answer, so the search itself is unchanged, and
    it never lowers the certified radius.

    With `escape_factor` (off by default, and off in every published sweep),
    the polished result then goes through escape_restarts: restarts aimed at
    escape_factor * D_opt, stopping after `escape_patience` in a row without a
    gain (default restart_patience, else ceil(3 ln p)) or `escape_time_limit`
    seconds. Everything before it is unchanged, so the phase can only raise
    the result, and a saved run can be continued with it (escape.py).
    """
    vprint = print if verbose else lambda *args, **kwargs: None
    if isinstance(container, str):
        container = CONTAINERS[container]()
    if Q is None:
        Q = default_Q(p)

    # The trace holds what the run would return if stopped now: with `polish`,
    # the tightened radius of the best configuration so far. Tightening takes
    # well under a second, its time is counted, and the search itself is
    # unchanged -- the copy is polished, not X_opt.
    last = {"X": None, "D": 0.0}

    def _record():
        if trace is None:
            return
        if last["X"] is not X_opt:
            if polish and D_opt > 0.0:
                last["D"] = float(polish_configuration(X_opt.copy(), container, metric)[1])
            else:
                last["D"] = float(ef.max_feasible_D(X_opt, container, metric))
            last["X"] = X_opt
        trace.append((time.perf_counter() - start_time, last["D"]))

    # One independent stream per restart, all derived from `seed`, so a run is
    # reproducible from (seed, p, container) alone and restarts stay decorrelated.
    seed_seq = np.random.SeedSequence(seed)
    restart_seeds = seed_seq.spawn(max(iterations, 1))
    rng = np.random.default_rng(restart_seeds[0])

    # Fresh ray directions for this run, fixed for its duration: E stays
    # deterministic in X while no single alignment biases every run.
    container.set_ray_directions(np.random.default_rng(seed_seq.spawn(1)[0]))

    D = D_init
    X = container.generate_initial_solution(p, rng)  # np.ndarray
    
    # The run's clock -- for time_limit and the trace -- is perf_counter, which
    # stops while the machine sleeps, as the baselines' clocks do. Wall time
    # charged a closed laptop lid's sleep to TSGO's budget alone.
    start_time = time.perf_counter()
    vprint(f"[TSGO] Starting optimization with D_init={D_init:.6f} for p={p} points in container {container.description}.")
    X = tabu_search(
        X0=X,
        D=D,
        container=container,
        Q=Q,
        beta_max=beta_max,
        eps=eps,
        metric=metric,
        optimization_method=optimization_method,
        verbose=verbose,
        rng=rng,
    )
    X, D = adjust_distance(
        X0=X,
        D0=D,
        container=container,
        metric=metric,
        optimization_method=optimization_method,
        eps=eps,
        verbose=verbose,
    )
    
    X_opt, D_opt = X, D
    _record()

    stalled = 0
    for restart in range(1, iterations):
        if time_limit is not None and time.perf_counter() - start_time >= time_limit:
            break
        if restart_patience is not None and stalled >= restart_patience:
            break
        vprint(f"[TSGO] Starting iteration with D={D:.6f}, Energy={ef.E(X, D, container=container, metric=metric):.6e}")

        rng = np.random.default_rng(restart_seeds[restart])
        improved = False
        D = D_opt
        X = container.generate_initial_solution(p, rng)  # np.ndarray
        X = tabu_search(
            X,
            D=D,
            container=container,
            beta_max=beta_max,
            Q=Q,
            optimization_method=optimization_method,
            metric=metric,
            eps=eps,
            verbose=verbose,
            rng=rng,
        )
        if ef.E(X, D, container=container, metric=metric) < eps:
            X, D = adjust_distance(
                X0=X,
                D0=D,
                container=container,
                metric=metric,
                optimization_method=optimization_method,
                eps=eps,
                verbose=verbose,
            )
            if D > D_opt and ef.E(X, D, container=container, metric=metric) < eps:
                improved = D > D_opt * (1.0 + RESTART_RTOL)
                X_opt, D_opt = X, D
        stalled = 0 if improved else stalled + 1
        _record()

    if polish and D_opt > 0.0:
        X_opt, D_opt = polish_configuration(X_opt, container, metric)

    if escape_factor and D_opt > 0.0:
        k = escape_patience or restart_patience or default_restart_patience(p)
        # The streams after the run's own, as escape_seeds reproduces them.
        seeds = seed_seq.spawn(max(iterations, 1))

        def _escape_record(X_e, D_e):
            nonlocal X_opt, D_opt
            X_opt, D_opt = X_e, D_e
            _record()

        D_before = D_opt
        X_opt, D_opt, _ = escape_restarts(
            X_opt, D_opt, p, container, metric, seeds, escape_factor, k, escape_time_limit,
            Q, beta_max, eps, optimization_method, verbose=verbose, on_start=_escape_record)
        if polish and D_opt > D_before:
            X_opt, D_opt = polish_configuration(X_opt, container, metric)

    elapsed_time = time.perf_counter() - start_time
    
    # Save the X, D, E values to file in case of crash
    output_dir = os.path.join(
        output_dir,
        'crashs', 
        f'{D_opt:.6f}'
    )
    
    save_to_file(
        p=p,
        parallel_iterations=1,
        iterations=iterations,
        D_init=D_init,
        X_opt=X_opt,
        D_opt=D_opt,
        container=container,
        elapsed_time=elapsed_time,
        metric_name=metric,
        output_dir=output_dir,
        Q=Q,
        beta_max=beta_max,
        eps=eps,
        optimization_method=optimization_method,
        seed=seed,
    )
    E = ef.E(X_opt, D_opt, container=container, metric=metric)
    print(f"[TSGO @ {get_datetime()}] [{elapsed_time:.2f}s] Completed process for p={p} in {container.description} | D_opt={D_opt:.6f} | E={E:.6e}")
    return X_opt, D_opt, E


class NoFeasibleConfiguration(RuntimeError):
    """No worker ended with a feasible configuration. A RuntimeError, as the
    bare exception raised here was before, so existing handlers still catch it."""


def _tsgo_traced(**kw):
    """_tsgo_algorithm with its trace returned, for a worker process: a list
    passed in would be filled in the worker's copy, not the caller's."""
    tr: list = []
    X, D, E = _tsgo_algorithm(trace=tr, **kw)
    return X, D, E, tr


# ALGORITHM 1: Main Framework of the TSGO Algorithm
# Output: The best configuration found (X_opt) and the maximal minimum distance between points
def tsgo_algorithm_parallel(
    p: int,
    D_init: float,
    container: Container,
    Q: Optional[int],
    beta_max: int,
    eps: float,
    metric: str,
    iterations: int,
    processes: int,
    optimization_method: str,
    output_dir: str,
    verbose: bool,
    seed: Optional[int] = None,
    restart_patience: Optional[int] = None,
    polish: bool = True,
    time_limit: Optional[float] = None,
    trace: Optional[list] = None,
    escape_factor: Optional[float] = None,
    escape_patience: Optional[int] = None,
    escape_time_limit: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Run `_tsgo_algorithm` in parallel `parallel_iterations` times and return the (X, D)
    pair with the largest D.

    Positional args:
        parallel_iterations: number of independent runs to perform.
        p: number of points (passed to _tsgo_algorithm).
        container: Container instance for the problem.
        D_init: initial minimum distance.

    Keyword-only args (organized by purpose):
        beta_max, Q, eps: algorithm control parameters.
        max_workers: number of processes to use (None lets Python decide).
        optimization_method: local optimization method name.

    Returns:
        (X_best, D_best): The solution with the largest D across all runs.

    With `trace`, a list, it is filled with the winning worker's trace: one
    (seconds, radius) per start, the radius being what that worker would have
    returned had it stopped there. The search is unchanged.
    """
    best_X: Optional[np.ndarray] = None
    best_D: float = -np.inf
    if Q is None:
        Q = default_Q(p)

    # Distinct, reproducible stream per worker. Workers are spawned on macOS, so
    # global RNG state would not carry over anyway -- seeds must be explicit.
    worker_seeds = np.random.SeedSequence(seed).spawn(processes)
    
    vprint = print if verbose else lambda *args, **kwargs: None
    
    vprint(f'[TSGO] Launching {processes} parallel processes | p={p} | D_init={D_init:.6f} | Q={Q} | beta_max={beta_max} | eps={eps:.2e} | metric={metric} | optimization_method={optimization_method}')

    with ProcessPoolExecutor(max_workers=processes) as executor:
        futures = [
            executor.submit(
                _tsgo_algorithm if trace is None else _tsgo_traced,
                p=p,
                D_init=D_init,
                container=container,
                iterations=iterations,
                Q=Q,
                beta_max=beta_max,
                eps=eps,
                metric=metric,
                optimization_method=optimization_method,
                output_dir=output_dir,
                verbose=verbose,
                seed=int(ws.generate_state(1)[0]),
                restart_patience=restart_patience,
                polish=polish,
                time_limit=time_limit,
                escape_factor=escape_factor,
                escape_patience=escape_patience,
                escape_time_limit=escape_time_limit,
            )
            for ws in worker_seeds
        ]

        best_tr: list = []
        for fut in as_completed(futures):
            res = fut.result()
            X_i, D_i, E_i = res[:3]
            if D_i > best_D and np.isclose(E_i, 0.0, atol=eps):
                best_D = D_i
                best_X = X_i
                best_tr = res[3] if trace is not None else []
        if trace is not None:
            trace[:] = best_tr

    if best_X is None:
        # No worker reached feasibility; surface it rather than returning None,
        # which previously crashed downstream in run_tsgo's energy report.
        raise NoFeasibleConfiguration(
            f"No feasible configuration found for p={p} in {getattr(container, 'description', container)} "
            f"(D_init={D_init}, eps={eps}, processes={processes}, seed={seed}). "
            f"Try a smaller D_init or more iterations."
        )

    return best_X, best_D

# ALGORITHM 2: Tabu Search
# Output: Best solution found through rearrangement of high energy points
def tabu_search(
    X0: np.ndarray,
    D: float,
    container: Container,
    Q: int,
    beta_max: int,
    eps: float,
    metric: str,
    optimization_method: str,
    verbose: bool,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Tabu search with aspiration + continuous-move tabu.

    rng seeds vacancy-site sampling and basin hopping; pass one for
    reproducible runs.
    """
    if rng is None:
        rng = np.random.default_rng()
    vprint = print if verbose else lambda *args, **kwargs: None
    vprint(f"[TABU SEARCH @ {get_datetime()}] Starting procedure.")
    
    tabu_tenure = 7 
    tabu_max_size = 20 
    forbid_reverse = True 
    
    start_time = time.time()
    Q = min(Q, len(X0))
    delta = max(D / 20.0, 1e-12)
    tabu_list: Dict[Tuple[Any, ...], int] = {}

    # --- Initialize ---
    X = local_optimization(X0, D, container, method=optimization_method, metric=metric, verbose=verbose)
    X_best = X.copy()
    best_energy = ef.E(X_best, D, container, metric=metric)

    no_improve = 0
    it = 0

    while (no_improve <= beta_max) and (best_energy > eps):
        vprint(f"[TABU SEARCH @ {get_datetime()}] No improvement count: {no_improve}/{beta_max} | Best Energy: {best_energy:.8f}")
        iteration_start_time = time.time()

        high_energy_indices = ef.select_high_energy_points(X, D, container, Q, metric=metric)
        low_energy_points = ef.find_vacancy_sites(X, D, container, Q, metric=metric, rng=rng)

        # CRITICAL: Initialize variables BEFORE the loops to avoid UnboundLocalError
        X_best_neighbor = X.copy()
        best_neighbor_energy = float('inf')
        chosen_move_key = None
        chosen_reverse_key = None

        for idx_to_move in high_energy_indices:
            # Pre-calculate previous cell for reverse tabu check
            prev_cell = tl.quantize(np.array(X[idx_to_move]), delta)
            
            for j in range(len(low_energy_points)):
                v_arr = np.array(low_energy_points[j])
                dest_cell = tl.quantize(v_arr, delta)
                
                move_key = ("move", idx_to_move, dest_cell)
                reverse_key = ("move", idx_to_move, prev_cell)

                # Generate trial neighbor
                X_trial = X.copy()
                X_trial[idx_to_move] = v_arr
                X_trial = local_optimization(X_trial, D, container, method=optimization_method, metric=metric, verbose=verbose)
                
                trial_energy = ef.E(X_trial, D, container, metric=metric)

                # Aspiration & Tabu logic
                aspired = trial_energy < best_energy
                tabu_hit = tl.is_tabu(tabu_list, move_key, it)
                if forbid_reverse:
                    tabu_hit = tabu_hit or tl.is_tabu(tabu_list, reverse_key, it)

                if (not tabu_hit) or aspired:
                    if trial_energy < best_neighbor_energy:
                        best_neighbor_energy = trial_energy
                        X_best_neighbor = X_trial.copy()
                        chosen_move_key = move_key
                        chosen_reverse_key = reverse_key

        # If we found an admissible neighbor, move to it
        if chosen_move_key is not None:
            X = X_best_neighbor.copy()
            tl.tabu_add(tabu_list, chosen_move_key, it, tabu_tenure, tabu_max_size)
            if forbid_reverse and (chosen_reverse_key is not None):
                tl.tabu_add(tabu_list, chosen_reverse_key, it, tabu_tenure, tabu_max_size)
        
        # Refine with MBH
        X = mbh(X, D, container, iterations=10, metric=metric,
                seed=int(rng.integers(0, 2**32 - 1)))
        current_energy = ef.E(X, D, container, metric=metric)

        # Update global best
        if current_energy < best_energy:
            X_best = X.copy()
            best_energy = current_energy
            no_improve = 0
        else:
            no_improve += 1

        it += 1
        iteration_elapsed_time = time.time() - iteration_start_time
        vprint(f"[TABU SEARCH @ {get_datetime()}] Iteration {it} done in {iteration_elapsed_time:.2f}s")

    elapsed_time = time.time() - start_time
    vprint(f"[TABU SEARCH] Finished. Time: {elapsed_time:.2f}s | Final Best Energy: {best_energy:.8f}")
    return X_best

# ALGORITHM 3: Adjust Distance (Adjustment Method for the Minimum Distance)
# Output: Feasible local optimal solution (X, D)
def adjust_distance(
    X0: np.ndarray, 
    D0: float, 
    container: Container, 
    metric: str,
    optimization_method: str, 
    eps: float,
    verbose: bool,
    mu: float = 1.0,
    iter_max: int = 15,
    _depth: int = 0,
) -> Tuple[np.ndarray, float]:
    
    X, D, mu = X0.copy(), D0, mu
    
    no_change_counter = 0
    prev_X = X.copy()
    prev_D = D
    
    iterations = 0
    
    vprint = print if verbose else lambda *args, **kwargs: None

    start_time = time.time()
    while iterations < iter_max:
        mu *= 10
        
        iteration_start_time = time.time()
        X_new, D_new = local_optimization_phi(X, D, container, mu, method=optimization_method, metric=metric, verbose=verbose)

        x_same = np.allclose(prev_X, X_new, atol=eps)
        d_same = np.isclose(prev_D, D_new, atol=eps)

        if x_same and d_same: 
            no_change_counter += 1
        else: 
            no_change_counter = 0

        prev_X = X_new.copy()
        prev_D = D_new
        X, D = X_new, D_new

        iteration_elapsed_time = time.time() - iteration_start_time
        decimal_places = max(0, -int(np.log10(eps)) + 1)
        energy = round(ef.E(X, D, container, metric=metric), decimal_places)
        phi = round(ef.phi(X, D, mu, container, metric=metric), decimal_places)
        vprint(f"[ADJUST DISTANCE @ {get_datetime()}] Completed {iterations+1}/{iter_max} in {iteration_elapsed_time:.4f}s | D: {D:.10f} | E: {energy:.8f} | phi: {phi:.8f}")

        if no_change_counter >= 3:
            break

        iterations += 1

    if ef.E(X, D, container, metric=metric) >= eps:
        vprint(f'[ADJUST DISTANCE @ {get_datetime()}] Solution not feasible (E={ef.E(X, D, container, metric=metric):.8f} >= eps={eps:.2e}). Reducing D and retrying.')
        # Snap D down to the largest radius this configuration actually
        # supports rather than walking down in 5% steps. The bound is exact,
        # so a D_init above the optimum is recovered in one step instead of
        # running out of retries still infeasible -- which is what made the
        # h-box and the cage collapse to D=0 at p=1. If a point has escaped
        # the container no positive radius is feasible, and the old
        # percentage step is the only thing left to try.
        D_feas = ef.max_feasible_D(X, container, metric=metric)
        D = min(D * 0.95, D_feas * (1.0 - 1e-9)) if D_feas > 0.0 else D * 0.95
        # Bounded retry: unbounded recursion here made run times heavy-tailed.
        if _depth < 5:
            X = local_optimization(X, D, container, method=optimization_method, metric=metric, verbose=verbose)
            X, D = adjust_distance(X, D, container, metric=metric, optimization_method=optimization_method, eps=eps, verbose=verbose, mu=1000.0, iter_max=5, _depth=_depth + 1)
        else:
            vprint(f'[ADJUST DISTANCE @ {get_datetime()}] Retry depth limit reached; returning best available solution.')

    # Return exactly the radius the configuration supports, whichever side of
    # it SUMT stopped on. Below it: D = 0 is a stationary point of phi at the
    # lower bound whenever the points are well separated (E = 0 and
    # dphi/dD = -2D = 0 there), so SUMT can settle on a trivially "feasible"
    # answer. Above it: E < eps admits a violation of up to sqrt(eps), and
    # the sweep reported radii up to 5.7e-6 (relative) beyond what their
    # configurations support. With D = B(X) every reported radius is
    # certified by the exact bound rather than by a tolerance. B(X) = 0 means
    # a point is outside, where no radius is feasible; D is left alone there
    # so the caller sees the infeasibility.
    D_feas = ef.max_feasible_D(X, container, metric=metric)
    if D_feas > 0.0:
        D = D_feas * (1.0 - 1e-9)
        vprint(f'[ADJUST DISTANCE @ {get_datetime()}] Set D to the exact feasible bound {D:.10f}.')

    elapsed_time = time.time() - start_time
    vprint(f"[ADJUST DISTANCE @ {get_datetime()}] Finished in {elapsed_time:.4f} seconds | Final D: {D:.6f} | Final E: {ef.E(X, D, container, metric=metric):.8f}")
    return X, D