from concurrent.futures import ProcessPoolExecutor, as_completed
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
from tsgo.container.container import Container
from tsgo.container.sample_polyhedron import CONTAINERS
import tsgo.algorithm.energy_functions as ef
from tsgo.utils.io import get_datetime, save_to_file
from tsgo.utils import tabu_list as tl

def _tsgo_algorithm(
    p: int, 
    D_init: float,
    container: Container,
    iterations: int,
    Q: int,
    beta_max: int,
    eps: float,
    metric: str,
    optimization_method: str,
    output_dir: str,
    verbose: bool,
) -> Tuple[np.ndarray, float, float]:
    vprint = print if verbose else lambda *args, **kwargs: None
    if isinstance(container, str):
        container = CONTAINERS[container]()
    
    D = D_init
    X = container.generate_initial_solution(p)  # np.ndarray
    
    start_time = time.time()
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
    
    for _ in range(iterations - 1):
        vprint(f"[TSGO] Starting iteration with D={D:.6f}, Energy={ef.E(X, D, container=container, metric=metric):.6e}")
        
        D = D_opt
        X = container.generate_initial_solution(p)  # np.ndarray
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
                X_opt, D_opt = X, D
        
    elapsed_time = time.time() - start_time
    
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
    )
    E = ef.E(X_opt, D_opt, container=container, metric=metric)
    print(f"[TSGO @ {get_datetime()}] [{elapsed_time:.2f}s] Completed process for p={p} in {container.description} | D_opt={D_opt:.6f} | E={E:.6e}")
    return X_opt, D_opt, E


# ALGORITHM 1: Main Framework of the TSGO Algorithm
# Output: The best configuration found (X_opt) and the maximal minimum distance between points
def tsgo_algorithm_parallel(
    p: int,
    D_init: float,
    container: Container,
    Q: int,
    beta_max: int,
    eps: float,
    metric: str,
    iterations: int,
    processes: int,
    optimization_method: str,
    output_dir: str,
    verbose: bool,
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
    """
    best_X: Optional[np.ndarray] = None
    best_D: float = -np.inf
    
    vprint = print if verbose else lambda *args, **kwargs: None
    
    vprint(f'[TSGO] Launching {processes} parallel processes | p={p} | D_init={D_init:.6f} | Q={Q} | beta_max={beta_max} | eps={eps:.2e} | metric={metric} | optimization_method={optimization_method}')

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                _tsgo_algorithm,
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
            )
            for _ in range(processes)
        ]

        for fut in as_completed(futures):
            X_i, D_i, E_i = fut.result()
            if D_i > best_D and np.isclose(E_i, 0.0, atol=eps):
                best_D = D_i
                best_X = X_i

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
) -> np.ndarray:
    """
    Tabu search with aspiration + continuous-move tabu.
    """
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
        low_energy_points = ef.find_vacancy_sites(X, D, container, Q, metric=metric)

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
        X = mbh(X, D, container, iterations=10, metric=metric)
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
        # Decrease D slightly and retry
        vprint(f'[ADJUST DISTANCE @ {get_datetime()}] Solution not feasible (E={ef.E(X, D, container, metric=metric):.8f} >= eps={eps:.2e}). Reducing D and retrying.')
        D *= 0.95
        X = local_optimization(X, D, container, method=optimization_method, metric=metric, verbose=verbose)
        X, D = adjust_distance(X, D, container, metric=metric, optimization_method=optimization_method, eps=eps, verbose=verbose, mu=1000.0, iter_max=5)

    elapsed_time = time.time() - start_time
    vprint(f"[ADJUST DISTANCE @ {get_datetime()}] Finished in {elapsed_time:.4f} seconds | Final D: {D:.6f} | Final E: {ef.E(X, D, container, metric=metric):.8f}")
    return X, D