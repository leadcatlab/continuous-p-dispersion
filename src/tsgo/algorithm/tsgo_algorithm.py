
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
import numpy as np
from typing import List, Optional, Tuple

from tsgo.algorithm.optimization import local_optimization, local_optimization_phi
from tsgo.container.container import Container
from tsgo.container.point import Point
from tsgo.utils.metric_enums import get_metric_function
import tsgo.algorithm.energy_functions as ef
from tsgo.utils.io import get_datetime

def _tsgo_algorithm(
    p: int, 
    container: Container, 
    D_init: float, 
    metric_name: str = 'EUCLIDEAN', 
    # algorithm control
    beta_max: int = 3, 
    Q: int = 3, 
    eps: float = 1e-16, 
    # optimization
    optimization_method: str = 'BFGS'
) -> Tuple[np.ndarray, float]:
    D = D_init
    X = container.generate_initial_solution(p)  # np.ndarray
    X = tabu_search(X, D, container, beta_max=beta_max, Q=Q, method=optimization_method, metric=metric_name)
    
    if np.isclose(ef.E(X, D, container, metric=metric_name), 0.0, atol=eps):
        X, D = adjust_distance(X, D, container, method=optimization_method, metric=metric_name)
    else:
        X, D = _tsgo_algorithm(p, container, D_init, metric_name, beta_max, Q, eps, optimization_method)
    
    return X, D

# ALGORITHM 1: Main Framework of the TSGO Algorithm
# Output: The best configuration found (X_opt) and the maximal minimum distance between points
# ALGORITHM 1: Main Framework of the TSGO Algorithm
# Output: The best configuration found (X_opt) and the maximal minimum distance between points
def tsgo_algorithm_parallel(
    iterations: int,
    p: int,
    container: Container,
    D_init: float,
    metric_name: str = 'EUCLIDEAN',
    *,
    # algorithm control
    beta_max: int = 3, # tabu-search depth
    Q: int = 3, # number of points to de-energize
    eps: float = 1e-16, # tuning parameter for precision
    # execution
    max_workers: Optional[int] = 6,
    # optimization
    optimization_method: str = 'BFGS'
) -> Tuple[np.ndarray, float]:
    """
    Run `_tsgo_algorithm` in parallel `iterations` times and return the (X, D)
    pair with the largest D.

    Positional args:
        iterations: number of independent runs to perform.
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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _tsgo_algorithm,
                p,
                container,
                D_init,
                metric_name,
                beta_max,
                Q,
                eps,
                optimization_method
            )
            for _ in range(iterations)
        ]

        for fut in as_completed(futures):
            X_i, D_i = fut.result()
            if D_i > best_D:
                best_D = D_i
                best_X = X_i

    return best_X, best_D

# ALGORITHM 2: Tabu Search
# Output: Best solution found through rearrangement of high energy points
def tabu_search(
    X0: np.ndarray, 
    D: float, 
    container: Container, 
    beta_max: int = 3, 
    Q: int = 3, 
    method: str = 'BFGS', 
    metric: str = 'EUCLIDEAN'
) -> List[Point]:
    start_time = time.time()
    Q = min(Q, len(X0))  # Ensure Q does not exceed the number of points in X0
    metric = get_metric_function(metric)
    X = local_optimization(X0, D, container, method=method, metric=metric)
    X_best = X.copy()
    
    no_improve, tabu_list, iterations, I = 0, [], 0, None

    while (no_improve <= beta_max) and (ef.E(X_best, D, container) > 1e-8):
        start_time = time.time()
        high_energy_points = ef.select_high_energy_points(X, D, container, Q, metric=metric)
        low_energy_points = ef.find_vacancy_sites(X, D, container, Q, metric=metric)

        X_best_neighbor = X_best.copy()
        for i in range(Q):
            for j in range(Q):
                X_neighbor = tabu_list.move_point(X, i, low_energy_points[j])
                X_neighbor = local_optimization(X_neighbor, D, container, method=method, metric=metric)
                
                if not tabu_list.forbidden(high_energy_points[i], tabu_list, iterations):
                    if ef.E(X_neighbor, D, container, metric=metric) < ef.E(X_best_neighbor, D, container, metric=metric):
                        X_best_neighbor = X_neighbor.copy()
                        I = (high_energy_points[i], low_energy_points[j])

        X = X_best_neighbor.copy()
        tabu_list = tabu_list.update_tabu_list(I, tabu_list, iterations)
        X = tabu_list.monotonic_basin_hopping(X, D, container, iterations=10, metric=metric)

        # Check if the point locations have changed
        if np.allclose(X_best, X, atol=1e-8):
            no_improve += 1
        else:
            X_best = X.copy()
            no_improve = 0

        iterations += 1
        
        energy = round(ef.E(X_best, D, container, metric=metric), 10)
        elapsed_time = time.time() - start_time

    elapsed_time = time.time() - start_time
    print(f"[TABU SEARCH @ {get_datetime()}] Total time: {elapsed_time:.4f} seconds")
    return X_best

# ALGORITHM 3: Adjust Distance (Adjustment Method for the Minimum Distance)
# Output: Feasible local optimal solution (X, D)
def adjust_distance(
    X0: np.ndarray, 
    D0: float, 
    container: Container, 
    iter_max: int = 15, 
    method: str = 'BFGS', 
    metric: str = 'EUCLIDEAN'
) -> Tuple[np.ndarray, float]:
    start_time = time.time()
    metric = get_metric_function(metric)
    X, D, mu = X0.copy(), D0, 10.0
    no_change_counter = 0
    prev_X = X.copy()
    prev_D = D

    for _ in range(iter_max):
        X_new, D_new = local_optimization_phi(X, D, container, mu, method=method, metric=metric)
        mu *= 10.0

        x_same = np.allclose(prev_X, X_new, atol=1e-8)
        d_same = np.isclose(prev_D, D_new, atol=1e-12)

        if x_same and d_same: no_change_counter += 1
        else: no_change_counter = 0

        prev_X = X_new.copy()
        prev_D = D_new
        X, D = X_new, D_new

        if no_change_counter >= 3:
            elapsed_time = time.time() - start_time
            print(f"[ADJUST DISTANCE @ {get_datetime()}] Completed in {elapsed_time:.4f} seconds -- D: {D}, Quality: {ef.E(X, D, container, metric=metric):.8f}")
            return X, D

    elapsed_time = time.time() - start_time
    print(f"[ADJUST DISTANCE @ {get_datetime()}] Completed in {elapsed_time:.4f} seconds -- D: {D}, Quality: {ef.E(X, D, container, metric=metric):.8f}")
    return X, D