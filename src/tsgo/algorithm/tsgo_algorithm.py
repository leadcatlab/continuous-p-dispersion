
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import time
import numpy as np
from typing import Dict, Tuple, Optional, Any, List


from tsgo.algorithm.optimization import local_optimization, local_optimization_phi, monotonic_basin_hopping as mbh
from tsgo.container.container import Container
from tsgo.utils.metric_enums import get_metric_function
import tsgo.algorithm.energy_functions as ef
from tsgo.utils.io import get_datetime
from tsgo.utils import tabu_list as tl

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
    optimization_method: str = 'L-BFGS-B'
) -> Tuple[np.ndarray, float]:
    D = D_init
    X = container.generate_initial_solution(p)  # np.ndarray
    X = tabu_search(X, D, container, beta_max=beta_max, Q=Q, method=optimization_method, metric=metric_name)
    
    if np.isclose(ef.E(X, D, container, metric=metric_name), 0.0, atol=eps):
        X, D = adjust_distance(X, D, container, method=optimization_method, metric=metric_name, tol=eps)
    else:
        X, D = _tsgo_algorithm(p, container, D_init, metric_name, beta_max, Q, eps, optimization_method)
    
    return X, D, ef.E(X, D, container, metric=metric_name)

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
            X_i, D_i, E_i = fut.result()
            print(f"[TSGO PARALLEL] Completed a run with D={D_i}, E={E_i}")
            if D_i > best_D and np.isclose(E_i, 0.0, atol=eps):
                best_D = D_i
                best_X = X_i

    return best_X, best_D

# ALGORITHM 2: Tabu Search
# Output: Best solution found through rearrangement of high energy points
def tabu_search(
    X0: np.ndarray,
    D: float,
    container: "Container",
    beta_max: int = 3,
    Q: int = 3,
    method: str = "BFGS",
    metric: str = "EUCLIDEAN",
    eps: float = 1e-16,
    *,
    tabu_tenure: int = 10,
    tabu_max_size: int = 200,
    # quantization scale for continuous vacancy points:
    # choose something tied to D so it scales with the problem
    delta: Optional[float] = None,
    # also forbid immediate reverse move ("undo") at coarse level
    forbid_reverse: bool = True,
) -> np.ndarray:
    """
    Tabu search with aspiration + continuous-move tabu.
    - Tabu attribute: (i, dest_cell) where dest_cell is a quantized bin of the destination.
    - Aspiration: allow tabu move if it improves best-so-far energy.
    """

    print(f"[TABU SEARCH @ {get_datetime()}] Starting.")
    start_time = time.time()

    Q = min(Q, len(X0))
    metric_fn = get_metric_function(metric)

    # Pick a quantization step if none given.
    # Rule of thumb: a fraction of D so "repeating" means "basically same relocation".
    if delta is None:
        delta = max(D / 20.0, 1e-12)

    # --- helpers (local, so you don't have to modify your tabu_list.py) ---
    # We store tabu entries in an insertion-ordered dict: key -> expiry_iter
    tabu_list: Dict[Tuple[Any, ...], int] = {}

    # --- initialize ---
    X = local_optimization(X0, D, container, method=method, metric=metric_fn)
    X_best = X.copy()
    best_energy = ef.E(X_best, D, container, metric=metric_fn)

    no_improve = 0
    it = 0

    while (no_improve <= beta_max) and (best_energy > eps):
        print(f"[TABU SEARCH @ {get_datetime()}] Beginning iteration {it}, Current Energy: {best_energy:.8f}")
        print(f"[TABU SEARCH @ {get_datetime()}] No improvement count: {no_improve}/{beta_max}")
        if tabu_list:
            print(f"[TABU SEARCH @ {get_datetime()}] Current tabu list {len(tabu_list)}: {tabu_list}")
        iteration_start_time = time.time()

        # NOTE: use *current* X (not X_best) for selecting high/low energy points, as you already do
        high_energy_points = ef.select_high_energy_points(X, D, container, Q, metric=metric_fn)
        low_energy_points  = ef.find_vacancy_sites(X, D, container, Q, metric=metric_fn)

        # We'll choose the best admissible neighbor (admissible = not tabu OR aspiration)
        X_best_neighbor = X_best.copy()
        best_neighbor_energy = ef.E(X_best_neighbor, D, container, metric=metric_fn)

        # Track best move for tabu update
        chosen_move_key: Optional[Tuple[Any, ...]] = None
        chosen_reverse_key: Optional[Tuple[Any, ...]] = None

        # Important: continuous vacancies => use a key based on index + quantized destination
        for i in range(Q):
            # If your energy functions return Point objects, you likely also know the index.
            # In your current code, you move by index i into X, so we keep that.
            for j in range(Q):
                v = low_energy_points[j]
                # ensure v is np array (3,) for quantization
                v_arr = np.array(v)

                dest_cell = tl.quantize(v_arr, delta)
                move_key = ("move", i, dest_cell)

                # Optional: forbid reverse as well (coarse)
                # Reverse is "move some point into the old cell of point i".
                # We approximate reverse as "move i back near its previous cell".
                prev_cell = tl.quantize(np.array(X[i]), delta)
                reverse_key = ("move", i, prev_cell)

                # Construct neighbor by relocating point i to vacancy v
                X_neighbor = X.copy()
                X_neighbor[i] = v_arr
                X_neighbor = local_optimization(X_neighbor, D, container, method=method, metric=metric_fn)

                neigh_energy = ef.E(X_neighbor, D, container, metric=metric_fn)

                # Aspiration: allow tabu if it improves global best
                aspired = neigh_energy < best_energy

                # Tabu check(s)
                tabu_hit = tl.is_tabu(tabu_list, move_key, it)
                if forbid_reverse:
                    tabu_hit = tabu_hit or tl.is_tabu(tabu_list, reverse_key, it)

                if (not tabu_hit) or aspired:
                    if neigh_energy < best_neighbor_energy:
                        best_neighbor_energy = neigh_energy
                        X_best_neighbor = X_neighbor.copy()
                        chosen_move_key = move_key
                        chosen_reverse_key = reverse_key

        # Move to best neighbor found (even if it doesn't improve global best)
        X = X_best_neighbor.copy()

        # Update tabu with chosen move(s)
        # (if no move was admissible, we skip tabu update)
        if chosen_move_key is not None:
            tl.tabu_add(tabu_list, chosen_move_key, it, tabu_tenure, tabu_max_size)
            if forbid_reverse and (chosen_reverse_key is not None):
                tl.tabu_add(tabu_list, chosen_reverse_key, it, tabu_tenure, tabu_max_size)

        # Your existing MBH step (kept)
        X = mbh(X, D, container, iterations=10, metric=metric_fn)

        # Update best tracking + no_improve
        if best_neighbor_energy >= best_energy:
            no_improve += 1
        else:
            X_best = X_best_neighbor.copy()
            best_energy = best_neighbor_energy
            no_improve = 0

        it += 1
        iteration_elapsed_time = time.time() - iteration_start_time
        print(f"[TABU SEARCH @ {get_datetime()}] Iteration {it} completed in {iteration_elapsed_time:.4f} seconds -- Energy: {round(best_energy, 10)}\n")

    elapsed_time = time.time() - start_time
    print(f"[TABU SEARCH @ {get_datetime()}] Total time: {elapsed_time:.4f} seconds")
    return X_best

# ALGORITHM 3: Adjust Distance (Adjustment Method for the Minimum Distance)
# Output: Feasible local optimal solution (X, D)
def adjust_distance(
    X0: np.ndarray, 
    D0: float, 
    container: Container, 
    mu: float = 1.0,
    iter_max: int = 15, 
    method: str = 'BFGS', 
    metric: str = 'EUCLIDEAN',
    tol: float=1e-16
) -> Tuple[np.ndarray, float]:
    start_time = time.time()
    
    metric_fn = get_metric_function(metric)
    X, D, mu = X0.copy(), D0, 1
    
    no_change_counter = 0
    prev_X = X.copy()
    prev_D = D
    
    iterations = 0

    while iterations < iter_max:
        mu *= 10
        
        print(f"[ADJUST DISTANCE @ {get_datetime()}] Starting iteration with D: {D}, mu: {mu}")
        iteration_start_time = time.time()
        X_new, D_new = local_optimization_phi(X, D, container, mu, method=method, metric=metric_fn)

        x_same = np.allclose(prev_X, X_new, atol=1e-8)
        d_same = np.isclose(prev_D, D_new, atol=1e-12)

        if x_same and d_same: no_change_counter += 1
        else: no_change_counter = 0

        prev_X = X_new.copy()
        prev_D = D_new
        X, D = X_new, D_new

        iteration_elapsed_time = time.time() - iteration_start_time
        decimal_places = max(0, -int(np.log10(tol)) + 1)
        energy = round(ef.E(X, D, container, metric=metric_fn), decimal_places)
        phi = round(ef.phi(X, D, mu, container, metric=metric_fn), decimal_places)
        print(f"[ADJUST DISTANCE @ {get_datetime()}] Completed in {iteration_elapsed_time:.4f} seconds. Current Energy: {energy:.{decimal_places}f}/{phi:.{decimal_places}f}, D: {D}.")

        if no_change_counter >= 3:
            break

        iterations += 1

    if ef.E(X, D, container, metric=metric) >= tol:
        print(f"[ADJUST DISTANCE @ {get_datetime()}] Energy not sufficiently minimized, restarting adjustment.")
        
        # Decrease D slightly and retry
        D *= 0.95
        X = local_optimization(X, D, container, method=method, metric=metric_fn)
        
        X, D = adjust_distance(X, D, container, mu, iter_max, method, metric)

    elapsed_time = time.time() - start_time
    print(f"[ADJUST DISTANCE @ {get_datetime()}] Completed in {elapsed_time:.4f} seconds -- D: {D}, Quality: {ef.E(X, D, container, metric=metric):.8f}")
    return X, D