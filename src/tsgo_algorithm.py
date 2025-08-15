import numpy as np
import src.energy_functions as ef
from src.container import Container
from src.utils import local_optimization, local_optimization_phi
import time
import src.utils as utils

# ALGORITHM 1: Main Framework of the TSGO Algorithm
# Output: The best configuration found (X_opt) and the maximal minimum distance between points
def tsgo_algorithm(p: int, container: Container, iter_max: int, rho: float):
    area = container.area()
    D = np.sqrt((rho * area) / (p * np.pi))
    print(f"initial distance estimate: {D}")
    X = container.generate_initial_solution(p) # ndarray of shape (p, 2)
    print(f"initial solution X: \n{X}")
    X = tabu_search(X, D, container, beta_max=1, Q=3)
    X, D = adjust_distance(X, D, container)
    print(f"Adjust Distance D: {D}\nAdjusted Solution X: \n{X}")
    
    X_opt, D_opt = X, D
    
    for i in range(iter_max):
        print(f"Iteration {i + 1}/{iter_max}, Current D: {D}, Current Quality: {ef.E(X, D, container)}")
        D = D_opt
        X = container.generate_initial_solution(p)
        X = tabu_search(X, D, container, beta_max=1, Q=3)

        if np.isclose(ef.E(X, D, container), 0.0):
            X, D = adjust_distance(X, D, container)
            if D > D_opt:
                D_opt, X_opt = D, X
                
    X_opt = np.array(X_opt).reshape(-1, 2)
    X_opt = np.round(X_opt, 5)  # Round to 5 decimal places
    
    D_opt = float(round(D_opt, 10))
    
    return X_opt, D_opt

# ALGORITHM 2: Tabu Search
# Output: Best solution found through rearrangement of high energy points
def tabu_search(X0: np.ndarray, D: float, container: Container, beta_max: int = 3, Q: int = 3):
    Q = min(Q, X0.shape[0])  # Ensure Q does not exceed the number of points in X0
    X = local_optimization(X0, D, container)
    X_best = X.copy()
    no_improve = 0
    tabu_list = []
    iterations = 0
    I = None
    prev_X = X_best.copy()

    while (no_improve <= beta_max) and (ef.E(X_best, D, container) > 1e-10):
        start_time = time.time()
        high_energy_points = ef.select_high_energy_points(X, D, container, Q)
        low_energy_points = ef.find_vacancy_sites(X, D, container, Q)

        X_best_neighbor = X_best.copy()
        for i in range(Q):
            for j in range(Q):
                X_neighbor = utils.move_point(X, high_energy_points[i], low_energy_points[j])
                X_neighbor = local_optimization(X_neighbor, D, container)
                
                if not utils.forbidden(high_energy_points[i], tabu_list, iterations):
                    if ef.E(X_neighbor, D, container) < ef.E(X_best_neighbor, D, container):
                        X_best_neighbor = X_neighbor.copy()
                        I = (high_energy_points[i], low_energy_points[j])

        X = X_best_neighbor.copy()
        tabu_list = utils.update_tabu_list(I, tabu_list, iterations)
        X = utils.monotonic_basin_hopping(X, D, container)
        curr_energy = ef.E(X, D, container)

        # Check if the point locations have changed
        if np.allclose(X_best, X, atol=1e-8):
            no_improve += 1
        else:
            X_best = X.copy()
            no_improve = 0

        iterations += 1
        elapsed_time = time.time() - start_time
        energy = round(ef.E(X_best, D, container), 10)
        print(f"Iteration {iterations}: Current X_best:\n{np.round(X_best, 5)}, Time = {elapsed_time:.4f}s, Energy = {energy}, no_improve: {no_improve}")
        prev_X = X_best.copy()
            
    return X_best

# ALGORITHM 3: Adjust Distance (Adjustment Method for the Minimum Distance)
# Output: Feasible local optimal solution (X, D)
def adjust_distance(X0: np.ndarray, D0: float, container: Container, iter_max: int = 5):
    X, D, mu = X0.copy(), D0, 10.0
    iterations = 0
    no_change = 0
    prev_D = D
    while iterations < iter_max or no_change < 3:
        X, D_new = local_optimization_phi(X, D, container, mu)
        mu *= 5
        iterations += 1
        print(iterations, ef.E(X, D_new, container), D_new)
        if np.isclose(D_new, prev_D, atol=1e-8):
            no_change += 1
        else:
            no_change = 0
        prev_D = D_new
        D = D_new
        if no_change >= 3:
            break
    return X, D
