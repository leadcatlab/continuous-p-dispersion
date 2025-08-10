import numpy as np
from src.point import Point
from old_version.src.container import Container
from src.energy_functions import candidate_quality
from src.tsgo_helpers import local_optimization, select_high_energy_points, find_vacancy_sites, move_point, monotonic_basin_hopping, local_optimization_phi, maximize_distance, is_feasible, update_tabu_list, forbidden

# ALGORITHM 1: Main Framework of the TSGO Algorithm
# Output: The best configuration found (X*) and the minimum distance between points (D*)
def tsgo_algorithm(p: int, container: Container, iter_max: int, rho: float):
    volume = container.volume()
    D = ((3 * rho * volume) / (4 * np.pi * p)) ** (1 / 3)
    print(f"Initial Distance D: {D}")
    X = container.generate_initial_solution(p)
    print(f"Initial Solution X: \n{X}")
    X = tabu_search(X, D, container)
    X, D = adjust_distance(X, D, container)
    print(f"Adjusted Solution X: \n{X} \nAdjusted Distance D: {D}")

    X_opt, D_opt = X, D
    
    for i in range(iter_max):
        print(f"Iteration {i + 1}/{iter_max}, Current D: {D}, Current Quality: {candidate_quality(X, D, container)}")
        D = D_opt
        X = container.generate_initial_solution(p)
        X = tabu_search(X, D, container)
        
        if np.isclose(candidate_quality(X, D, container), 0.0):
            X, D = adjust_distance(X, D, container)
            if D > D_opt:
                D_opt = D
                X_opt = X
                
    # format X_opt to a list of Point objects with entries rounded to 5 decimal places
    X_opt = np.array(X_opt).reshape(-1, 3)
    X_opt = np.round(X_opt, 5)  # Round to 3 decimal
    X_opt = [Point(x[0], x[1], x[2]).as_array() for x in X_opt]
    
    # round D_opt to 10 decimal places
    D_opt = float(round(D_opt, 10))
    
    return X_opt, D_opt

# ALGORITHM 2: TS Method for the Continuous p-Dispersion Problems
# Output: The best solution found (Xb)
def tabu_search(X0, D, container, beta_max = 3, Q = 3):
    Q = min(Q, len(X0))  # Ensure Q does not exceed the number of points in X0
    
    X = local_optimization(X0, D, container)
    
    X_best = X.copy()
    no_improve = 0
    
    tabu_list = []
    
    iterations = 0
    
    I = None
    
    while (no_improve <= beta_max) and (candidate_quality(X_best, D, container) > 1e-25):
        iterations += 1
        high_energy_points = select_high_energy_points(X, D, container, Q)
        low_energy_points = find_vacancy_sites(X, D, container, Q)
        
        X_best_neighbor = X_best
        for i in range(Q):
            for j in range(Q):
                X_neighbor = move_point(X, high_energy_points[i], low_energy_points[j])
                X_neighbor = local_optimization(X_neighbor, D, container)

                if not forbidden(high_energy_points[i], tabu_list, iterations):
                    if candidate_quality(X_neighbor, D, container) < candidate_quality(X_best_neighbor, D, container):
                        X_best_neighbor = X_neighbor
                        I = high_energy_points[i]
        
        X = X_best_neighbor.copy()
        
        tabu_list = update_tabu_list(tabu_list, I, iterations)
        X = monotonic_basin_hopping(X, D, container)
        if candidate_quality(X, D, container) < candidate_quality(X_best, D, container):
            X_best = X.copy()
            no_improve = 0
        else:
            no_improve += 1
    
    return X_best


# ALGORITHM 3: Adjust Distance (Adjustment Method for the Minimum Distance D Between Points)
# Output: Feasible local optimum solution (X, D)

def adjust_distance(X0, D0, container, max_iter=15):
    X, D, µ = X0.copy(), D0, 10
    iterations = 0
    while iterations < max_iter or not is_feasible(X, D, container):
        X, D = local_optimization_phi(X, D, container, µ)
        µ *= 5
        iterations += 1
    return X, D