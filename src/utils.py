from unittest import result
import numpy as np
import src.energy_functions as ef
from src.container import Container
from scipy.optimize import minimize, basinhopping, Bounds
import heapq
import random

# @param X0: Initial solution 2D array where each row is a tuple and there are p rows.
# @param D0: Initial distance.
# @param container: The container object defining the problem space.
# @param D_constant: If True, use a constant distance during optimization.
# @return: Optimized points as a numpy array
def local_optimization(X0: np.ndarray, D0: float, container: Container):
    function = lambda X: ef.E(X.reshape(-1, 2), D0, container)
    result = minimize(method='L-BFGS-B', fun=function, x0=X0.flatten())
    optimized_points = result.x.reshape(-1, 2)
    return optimized_points
    
def local_optimization_phi(X0: np.ndarray, D0: float, container: Container, mu: float):
    
    def function(X):
        D = X[-1]
        X = X[:-1].reshape(-1, 2)
        return ef.phi(X, D, mu, container)

    x0 = np.append(X0.flatten(), D0)
    # Set bounds: all coordinates unbounded, last variable (D) >= 0
    num_vars = x0.shape[0]
    lower_bounds = [-np.inf] * (num_vars - 1) + [0.0]
    upper_bounds = [np.inf] * num_vars
    bounds = Bounds(lower_bounds, upper_bounds)
    result = minimize(method='L-BFGS-B', fun=function, x0=x0, bounds=bounds)
    optimized_points = result.x[:-1].reshape(-1, 2)
    optimized_d = result.x[-1]

    return optimized_points, optimized_d

def move_point(X: np.ndarray, high_energy_point: np.ndarray, low_energy_point: np.ndarray):
    # Move the high energy point to the low energy point
    X_new = X.copy()
    for i in range(X.shape[0]):
        if np.array_equal(X[i], high_energy_point):
            X_new[i] = low_energy_point
            break
    return X_new

def forbidden(high_energy_point: np.ndarray, tabu_list: list, iterations: int):
    # Check if the high energy point is in the tabu list
    for entry in tabu_list:
        if entry[0] == high_energy_point and entry[1] > iterations:
            return True
    return False

def update_tabu_list(I: tuple, tabu_list: list, iterations: int, max_size: int = 100):
    # Add a new entry to the tabu list
    tabu_list.append((I, iterations + 10))
    # Remove the oldest entry if the list exceeds max_size
    if len(tabu_list) > max_size:
        tabu_list.pop(0)
    return tabu_list

def monotonic_basin_hopping(X: np.ndarray, D: float, container: Container):
    # Perform basin hopping to escape local minima
    result = basinhopping(func=lambda x: ef.E(x.reshape(-1, 2), D, container), x0=X.flatten(), niter=100)
    return result.x.reshape(-1, 2)