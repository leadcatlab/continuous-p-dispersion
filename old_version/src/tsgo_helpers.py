import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from old_container import Container
from src.energy_functions import candidate_quality, E, phi
from scipy.optimize import minimize, basinhopping
import heapq
import random

def local_optimization(X0, D0, container):
    """
    Perform local optimization on the initial solution X0 with distance D0.
    This function minimizes the energy function E with respect to the points in X0.
    
    Parameters:
    - X0: Initial solution 2D array where each row is a three-ple and there are p rows.
    - D0: Initial distance.
    - container: The container object defining the problem space.
    
    Returns:
    - Optimized points as a numpy array 
    """
    
    def energy_function(X):
        X = X.reshape(-1, 3)  # Reshape X to a 2D array of points
        return candidate_quality(X, D0, container)
    
    # Convert Point objects to numpy array for optimization
    X0_np = np.array(X0)
    
    # Optimize using scipy's minimize function
    result = minimize(method='L-BFGS-B', fun=energy_function, x0=X0_np.flatten(), options={'disp': True})

    # Convert back to Point objects
    optimized_points = result.x.reshape(-1, 3)
    
    return optimized_points

def select_high_energy_points(X, D, container, Q):
    """
    Select Q points from X that have the highest energy.
    
    Parameters:
    - X: Current solution as a numpy array of points.
    - D: Distance parameter.
    - container: The container object defining the problem space.
    - Q: Number of points to select.
    
    Returns:
    - List of indices of the selected high-energy points.
    """
    high_energy_points = []
    for i in range(len(X)):
        energy = candidate_quality(np.array([X[i]]), D, container)
        if len(high_energy_points) < Q:
            heapq.heappush(high_energy_points, (energy, i))
        else:
            heapq.heappushpop(high_energy_points, (energy, i))

    high_energy_points.sort(key=lambda x: x[0], reverse=True)
    high_energy_points = high_energy_points[:Q]
    assert len(high_energy_points) == Q, "The number of selected points should be equal to Q."
    return np.array([X[i] for _, i in high_energy_points])

def find_vacancy_sites(X, D, container, Q):
    """
    Find Q vacancy sites in the container where new points can be placed.
    
    Parameters:
    - X: Current solution as a numpy array of points.
    - D: Distance parameter.
    - container: The container object defining the problem space.
    - Q: Number of vacancy sites to find.
    
    Returns:
    - List of vacancy sites as Point objects.
    """
    p = len(X)
    
    # Generate 100 random points in the container and rank their energies and return the lowest Q points
    random_points = np.array([container.random_point(as_array=True) for _ in range(5 * p)])

    # create a heap with the lowest Q energies
    vacancy_sites = []
    for random_point in random_points:
        energy = candidate_quality(np.array([random_point]), D, container)
        if len(vacancy_sites) < Q:
            heapq.heappush(vacancy_sites, (energy, random_point))
        else:
            heapq.heappushpop(vacancy_sites, (energy, random_point))

    # Return the vacancy sites sorted by energy
    vacancy_sites.sort(key=lambda x: x[0])
    return np.array([site[1] for site in vacancy_sites[:Q]])


def move_point(X, old_point, new_point):
    """
    Move a point in the solution from old_point to new_point.
    
    Parameters:
    - X: Current solution as a numpy array of points.
    - old_point: The point to move (as a numpy array).
    - new_point: The new location for the point (as a numpy array).

    Returns:
    - Updated solution as a numpy array of points.
    """
    X_new = X.copy()
    for i in range(len(X_new)):
        if np.array_equal(X_new[i], old_point):
            X_new[i] = new_point
            break
    return X_new

def monotonic_basin_hopping(X, D, container):
    """
    Perform monotonic basin hopping to refine the solution X.
    
    Parameters:
    - X: Current solution as a numpy array of points.
    - D: Distance parameter.
    - container: The container object defining the problem space.
    
    Returns:
    - Refined solution as a numpy array of points.
    """
    # Perform local optimization on the current solution
    def func(X):
        X = X.reshape(-1, 3)  # Reshape X to a 2D array of points
        return candidate_quality(X, D, container)

    result = basinhopping(func=func, x0=X.flatten(), niter=10, T=1.0, stepsize=0.5)
    optimized_points = result.x.reshape(-1, 3)
    return optimized_points

def local_optimization_phi(X, D, container, mu):
    """
    Perform local optimization using the phi function.
    
    Parameters:
    - X: Current solution as a numpy array of points.
    - D: Distance parameter.
    - container: The container object defining the problem space.
    
    Returns:
    - Optimized points as a numpy array 
    """
    
    def energy_function(X):
        D = X[-1]  # Last element is the distance D
        X = X[:-1]  # All but the last element are the points
        X = X.reshape(-1, 3)  # Reshape X to a 2D array of points
        return phi(mu, X, D, container)
    
    # Convert Point objects to numpy array for optimization
    X_np = np.array(X).flatten()
    X_np = np.append(X_np, D)  # Append D to the end of the array
    
    # Optimize using scipy's minimize function
    result = minimize(method='L-BFGS-B', fun=energy_function, x0=X_np)
    # print(result)

    # Convert back to Point objects
    optimized_points = result.x[:-1].reshape(-1, 3)  # All but the last element are the points
    D_opt = result.x[-1]  # Last element is the distance D
    # print("Optimized points after local optimization:", optimized_points)
    # print("Optimized distance D after local optimization:", D_opt)
    
    # D_opt is the smallest distance between any two points in the optimized solution
    for i in range(len(optimized_points)):
        for j in range(i + 1, len(optimized_points)):
            dist = np.linalg.norm(optimized_points[i] - optimized_points[j])
            if dist < D_opt:
                D_opt = dist
    
    return optimized_points, D_opt

def maximize_distance(X, D, container):
    """
    Maximize the distance between points in the solution X.
    
    Parameters:
    - X: Current solution as a numpy array of points.
    - D: Distance parameter.
    - container: The container object defining the problem space.
    
    Returns:
    - Optimized points as a numpy array 
    """
    
    def energy_function(X):
        X = X.reshape(-1, 3)  # Reshape X to a 2D array of points
        return -E(X, D, container)  # Negate to maximize distance
    
    # Convert Point objects to numpy array for optimization
    X_np = np.array(X)
    
    # Optimize using scipy's minimize function
    result = minimize(method='L-BFGS-B', fun=energy_function, x0=X_np.flatten(), options={'disp': True})

    # Convert back to Point objects
    optimized_points = result.x.reshape(-1, 3)
    
    return optimized_points

def is_feasible(X, D, container):
    """
    Check if the solution X with distance D is feasible within the container.
    
    Parameters:
    - X: Current solution as a numpy array of points.
    - D: Distance parameter.
    - container: The container object defining the problem space.
    
    Returns:
    - True if the solution is feasible, False otherwise.
    """
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if np.linalg.norm(X[i] - X[j]) < 2 * D:
                return False
    return all(container.contains(Point(*x)) for x in X)

def update_tabu_list(tabu_list, point, iterations):
    if point is None:
        return tabu_list
    lifetime = 5 + random.randint(0, 5)  # Random lifetime between 5 and 10 iterations
    tabu_list.append((point, iterations + lifetime))
    
    # Remove points that are no longer tabu
    tabu_list[:] = [(p, t) for p, t in tabu_list if t > iterations]
    # print(f"Updated tabu list: {tabu_list}")
    return tabu_list
    
def forbidden(point, tabu_list, iterations):
    """
    Check if a point is forbidden based on the tabu list.
    
    Parameters:
    - point: The point to check.
    - tabu_list: The tabu list containing forbidden points.
    
    Returns:
    - True if the point is forbidden, False otherwise.
    """
    for p, t in tabu_list:
        if np.array_equal(point, p) and t > iterations:
            return True
    return False
