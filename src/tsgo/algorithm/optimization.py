from typing import Tuple

import numpy as np
from scipy.optimize import Bounds, basinhopping, minimize

from tsgo.algorithm import energy_functions as ef
from tsgo.container.container import Container
from tsgo.utils.metric_enums import MetricType

def local_optimization(
    X0: np.ndarray, 
    D0: float, 
    container: Container, 
    method: str = 'BFGS', 
    metric: callable = MetricType['EUCLIDEAN']
) -> np.ndarray:
    """
    Perform local optimization on the given 3D points.

    Args:
        X0 (np.ndarray): Initial 3D points.
        D0 (float): Initial distance.
        container (Container): The container object defining the problem space.

    Returns:
        List[Point]: Locally optimized 3D points.
    """
    function = lambda X: ef.E(X, D0, container, metric)
    X0 = X0.flatten()
    result = minimize(fun=function, x0=X0, method=method, options={'maxiter': 100, 'disp': False})
    optimized_points = result.x.reshape(-1, 3)

    return optimized_points

def local_optimization_phi(
    X0: np.ndarray, 
    D0: float, 
    container: Container, 
    mu: float, 
    method: str = 'BFGS', 
    metric: callable = MetricType['EUCLIDEAN']
) -> Tuple[np.ndarray, float]:
    """
    Perform local optimization using the phi function.

    Args:
        X0 (np.ndarray): Initial 3D points.
        D0 (float): Initial distance.
        container (Container): The container object defining the problem space.
        mu (float): Parameter for the phi function.

    Returns:
        Tuple[List[Point], float]: The optimized 3D points and the optimized distance.
    """
    def function(X: np.ndarray) -> float:
        D = X[-1]
        X = X[:-1].reshape(-1, 3)
        return ef.phi(X, D, mu, container, metric=metric)

    x0 = np.append(X0.flatten(), D0)

    # Set bounds: all coordinates unbounded, last variable (D) >= 0
    num_vars = x0.shape[0]
    lower_bounds = [-np.inf] * (num_vars - 1) + [0.0]
    upper_bounds = [np.inf] * num_vars
    bounds = Bounds(lower_bounds, upper_bounds)
    if method == 'BFGS':
        method = 'L-BFGS-B'  # Use L-BFGS-B when bounds are present

    result = minimize(method=method, fun=function, x0=x0, bounds=bounds)

    optimized_points = result.x[:-1].reshape(-1, 3)
    optimized_d = result.x[-1]

    return optimized_points, optimized_d

def monotonic_basin_hopping(X: np.ndarray, D: float, container: Container, iterations: int = 10, metric: callable = MetricType['EUCLIDEAN']) -> np.ndarray:
    """
    Function that performs monotonic basin hopping on the energy function E.

    Args:
        X (np.ndarray): A configuration of dispersion points.
        D (float): The minimum distance between points.
        container (Container): The container object defining the problem space.
        iterations (int, optional): Number of basin hopping iterations. Defaults to 10.
        metric (callable, optional): Distance metric to use. Defaults to MetricType['EUCLIDEAN'].
    Returns:
        np.ndarray: _description_
    """

    x0 = X.flatten()

    # Perform basin hopping to escape local minima
    result = basinhopping(func=lambda x: ef.E(x.reshape(-1, 3), D, container, metric=metric), x0=x0, niter=iterations)

    return result.x.reshape(-1, 3)
