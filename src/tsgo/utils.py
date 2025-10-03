from datetime import datetime
import time
import pickle
import os
import shutil
from typing import List, Tuple, Union
import numpy as np
from scipy.optimize import minimize, basinhopping, Bounds

from tsgo.point import Point
from tsgo.container import Container
from tsgo.metric_enums import MetricType
import tsgo.energy_functions as ef

def get_datetime() -> str:
    """
    Gets the current date and time and formats as yyyy-mm-dd hh:mm:ss.sss

    Returns:
        str: Formatted date and time string.
    """
    datetime_now = datetime.now()
    datetime_now = datetime_now.strftime('%Y-%m-%d %H:%M:%S.') + datetime_now.strftime('%f')[:3]
    return datetime_now

def point_to_line_dist(
    point: Point, 
    line: Tuple[Point, Point], 
    metric: callable = MetricType['EUCLIDEAN']
) -> float:
    """
    Computes the distance from a point to a line.

    Args:
        point (Point): The point.
        line (Tuple[Point, Point]): The line defined by two endpoints.
        metric (callable, optional): The distance metric to use. Defaults to MetricType['EUCLIDEAN'].

    Returns:
        float: The distance from the point to the line.
    """
    def to_array(obj):
        if hasattr(obj, "as_array"):
            return obj.as_array()
        if isinstance(obj, np.ndarray):
            return obj
        # assume object has x, y, z attributes
        return np.array([obj.x, obj.y, obj.z])

    p = to_array(point)
    a = to_array(line[0])
    b = to_array(line[1])

    line_vec = b - a
    point_vec = p - a
    denom = np.dot(line_vec, line_vec)
    # If the line endpoints coincide, treat as distance to a point
    if denom == 0:
        return metric(p, a)

    # projection parameter t along the segment (0 -> a, 1 -> b)
    t = np.dot(point_vec, line_vec) / denom

    # If projection falls outside the segment, return distance to nearest endpoint using the metric
    if t < 0 or t > 1:
        d0 = metric(p, a)
        d1 = metric(p, b)
        return float(min(d0, d1))

    # Closest point on the segment
    closest = a + t * line_vec
    return float(metric(p, closest))


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

def move_point(X: List[Point], i: int, low_energy_point: Point) -> List[Point]:
    """
    Move the high energy point to the low energy point.
    
    Args:
        X (List[Point]): The array of points.
        high_energy_point (int): The index of the point in X considered to have high energy.
        low_energy_point (Point): The point to which the high energy point will be moved.

    Returns
    -------
    List[Point]
        A copy of X with the high energy point replaced by the low energy point.
    """
    
    X_copy = X.copy()
    X_copy[i] = low_energy_point
    return X_copy

def forbidden(high_energy_point: Point, tabu_list: list, iterations: int):
    # Check if the high energy point is in the tabu list
    # 0th index: movement (high energy point, low energy point)
    # 1st index: iteration when the move becomes valid again

    for entry in tabu_list:
        if entry is None:
            continue
        if entry[0][0] == high_energy_point and entry[1] > iterations:
            return True
    return False

def update_tabu_list(I: tuple, tabu_list: list, iterations: int, max_size: int = 100):
    if I is None or I[0] is None or I[1] is None:
        return tabu_list
    # Add a new entry to the tabu list
    tabu_list.append((I, iterations + 10))
    # Remove the oldest entry if the list exceeds max_size
    while len(tabu_list) > max_size:
        tabu_list.pop(0)
    return tabu_list

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

def save_to_file(p: int, iterations: int, D_init: float, X_opt: Union[np.ndarray, List[Point]], D_opt: float, container: Container, elapsed_time: float, metric_name: str = 'EUCLIDEAN') -> None:
    string = ""
    string += f"Number of Points: {p}\nIterations: {iterations}\nD_initial: {D_init}\n"
    string += "\nOptimal Arrangement of Points:\n"
    for point in X_opt:
        if isinstance(point, np.ndarray):
            string += f"{point[0]}, {point[1]}, {point[2]}\n"
        elif isinstance(point, Point):
            string += f"{point.x}, {point.y}, {point.z}\n"
        else:
            string += str(point) + "\n"
    
    string += f"\nOptimal Minimum Distance between Points: {D_opt}\n"
    string += f"Time taken: {elapsed_time:.2f} seconds\n"
    filename = f"TSGO_{p}_data.txt"
    with open(filename, "w") as f:
        f.write(string)

    # save p, iterations, D_init, X_opt, D_opt, container, elapsed_time, metric to a pickle file
    with open(filename.replace(".txt", ".pkl"), "wb") as f:
        pickle.dump({
            "optimized points": X_opt,
            "D": D_opt,
            "container": container,
            "elapsed_time": elapsed_time,
            "p": p,
            "iterations": iterations,
            "D_init": D_init,
            "metric": metric_name
        }, f)
        f.close()

    subfolder_name = os.path.join("output", metric_name.lower(), container.description.replace(" ", "_"), f"p={p}", f"iterations={iterations}", f"D_init={D_init}")
    os.makedirs(subfolder_name, exist_ok=True)
    
    dest_txt = os.path.join(subfolder_name, os.path.basename(filename))
    dest_pkl = os.path.join(subfolder_name, os.path.basename(filename.replace(".txt", ".pkl")))
    # Remove destination files if they exist
    if os.path.exists(dest_txt):
        os.remove(dest_txt)
    if os.path.exists(dest_pkl):
        os.remove(dest_pkl)

    shutil.move(filename, subfolder_name)
    shutil.move(filename.replace(".txt", ".pkl"), subfolder_name)