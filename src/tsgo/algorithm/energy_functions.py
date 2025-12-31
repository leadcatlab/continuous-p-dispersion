import time
import numpy as np
import heapq
from typing import List, Union

from tsgo.container.point import Point
from tsgo.container.container import Container
from tsgo.utils.metric_enums import MetricType, get_metric_function
from tsgo.container.geometry import point_to_line_dist

ALPHA = 1.0
GAMMA = 2.0
LAMBDA = 0.5

TOL = 1e-4

# keep a cache of results of l_ij, Oi0, and E
l_ij_cache = {}
Oi0_cache = {}
E_cache = {}

# NOTE: D will symbolize the radius of the sphere centered around each dispersion point

def l_ij(x_i: Union[Point, np.ndarray], x_j: Union[Point, np.ndarray], D: float, metric: callable = MetricType['EUCLIDEAN']) -> float:
    """
    Computes the penalty term for the pairwise distance between dispersion points x_i and x_j.

    Args:
        x_i (Union[Point, np.ndarray]): first dispersion point
        x_j (Union[Point, np.ndarray]): second dispersion point
        D (float): radius of the sphere
        metric (Union[MetricType, callable], optional): a callable that computes the distance between two points (default: Euclidean)

    Returns:
        The penalty term for the pairwise distance between dispersion points x_i and x_j.
    """
    x_i_point = Point.from_array(x_i) if isinstance(x_i, np.ndarray) else x_i
    x_j_point = Point.from_array(x_j) if isinstance(x_j, np.ndarray) else x_j
    if (x_i_point, x_j_point, D) in l_ij_cache:
        return l_ij_cache[(x_i_point, x_j_point, D)]

    x_i = x_i_point.as_array()
    x_j = x_j_point.as_array()
    result = max(0, (2 * D) - metric(x_i, x_j))
    l_ij_cache[(x_i_point, x_j_point, D)] = result
    
    return result

def Oi0(
    x_i: Union[Point, np.ndarray],
    D: float,
    container: Container,
    metric: callable = MetricType["EUCLIDEAN"],
) -> float:
    """
    LOGIC-EQUIVALENT to your original Oi0:
      - same cache key style (Point, D)
      - same refs = vertices + active_footpoints
      - same inside/outside branches
      - same edge penalties
    Only micro-optimizations: fewer allocations + fewer dict lookups.
    """
    x_pt = Point.from_array(x_i) if isinstance(x_i, np.ndarray) else x_i
    x = x_pt.as_array()

    key = (x_pt, D)
    cached = Oi0_cache.get(key)
    if cached is not None:
        return cached

    # Same logic: always include vertices + active footpoints
    footpoints = container.get_active_footpoints(x_pt)
    refs = container.vertices + footpoints

    if container.contains(x_pt):
        s = 0.0

        # Same math as sum([(max(0, D - metric(...)))**2 for ...]) but no list allocation
        for v in refs:
            dv = metric(x, v.as_array())
            t = D - dv
            if t > 0.0:
                s += t * t

        # Same edge loop; avoid max() call
        for e in container.edges:
            dist = point_to_line_dist(x_pt, e, metric=metric)
            if dist is not None:
                t = D - dist
                if t > 0.0:
                    s += t * t

        result = s
    else:
        # Same logic: GAMMA * (D + min(metric(...)))**2
        dmin = float("inf")
        for v in refs:
            dv = metric(x, v.as_array())
            if dv < dmin:
                dmin = dv
        result = GAMMA * (D + dmin) ** 2

    Oi0_cache[key] = result
    return result

def dispersion_penalty(X: np.ndarray, D: float, metric: callable = MetricType['EUCLIDEAN']) -> float:
    """
    This function computes the penalty of a configuration incurred solely by the distance between points.

    Args:
        X (np.ndarray): The configuration of dispersion points.
        D (float): The radius of the sphere.
        metric (callable, optional): A callable that computes the distance between two points. Defaults to MetricType['EUCLIDEAN'].

    Returns:
        float: The penalty of the configuration incurred solely by the distance between points.
    """
    if len(X) == 1: return 0
    return np.sum([l_ij(X[i], X[j], D, metric) ** 2 for i in range(len(X)) for j in range(i + 1, len(X))])

def container_penalty(X: np.ndarray, D: float, container: Container, metric: callable = MetricType['EUCLIDEAN']) -> float:
    """
    Computes the penalty term incurred by the distance between points and the boundary of the container in a configurations.

    Args:
        X (np.ndarray): The configuration of dispersion points.
        D (float): The radius of the sphere.
        container (Container): The container object.
        metric (callable, optional): A callable that computes the distance between two points. Defaults to MetricType['EUCLIDEAN'].

    Returns:
        float: The penalty term incurred by the distance between points and the boundary of the container.
    """
    return np.sum([Oi0(X[i], D, container, metric) for i in range(len(X))])

def E(X: np.ndarray, D: float, container: Container, metric: Union[str, callable] = 'EUCLIDEAN', verbose=False) -> float:
    """
    Calculates the total energy of the system.

    Args:
        X (np.ndarray): The set of dispersion points.
        D (float): The radius of the sphere.
        metric (str, optional): The name of the metric to use. Defaults to 'EUCLIDEAN'.
        container (Container): The container object.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.

    Returns:
        float: The total energy of the system.
    """

    start_time = time.time()
    if isinstance(metric, str):
        metric = get_metric_function(metric)
    X = X.reshape(-1, 3)

    disp_penalty = dispersion_penalty(X, D, metric)
    cont_penalty = container_penalty(X, D, container, metric)


    total_energy = disp_penalty + ALPHA * cont_penalty

    elapsed_time = time.time() - start_time
    
    if verbose:
        print("[E] Dispersion Penalty:", disp_penalty)
        print("[E] Container Penalty:", cont_penalty)
        print(f"[E] Execution time: {elapsed_time:.6f} seconds")

    return total_energy

def point_energy(X: Union[List[Point], np.ndarray], i: int, D: float, container: Container, metric: callable = MetricType['EUCLIDEAN']) -> float:
    """
    Calculates the energy of the point X[i] within the system.

    Args:
        X (Union[List[Point], np.ndarray]): The set of dispersion points.
        i (int): The index of the point to calculate the energy.
        D (float): The radius of the sphere.
        container (Container): The container object.
        metric (callable, optional): A callable that computes the distance between two points. Defaults to MetricType['EUCLIDEAN'].

    Returns:
        float: The energy of the point X[i] within the system.
    """
    dispersion_penalty = sum(l_ij(X[i], X[j], D, metric) ** 2 for j in range(len(X)) if j != i)
    container_penalty = Oi0(X[i], D, container, metric)
    return dispersion_penalty + ALPHA * container_penalty

def vacancy_energy(X: Union[List[Point], np.ndarray], vacancy_point: Union[Point, np.ndarray], D: float, container: Container, metric: callable = MetricType['EUCLIDEAN']) -> float:
    """
    Calculates the energy of the vacancy point within the system.

    Args:
        X (List[Point]): The set of dispersion points.
        vacancy_point (Point): The vacancy point.
        D (float): The radius of the sphere.
        container (Container): The container object.
        metric (callable, optional): A callable that computes the distance between two points. Defaults to MetricType['EUCLIDEAN'].

    Returns:
        float: The energy of the vacancy point within the system.
    """

    dispersion_penalty = sum(l_ij(vacancy_point, X[j], D, metric) ** 2 for j in range(len(X)))
    container_penalty = Oi0(vacancy_point, D, container, metric)
    return dispersion_penalty + ALPHA * container_penalty

def phi(X: Union[List[Point], np.ndarray], D: float, mu: float, container: Container, metric: callable = MetricType['EUCLIDEAN']) -> float:
    """
    Calculates the grand potential of the system.

    Args:
        X (Union[List[Point], np.ndarray]): The set of dispersion points.
        D (float): The radius of the sphere.
        mu (float): The chemical potential.
        container (Container): The container object.

    Returns:
        float: The grand potential of the system.
    """
    value = (mu * E(X, D, container, metric)) - np.power(D, 2, dtype=float)
    return value

def select_high_energy_points(X: List[Point], D: float, container: Container, Q: int, metric: callable = MetricType['EUCLIDEAN']) -> List[int]:
    """
    Determines the top Q points in a configuration of dispersion points with the higherst energies. Utilizes
    a heap structure in order to determine the top Q items.

    Args:
        X (List[Point]): A configuration of dispersion points.
        D (float): The radius of the sphere.
        container (Container): The container object.
        Q (int): The number of top energy points to select.
        metric (callable, optional): A callable that computes the distance between two points. Defaults to MetricType['EUCLIDEAN'].
    Returns:
        List[int]: _description_
    """
    energies = []
    for i in range(len(X)):
        heapq.heappush(energies, (-point_energy(X, i, D, container, metric), i))
    high_energy_points = [heapq.heappop(energies)[1] for _ in range(Q)]
    return high_energy_points

def find_vacancy_sites(X: List[Point], D: float, container: Container, Q: int, metric: callable = MetricType['EUCLIDEAN']) -> List[np.ndarray]:
    random_points = [container._sample_random_point() for _ in range(10 * len(X))]
    random_points = [(point, vacancy_energy(X, point, D, container, metric)) for point in random_points]

    # return the Q points with the lowest energy
    random_points.sort(key=lambda x: x[1])
    return [random_points[i][0] for i in range(Q)]