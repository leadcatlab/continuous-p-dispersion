import time
from typing import List, Union, Tuple, Dict, Optional

import numpy as np

from tsgo.container.point import Point
from tsgo.container.container import Container
from tsgo.container.geometry import distance, point_to_line_dist

ALPHA = 1.0
GAMMA = 2.0
LAMBDA = 0.5

TOL = 1e-4

# ============================================================
# Module-level caches
# ============================================================

# Oi0 cache keyed by quantized point, D, container id, metric
Oi0_cache: Dict[Tuple[Tuple[float, float, float], float, int, str], float] = {}
E_cache: Dict[Tuple[bytes, float, int, str], float] = {}

# Fast geometry caches (per container instance)
_VERTS_ARRAY_CACHE: Dict[int, np.ndarray] = {}  # id(container) -> (nV, 3)
_EDGES_BBOX_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, List]] = {}  # id(container) -> (emin, emax, edges_list)
_CONTAINS_CACHE: Dict[Tuple[int, Tuple[float, float, float]], bool] = {}  # (id(container), key) -> bool
_VERTS_CACHE: Dict[int, np.ndarray] = {}  # id(container) -> (nV,3)
_EDGES_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, List]] = {}  # id(container) -> (emin, emax, edges)

# Pair index cache for dispersion penalty
_PAIR_IDX_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # p -> (iu, ju)

# ============================================================
# Timing / profiling controls
# ============================================================
TIMES_CALLED = 50000

_TIME_STATS = {
    "dispersion_penalty": {"calls": 0, "time": 0.0},
    "container_penalty": {"calls": 0, "time": 0.0},
    "E": {"calls": 0, "time": 0.0},
}


def _record_time(name: str, dt: float) -> None:
    stat = _TIME_STATS[name]
    stat["calls"] += 1
    stat["time"] += dt

    if stat["calls"] % TIMES_CALLED == 0:
        avg = stat["time"] / stat["calls"]
        total = avg * stat["calls"]
        print(
            f"[TIMING] {name:18s} | "
            f"calls = {stat['calls']:6d} | "
            f"avg = {avg:.6f} s | "
            f"total = {total:.6f} s"
        )

# ============================================================
# Cache helpers
# ============================================================
def _contains_cached(container: Container, pt: Point, decimals: int = 6) -> bool:
    key = (round(pt.x, decimals), round(pt.y, decimals), round(pt.z, decimals))
    ck = (id(container), key)
    hit = _CONTAINS_CACHE.get(ck)
    if hit is not None:
        return hit
    val = container.contains(pt)
    _CONTAINS_CACHE[ck] = val
    return val

def _pair_indices(p: int) -> Tuple[np.ndarray, np.ndarray]:
    ij = _PAIR_IDX_CACHE.get(p)
    if ij is None:
        ij = np.triu_indices(p, k=1)
        _PAIR_IDX_CACHE[p] = ij
    return ij

def _get_verts(container: Container) -> np.ndarray:
    cid = id(container)
    V = _VERTS_CACHE.get(cid)
    if V is None:
        V = np.asarray([v.as_array() for v in container.vertices], dtype=float)
        _VERTS_CACHE[cid] = V
    return V

def _get_edges_bbox(container: Container) -> Tuple[np.ndarray, np.ndarray, List]:
    cid = id(container)
    cached = _EDGES_CACHE.get(cid)
    if cached is not None:
        return cached

    edges = list(container.edges)
    if not edges:
        emin = np.zeros((0, 3), dtype=float)
        emax = np.zeros((0, 3), dtype=float)
        _EDGES_CACHE[cid] = (emin, emax, edges)
        return emin, emax, edges

    A = np.asarray([e[0].as_array() for e in edges], dtype=float)
    B = np.asarray([e[1].as_array() for e in edges], dtype=float)
    emin = np.minimum(A, B)
    emax = np.maximum(A, B)
    _EDGES_CACHE[cid] = (emin, emax, edges)
    return emin, emax, edges

# ============================================================
# Geometry helpers
# ============================================================
def _point_to_line_dist_compat(x_pt: Point, edge, metric: str) -> Optional[float]:
    """Call point_to_line_dist but remain compatible if it expects a callable or a string."""
    try:
        return point_to_line_dist(x_pt, edge, metric=metric)
    except TypeError:
        # Fallback: pass a callable based on our distance()
        def _m(a, b):
            return distance(a, b, metric)
        return point_to_line_dist(x_pt, edge, metric=_m)


def _pairwise_dists(A: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    diff = A[:, None, :] - B[None, :, :]
    m = metric.upper()
    if m == "EUCLIDEAN":
        return np.sqrt(np.sum(diff * diff, axis=2))
    if m == "TAXICAB":
        return np.sum(np.abs(diff), axis=2)
    if m == "SUP_NORM":
        return np.max(np.abs(diff), axis=2)
    raise ValueError(f"Unknown metric '{metric}'")


def _dists_point_to_rows(x: np.ndarray, R: np.ndarray, metric: str) -> np.ndarray:
    diff = R - x[None, :]
    m = metric.upper()
    if m == "EUCLIDEAN":
        return np.sqrt(np.sum(diff * diff, axis=1))
    if m == "TAXICAB":
        return np.sum(np.abs(diff), axis=1)
    if m == "SUP_NORM":
        return np.max(np.abs(diff), axis=1)
    raise ValueError(f"Unknown metric '{metric}'")


# ============================================================
# Core penalty terms
# ============================================================
def l_ij(
    x_i: Union[Point, np.ndarray],
    x_j: Union[Point, np.ndarray],
    D: float,
    metric: str = "EUCLIDEAN",
) -> float:
    """
    Computes the dispersion penalty term for two dispersion points.

    Args:
        x_i (Union[Point, np.ndarray]): First dispersion point.
        x_j (Union[Point, np.ndarray]): Second dispersion point.
        D (float): Fixed radius of each sphere.
        metric (str, optional): Distance metric to use. Defaults to "EUCLIDEAN".

    Returns:
        float: Dispersion penalty term.
    """
    d = distance(x_i, x_j, metric)
    return max(0.0, 2 * D - d)


def Oi0(
    x_i: Union[Point, np.ndarray],
    D: float,
    container: Container,
    metric: str = "EUCLIDEAN",
) -> float:
    """Penalty term for point containment and proximity to container features.
    
    Computes a penalty value based on whether a point is contained within a container
    and its proximity to container features (vertices, footpoints, and edges).
    The result is cached to improve performance in repeated calls.

    Args:
        x_i (Union[Point, np.ndarray]): The point to evaluate.
        D (float): The distance threshold for penalty calculation.
        container (Container): The container object defining the spatial boundaries.
        metric (str, optional): The distance metric to use. Defaults to "EUCLIDEAN".
        
    Returns:
        float: The computed penalty value.

    Notes:
        Results are cached based on the point coordinates (rounded to 6 decimal places),
        distance threshold, container id, and metric type.
    """
    x_pt = Point.from_array(x_i) if isinstance(x_i, np.ndarray) else x_i
    key = (round(x_pt.x, 6), round(x_pt.y, 6), round(x_pt.z, 6)), D, id(container), metric
    cached = Oi0_cache.get(key)
    if cached is not None:
        return cached

    x = x_pt.as_array()
    footpoints = container.get_active_footpoints(x_pt)
    refs = container.vertices + footpoints

    if _contains_cached(container, x_pt):
        s = 0.0

        # refs penalties
        for v in refs:
            dv = distance(x, v.as_array(), metric)
            t = D - dv
            if t > 0.0:
                s += t * t

        # edge penalties (ALL edges; slow)
        for e in container.edges:
            dist = _point_to_line_dist_compat(x_pt, e, metric)
            if dist is not None:
                t = D - dist
                if t > 0.0:
                    s += t * t

        result = s
    else:
        dmin = float("inf")
        for v in refs:
            dv = distance(x, v.as_array(), metric)
            if dv < dmin:
                dmin = dv
        result = GAMMA * (D + dmin) ** 2

    Oi0_cache[key] = result
    return result


# ============================================================
# Vectorized penalties
# ============================================================
def dispersion_penalty(X: np.ndarray, D: float, metric: str = "EUCLIDEAN") -> float:
    """
    Computes the dispersion penalty for a given configuration of points.
    Args:
        X (np.ndarray): Configuration of dispersion points.
        D (float): Fixed radius of each sphere.
        metric (str, optional): Distance metric to use. Defaults to "EUCLIDEAN".
        
    Raises:
        ValueError: If X is flat but not a multiple of 3 in length, 
                    or if X is 2D but does not have 3 columns, 
                    or if X is not 1D or 2D.
                    
    Returns:
        float: Dispersion penalty of the configuration.
    """
    t0 = time.perf_counter()

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        if X.size % 3 != 0:
            raise ValueError(f"dispersion_penalty expected flat X of length multiple of 3, got {X.size}")
        X = X.reshape(-1, 3)
    elif X.ndim == 2:
        if X.shape[1] != 3:
            raise ValueError(f"dispersion_penalty expected X with 3 columns, got {X.shape}")
    else:
        raise ValueError(f"dispersion_penalty expected 1D or 2D X, got ndim={X.ndim}")

    p = X.shape[0]
    if p <= 1:
        val = 0.0
    else:
        iu, ju = _pair_indices(p)
        diffs = X[iu] - X[ju]  # (m, 3)

        m = metric.upper()
        if m == "EUCLIDEAN":
            d = np.sqrt(np.sum(diffs * diffs, axis=1))
        elif m == "TAXICAB":
            d = np.sum(np.abs(diffs), axis=1)
        elif m == "SUP_NORM":
            d = np.max(np.abs(diffs), axis=1)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Expected 'EUCLIDEAN', 'TAXICAB', or 'SUP_NORM'.")

        u = (2.0 * D) - d
        u[u < 0.0] = 0.0
        val = float(np.dot(u, u))  # sum(u^2)

    _record_time("dispersion_penalty", time.perf_counter() - t0)
    return val


def container_penalty(X: np.ndarray, D: float, container: Container, metric: str = "EUCLIDEAN") -> float:
    """
    Computes the container penalty for a given configuration of points within a container.

    Args:
        X (np.ndarray): Configuration of dispersion points.
        D (float): Fixed radius of each sphere.
        container (Container): Container object which is being used for p-dispersion.
        metric (str, optional): _description_. Defaults to "EUCLIDEAN".

    Raises:
        ValueError: If X is flat but not a multiple of 3 in length, 
                    or if X is 2D but does not have 3 columns, 
                    or if X is not 1D or 2D.

    Returns:
        float: Total energy of the configuration.
    """
    t0 = time.perf_counter()

    # --- FAST PATH: axis-aligned cube with walls at -0.5 and 0.5 ---
    if getattr(container, "description", "").lower() in {"cube", "unit cube"}:
        m = metric.upper()

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        d_faces = np.stack(
            [0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y, 0.5 - z, 0.5 + z],
            axis=1
        )  # (p,6)

        u = D - d_faces
        np.maximum(u, 0.0, out=u)
        total = float(np.sum(u * u))

        corners = np.array(
            [[sx * 0.5, sy * 0.5, sz * 0.5]
             for sx in (-1.0, 1.0)
             for sy in (-1.0, 1.0)
             for sz in (-1.0, 1.0)],
            dtype=np.float64
        )  # (8,3)

        diffs = X[:, None, :] - corners[None, :, :]  # (p,8,3)
        if m == "EUCLIDEAN":
            d_verts = np.sqrt(np.sum(diffs * diffs, axis=2))  # (p,8)
        elif m == "TAXICAB":
            d_verts = np.sum(np.abs(diffs), axis=2)
        elif m == "SUP_NORM":
            d_verts = np.max(np.abs(diffs), axis=2)
        else:
            raise ValueError(f"Unknown metric '{metric}'. Expected 'EUCLIDEAN', 'TAXICAB', or 'SUP_NORM'.")

        u = D - d_verts
        np.maximum(u, 0.0, out=u)
        total += float(np.sum(u * u))

        _record_time("container_penalty", time.perf_counter() - t0)
        return total

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 3)

    p = X.shape[0]
    V = _get_verts(container)
    dV = _pairwise_dists(X, V, metric) if V.size else np.zeros((p, 0), dtype=float)

    X_pts = [Point(float(X[i, 0]), float(X[i, 1]), float(X[i, 2])) for i in range(p)]
    inside = np.fromiter((container.contains(pt) for pt in X_pts), dtype=bool, count=p)

    emin, emax, edges = _get_edges_bbox(container)
    total = 0.0

    inside_idx = np.nonzero(inside)[0]
    if inside_idx.size:
        if dV.shape[1]:
            u = D - dV[inside_idx]
            np.maximum(u, 0.0, out=u)
            total += float(np.sum(u * u))

        for i in inside_idx:
            x = X[i]
            x_pt = X_pts[i]

            fps = container.get_active_footpoints(x_pt)
            if fps:
                F = np.asarray([fp.as_array() for fp in fps], dtype=float)
                u = D - _dists_point_to_rows(x, F, metric)
                u[u < 0.0] = 0.0
                total += float(np.dot(u, u))

            if edges and emin.shape[0]:
                mask = np.all((x[None, :] >= (emin - D)) & (x[None, :] <= (emax + D)), axis=1)
                idxs = np.nonzero(mask)[0]
                for ei in idxs:
                    dist = point_to_line_dist(x_pt, edges[ei], metric=metric)
                    t = D - dist
                    if t > 0.0:
                        total += t * t

    outside_idx = np.nonzero(~inside)[0]
    if outside_idx.size:
        if dV.shape[1]:
            dmin = np.min(dV[outside_idx], axis=1)
        else:
            dmin = np.full(outside_idx.shape[0], float("inf"), dtype=float)

        for k, i in enumerate(outside_idx):
            x = X[i]
            x_pt = X_pts[i]
            fps = container.get_active_footpoints(x_pt)
            if fps:
                F = np.asarray([fp.as_array() for fp in fps], dtype=float)
                df = float(np.min(_dists_point_to_rows(x, F, metric)))
                if df < dmin[k]:
                    dmin[k] = df

        total += float(np.sum(GAMMA * (D + dmin) ** 2))

    _record_time("container_penalty", time.perf_counter() - t0)
    return total


# ============================================================
# Total energy
# ============================================================
def E(X: np.ndarray, D: float, container: Container, metric: str = "EUCLIDEAN") -> float:
    """
    Computes the total energy of a solution (X, D)

    Args:
        X (np.ndarray): Configuration of dispersion points.
        D (float): Fixed radius of each sphere.
        container (Container): Container object which is being used for p-dispersion.
        metric (str, optional): Distance metric to use. Defaults to "EUCLIDEAN".

    Raises:
        ValueError: If X is flat but not a multiple of 3 in length.
        ValueError: If X is 2D but does not have 3 columns.
        ValueError: If X is not 1D or 2D.

    Returns:
        float: Total energy of the configuration.
    """
    t0 = time.perf_counter()

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        if X.size % 3 != 0:
            raise ValueError(f"E expected flat X of length multiple of 3, got {X.size}")
        X = X.reshape(-1, 3)
    elif X.ndim == 2:
        if X.shape[1] != 3:
            raise ValueError(f"E expected X with 3 columns, got shape {X.shape}")
    else:
        raise ValueError(f"E expected 1D or 2D X, got ndim={X.ndim}")

    disp_pen = dispersion_penalty(X, D, metric)
    cont_pen = container_penalty(X, D, container, metric)
    total = disp_pen + ALPHA * cont_pen

    _record_time("E", time.perf_counter() - t0)
    return float(total)

# ============================================================
# Pointwise / search utilities (used by tabu / vacancy search)
# ============================================================
def point_energy(
    X: Union[List[Point], np.ndarray],
    i: int,
    D: float,
    container: Container,
    metric: str = "EUCLIDEAN",
) -> float:
    """
    Compute the energy of a single dispersion point within a solution (X, D)

    Args:
        X (Union[List[Point], np.ndarray]): Configuration of dispersion points.
        i (int): Index of the point in X to compute energy for.
        D (float): Fixed radius of each sphere.
        container (Container): Container object which is being used for p-dispersion.
        metric (str, optional): Distance metric to use. Defaults to "EUCLIDEAN".

    Returns:
        float: Energy of the specified point within the configuration.
    """
    if isinstance(X, np.ndarray):
        x_i = X[i]
    else:
        x_i = X[i].as_array()

    disp = 0.0
    for j in range(len(X)):
        if j == i:
            continue
        x_j = X[j] if isinstance(X, np.ndarray) else X[j].as_array()
        disp += l_ij(x_i, x_j, D, metric) ** 2

    cont = Oi0(x_i, D, container, metric)
    return float(disp + ALPHA * cont)


def vacancy_energy(
    X: Union[List[Point], np.ndarray],
    vacancy_point: Union[Point, np.ndarray],
    D: float,
    container: Container,
    metric: str = "EUCLIDEAN",
) -> float:
    """
    Compute the energy of a vacancy site.

    Args:
        X (Union[List[Point], np.ndarray]): Existing dispersion points.
        vacancy_point (Union[Point, np.ndarray]): Candidate vacancy point.
        D (float): Radius of the sphere centered at the vacancy point.
        container (Container): The spatial container defining the bounds for sampling.
        metric (str, optional): Distance metric to use. Defaults to "EUCLIDEAN".

    Returns:
        float: Energy value of the vacancy site.
    """
    disp = 0.0
    for x in X:
        disp += l_ij(vacancy_point, x, D, metric) ** 2

    cont = Oi0(vacancy_point, D, container, metric)
    return float(disp + ALPHA * cont)


def select_high_energy_points(
    X: Union[List[Point], np.ndarray],
    D: float,
    container: Container,
    Q: int = 3,
    metric: str = "EUCLIDEAN",
) -> List[int]:
    """
    Select a set of indices corresponding to the points with the highest energy.

    Args:
        X (Union[List[Point], np.ndarray]): A list or array of points from which to select high energy points.
        D (float): A distance parameter used in the energy calculation.
        container (Container): An object that contains necessary data for energy computation.
        Q (int, optional): The number of high energy points to select (default is 3).
        metric (str, optional): The distance metric to use for energy calculation (default is "EUCLIDEAN").

    Returns:
        List[int]: A list of indices corresponding to the selected high energy points.
    """
    energies = [(i, point_energy(X, i, D, container, metric)) for i in range(len(X))]
    energies.sort(key=lambda t: t[1], reverse=True)
    return [idx for idx, _ in energies[:Q]]


def find_vacancy_sites(
    X: List[Point],
    D: float,
    container: Container,
    Q: int = 3,
    metric: str = "EUCLIDEAN",
) -> List[Point]:
    """
    Sample candidate vacancy sites and return the Q best sites by vacancy energy.
    
    This function generates random candidate points within the container and scores
    them based on their vacancy energy. The top Q candidates with the lowest vacancy
    energy values are returned, making them suitable locations for placing new points
    in a dispersion optimization problem.
    
    Args:
        X (List[Point]): A list of existing points in the problem space.
        D (float): The minimum required distance threshold between points.
        container (Container): The spatial container that defines the bounds for sampling.
        Q (int, optional): The number of best vacancy sites to return. Defaults to 3.
        metric (str, optional): The distance metric to use for calculations. 
            Defaults to "EUCLIDEAN".
    
    Returns:
        List[Point]: A list of up to Q points representing the best vacancy sites,
            sorted by vacancy energy (lowest to highest).
    
    Notes:
        - Generates 10 * len(X) random candidate points for evaluation.
        - Returns fewer than Q points if fewer candidates are available.
    """
    random_points = [container._sample_random_point() for _ in range(10 * len(X))]
    scored = [(pt, vacancy_energy(X, pt, D, container, metric)) for pt in random_points]
    scored.sort(key=lambda t: t[1])
    return [scored[i][0] for i in range(min(Q, len(scored)))]


def phi(
    X: Union[List[Point], np.ndarray],
    D: float,
    mu: float,
    container: Container,
    metric: str = "EUCLIDEAN",
) -> float:
    """
    Computes the Grand Potential Energy of the solution (X, D) with penalty factor mu.

    Args:
        X (Union[List[Point], np.ndarray]): Configuration of dispersion points.
        D (float): Fixed radius of each sphere.
        mu (float): Penalty factor.
        container (Container): Container object which is being used for p-dispersion.
        metric (str, optional): Distance metric to use. Defaults to "EUCLIDEAN".

    Returns:
        float: Grand Potential Energy of the solution.
    """
    
    return float((mu * E(X, D, container, metric)) - np.power(D, 2, dtype=float))