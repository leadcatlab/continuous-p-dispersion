import os
import time
from typing import List, Union, Tuple, Dict, Optional

import numpy as np

from tsgo.container.point import Point
from tsgo.container.container import Container, CACHE_LIMIT
from tsgo.container.geometry import distance, point_to_line_dist
from tsgo.container.norms import segment_distance, segment_distance_grad
from tsgo.container.face_batch import (face_arrays as _face_arrays,
                                       faces_contain as _faces_contain,
                                       active_feet as _active_feet,
                                       active_feet_many as _active_feet_many,
                                       EPS_PLANE as _EPS_PLANE)

ALPHA = 1.0
GAMMA = 2.0   # retained for reference; superseded by EXTERIOR_SLOPE below
LAMBDA = 0.5

# Exterior boundary penalty: (D + EXTERIOR_SLOPE * d)^2 for a point d outside
# the container. At d = 0 this equals D^2 for any value of the constant, so
# continuity and severity are independent here.
#
# Whether that makes E continuous across the wall depends on BOUNDARY_MODE, and
# only "min" gets it. Under "min" the interior branch tends to (D - 0)^2 = D^2
# at the wall and the two sides agree. Under "sum" the interior limit is the
# sum of (D - d_f)^2 over *every* feature within D, which exceeds D^2 whenever
# more than one is in range -- near an edge or a corner, always. Measured on the
# cube at D = 0.15, crossing the floor 0.04 from an edge: sum jumps 0.0467 ->
# 0.0225, while min reads 0.0225 on both sides.
# The slope at the wall is 2 * EXTERIOR_SLOPE * D; at 2.0 that reproduces the
# restoring force of the previous GAMMA * (D + d)^2 form (measured 0.2080 vs
# 0.2040 at D = 0.05) while removing its jump of D^2. The manuscript's
# D + 2d^2 is not used: its slope at the wall is zero, so escaped points feel
# no restoring force and the optimiser stalls at the constant floor.
EXTERIOR_SLOPE = 2.0

# How the boundary penalty aggregates the distances to nearby boundary features.
#
#   "sum"  the published model: sum of (D - d_f)^2 over every footpoint, edge and
#          vertex within D. A face's active footpoint and that face's bounding edge
#          coincide in distance when the footpoint sits on the edge, so this counts
#          one constraint twice and jumps discontinuously as the footpoint enters
#          the face -- inside the container.
#   "min"  O_i = max(0, D - min_f d_f)^2, the true distance to the boundary. Same
#          zero set (min >= D iff every feature is >= D), so E = 0 <=> feasible is
#          unchanged, but continuous and Lipschitz.
#
# Read from the environment at import so multiprocessing workers inherit it.
BOUNDARY_MODE = os.environ.get("CPDP_BOUNDARY_MODE", "sum").lower()


def set_boundary_mode(mode: str) -> None:
    """Set the boundary aggregation mode ('sum' or 'min') for the current process."""
    global BOUNDARY_MODE
    mode = mode.lower()
    if mode not in ("sum", "min"):
        raise ValueError(f"boundary mode must be 'sum' or 'min', got {mode!r}")
    BOUNDARY_MODE = mode

TOL = 1e-4

# ============================================================
# Module-level caches
# ============================================================

# Oi0 cache keyed by exact point, D, container id, metric
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


def clear_caches() -> None:
    """Drop every cache keyed by id(container).

    CPython reuses the id of a collected object, so a container built after
    another one has been freed can inherit its cached geometry and verdicts.
    Harmless while a process only ever builds one kind of container; call this
    before checking a result in a process that may have built others."""
    for cache in (Oi0_cache, E_cache, _VERTS_ARRAY_CACHE, _EDGES_BBOX_CACHE,
                  _CONTAINS_CACHE, _VERTS_CACHE, _EDGES_CACHE,
                  _NORMAL_NORM_CACHE, _EDGE_ARR_CACHE, _PAIR_LIST):
        cache.clear()
    from tsgo.container import face_batch
    face_batch._CACHE.clear()

# ============================================================
# Timing / profiling controls
# ============================================================
TIMES_CALLED = 1_000_000

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
def _contains_cached(container: Container, pt: Point) -> bool:
    # Exact coordinates: see Container.contains for what a rounded key cost.
    ck = (id(container), (pt.x, pt.y, pt.z))
    hit = _CONTAINS_CACHE.get(ck)
    if hit is not None:
        return hit
    val = container.contains(pt)
    if len(_CONTAINS_CACHE) >= CACHE_LIMIT:     # bounded; see Container.contains
        _CONTAINS_CACHE.clear()
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

_NORMAL_NORM_CACHE: Dict[Tuple[int, str], np.ndarray] = {}


def _face_normal_norms(container: Container, metric: str) -> np.ndarray:
    """Dual norm ||N_F||_* of each face normal. Geometry only, so cached.

    A face at signed distance h contributes |h| / ||N||_* to the boundary
    distance, not |h| * ||N||. The previous form multiplied by the metric norm
    of the normal, which happens to agree for axis-aligned faces but not in
    general, and was part of why slanted containers over-estimated clearance.
    """
    from tsgo.container.norms import dual_norm
    key = (id(container), metric.upper())
    hit = _NORMAL_NORM_CACHE.get(key)
    if hit is None:
        hit = np.asarray([dual_norm(np.asarray(f.normal, dtype=float), metric)
                          for f in container.faces], dtype=float)
        _NORMAL_NORM_CACHE[key] = hit
    return hit


_EDGE_ARR_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _get_edge_arrays(container: Container) -> Tuple[np.ndarray, np.ndarray]:
    """Edge endpoints as two (E, 3) arrays, so every edge of the container can
    be measured against a point in one vectorised call instead of a Python
    loop. This is where the general segment-distance formulation pays: per
    pair it is slower than a scalar routine, per batch far faster."""
    cid = id(container)
    cached = _EDGE_ARR_CACHE.get(cid)
    if cached is not None:
        return cached
    edges = list(container.edges)
    if not edges:
        out = (np.zeros((0, 3)), np.zeros((0, 3)))
    else:
        out = (np.asarray([e[0].as_array() for e in edges], dtype=float),
               np.asarray([e[1].as_array() for e in edges], dtype=float))
    _EDGE_ARR_CACHE[cid] = out
    return out


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
        Results are cached on the exact point coordinates, distance threshold,
        container id, and metric type. The key used to round to 6 decimals,
        which let a point just outside a wall reuse the zero penalty of a
        neighbour just inside it.
    """
    x_pt = Point.from_array(x_i) if isinstance(x_i, np.ndarray) else x_i
    key = (x_pt.x, x_pt.y, x_pt.z), D, id(container), metric
    cached = Oi0_cache.get(key)
    if cached is not None:
        return cached

    x = x_pt.as_array()
    footpoints = container.get_active_footpoints(x_pt, metric)
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
        result = (D + EXTERIOR_SLOPE * dmin) ** 2

    if len(Oi0_cache) >= CACHE_LIMIT:            # bounded; see Container.contains
        Oi0_cache.clear()
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
    """Container penalty of a configuration. Dispatches on BATCHED_ENERGY;
    both paths return the same value to the last bit."""
    if BATCHED_ENERGY:
        t0 = time.perf_counter()
        val = _container_penalty_batched(X, D, container, metric)
        _record_time("container_penalty", time.perf_counter() - t0)
        return val
    return _container_penalty_loop(X, D, container, metric)


def _container_penalty_batched(X: np.ndarray, D: float, container: Container,
                               metric: str = "EUCLIDEAN") -> float:
    """_container_penalty_loop with the geometry computed for all points at once.

    Containment, active feet and edge distances are batched; what stays per
    point is only the small dot products, kept as the loop forms them so the
    sum is bit-identical.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 3)
    p = X.shape[0]
    V = _get_verts(container)
    dV = _pairwise_dists(X, V, metric) if V.size else np.zeros((p, 0), dtype=float)
    inside = container.contains_many(X) if p else np.zeros(0, dtype=bool)

    fa = _face_arrays(container)
    F = fa["F"]
    if F and p:
        hs = np.einsum("fj,pfj->pf", fa["N"], X[:, None, :] - fa["P0"][None, :, :])
        act = np.where(inside[:, None], hs >= -_EPS_PLANE, hs < 0.0)
        feet, keep, _ = _active_feet_many(container, X, act, metric, hs=hs)
        # Distance to each foot as get_active_footpoints + _dists_point_to_rows form it.
        dF = _pairwise_rows(feet - X[:, None, :], metric)                 # (p, F)
    else:
        keep = np.zeros((p, 0), dtype=bool)
        dF = np.zeros((p, 0))

    emin, emax, edges = _get_edges_bbox(container)
    use_min = BOUNDARY_MODE == "min"
    total = 0.0

    inside_idx = np.nonzero(inside)[0]
    if inside_idx.size:
        if dV.shape[1] and not use_min:
            u = D - dV[inside_idx]
            np.maximum(u, 0.0, out=u)
            total += float(np.sum(u * u))

        edge_d = {}
        if edges and emin.shape[0]:
            Xi = X[inside_idx]
            near = np.all((Xi[:, None, :] >= (emin - D)[None, :, :]) &
                          (Xi[:, None, :] <= (emax + D)[None, :, :]), axis=2)
            ri, ei = np.nonzero(near)
            if ri.size:
                ES, EE = _get_edge_arrays(container)
                dists, _ = segment_distance(Xi[ri], ES[ei], EE[ei], metric)
                bounds = np.searchsorted(ri, np.arange(inside_idx.size + 1))
                for r in range(inside_idx.size):
                    if bounds[r + 1] > bounds[r]:
                        edge_d[r] = dists[bounds[r]:bounds[r + 1]]

        for r, i in enumerate(inside_idx):
            nearest = float(np.min(dV[i])) if (use_min and dV.shape[1]) else float("inf")
            kf = keep[i]
            if kf.any():
                dFi = dF[i, kf]
                if use_min:
                    nearest = min(nearest, float(np.min(dFi)))
                else:
                    u = D - dFi
                    u[u < 0.0] = 0.0
                    total += float(np.dot(u, u))
            de = edge_d.get(r)
            if de is not None:
                if use_min:
                    nearest = min(nearest, float(de.min()))
                else:
                    t = D - de
                    t = t[t > 0.0]
                    total += float(np.dot(t, t))
            if use_min:
                t = D - nearest
                if t > 0.0:
                    total += t * t

    outside_idx = np.nonzero(~inside)[0]
    if outside_idx.size:
        if dV.shape[1]:
            dmin = np.min(dV[outside_idx], axis=1)
        else:
            dmin = np.full(outside_idx.shape[0], float("inf"), dtype=float)
        if keep.shape[1]:
            df = np.where(keep[outside_idx], dF[outside_idx], np.inf).min(axis=1)
            dmin = np.where(df < dmin, df, dmin)
        total += float(np.sum((D + EXTERIOR_SLOPE * dmin) ** 2))

    return total


def _pairwise_rows(diff: np.ndarray, metric: str) -> np.ndarray:
    """Metric norm over the last axis, as _dists_point_to_rows computes it."""
    m = metric.upper()
    if m == "EUCLIDEAN":
        return np.sqrt(np.sum(diff * diff, axis=-1))
    if m == "TAXICAB":
        return np.sum(np.abs(diff), axis=-1)
    if m == "SUP_NORM":
        return np.max(np.abs(diff), axis=-1)
    raise ValueError(f"Unknown metric '{metric}'")


def _container_penalty_loop(X: np.ndarray, D: float, container: Container, metric: str = "EUCLIDEAN") -> float:
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

    use_min = BOUNDARY_MODE == "min"

    inside_idx = np.nonzero(inside)[0]
    if inside_idx.size:
        if dV.shape[1] and not use_min:
            u = D - dV[inside_idx]
            np.maximum(u, 0.0, out=u)
            total += float(np.sum(u * u))

        for i in inside_idx:
            x = X[i]
            x_pt = X_pts[i]

            # nearest boundary feature seen so far, for the "min" aggregation
            nearest = float(np.min(dV[i])) if (use_min and dV.shape[1]) else float("inf")

            fps = container.get_active_footpoints(x_pt, metric)
            if fps:
                F = np.asarray([fp.as_array() for fp in fps], dtype=float)
                dF = _dists_point_to_rows(x, F, metric)
                if use_min:
                    nearest = min(nearest, float(np.min(dF)))
                else:
                    u = D - dF
                    u[u < 0.0] = 0.0
                    total += float(np.dot(u, u))

            if edges and emin.shape[0]:
                # Any edge failing this box test is further than D away in some
                # coordinate, so it can neither contribute nor be the binding minimum.
                mask = np.all((x[None, :] >= (emin - D)) & (x[None, :] <= (emax + D)), axis=1)
                idxs = np.nonzero(mask)[0]
                if idxs.size:
                    ES, EE = _get_edge_arrays(container)
                    dists, _ = segment_distance(np.broadcast_to(x, (idxs.size, 3)),
                                                ES[idxs], EE[idxs], metric)
                    if use_min:
                        nearest = min(nearest, float(dists.min()))
                    else:
                        t = D - dists
                        t = t[t > 0.0]
                        total += float(np.dot(t, t))

            if use_min:
                t = D - nearest
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
            fps = container.get_active_footpoints(x_pt, metric)
            if fps:
                F = np.asarray([fp.as_array() for fp in fps], dtype=float)
                df = float(np.min(_dists_point_to_rows(x, F, metric)))
                if df < dmin[k]:
                    dmin[k] = df

        total += float(np.sum((D + EXTERIOR_SLOPE * dmin) ** 2))

    _record_time("container_penalty", time.perf_counter() - t0)
    return total


# ============================================================
# Exact feasibility bound
# ============================================================
def max_feasible_D(X: np.ndarray, container: Container, metric: str = "EUCLIDEAN") -> float:
    """Largest radius D for which the configuration X is feasible.

    Both constraints tighten monotonically in D -- dispersion needs
    d(c_i, c_j) >= 2D and containment needs d(c_i, dC) >= D -- so for a fixed
    X the feasible set of radii is the interval [0, B] with

        B = min( min_{i<j} d(c_i,c_j)/2 ,  min_i d(c_i, dC) ).

    This is exact and costs one pass over the geometry, so it replaces walking
    D downwards in fixed percentage steps and hoping to land in the feasible
    region. It is independent of BOUNDARY_MODE: the penalty aggregates the
    boundary features differently in "sum" and "min", but E = 0 requires every
    feature to be at least D away either way.

    Returns 0.0 if any point lies outside the container, where no positive
    radius is feasible.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 3)
    p = X.shape[0]
    if p == 0:
        return 0.0

    X_pts = [Point(float(X[i, 0]), float(X[i, 1]), float(X[i, 2])) for i in range(p)]
    if not all(container.contains(pt) for pt in X_pts):
        return 0.0

    bound = float("inf")

    if p > 1:
        iu, ju = _pair_indices(p)
        d_pairs = _dists_point_to_rows_pairwise(X[iu], X[ju], metric)
        bound = min(bound, 0.5 * float(np.min(d_pairs)))

    V = _get_verts(container)
    if V.size:
        bound = min(bound, float(np.min(_pairwise_dists(X, V, metric))))

    ES, EE = _get_edge_arrays(container)
    for i in range(p):
        fps = container.get_active_footpoints(X_pts[i], metric)
        if fps:
            F = np.asarray([fp.as_array() for fp in fps], dtype=float)
            bound = min(bound, float(np.min(_dists_point_to_rows(X[i], F, metric))))
        if ES.shape[0]:
            d_edges, _ = segment_distance(np.broadcast_to(X[i], (ES.shape[0], 3)),
                                          ES, EE, metric)
            bound = min(bound, float(d_edges.min()))

    return max(0.0, float(bound))


def signed_clearance_and_grad(x: np.ndarray, container: Container,
                              metric: str = "EUCLIDEAN") -> Tuple[float, np.ndarray]:
    """Signed distance from one point to the container's boundary, with its
    (sub)gradient: positive inside, negative outside.

    For the constrained-program baselines on non-convex containers, which need
    containment as a constraint s(c) >= D. max_feasible_D would serve inside,
    but it returns 0 everywhere outside, which gives a solver no direction back
    once a point has left. The distance is the same minimum over vertices,
    active footpoints and edges that max_feasible_D takes; outside, the active
    faces are those the point lies beyond, as in get_active_footpoints. The
    gradient is that of the nearest feature (Danskin), so it is exact wherever
    the nearest feature is unique.
    """
    from tsgo.container.norms import segment_distance_grad
    x = np.asarray(x, dtype=float).reshape(3)
    inside = container.contains(Point(float(x[0]), float(x[1]), float(x[2])))

    best_d, best_g = float("inf"), np.zeros(3)
    V = _get_verts(container)
    if V.size:
        dv = _dists_point_to_rows(x, V, metric)
        k = int(np.argmin(dv))
        best_d, best_g = float(dv[k]), _metric_subgrad(x - V[k], metric)[0]

    fa = _face_arrays(container)
    hs = np.einsum("fj,fj->f", fa["N"], x[None, :] - fa["P0"])
    act = (hs >= -_EPS_PLANE) if inside else (hs < 0.0)
    if act.any():
        _, keep, _ = _active_feet(container, x, act, metric)
        idx = np.nonzero(keep)[0]
        if idx.size:
            mn = _face_normal_norms(container, metric)[idx]
            dd = np.abs(hs[idx]) / mn
            k = int(np.argmin(dd))
            if dd[k] < best_d:
                best_d = float(dd[k])
                best_g = (np.sign(hs[idx[k]]) / mn[k]) * fa["N"][idx[k]]

    ES, EE = _get_edge_arrays(container)
    if ES.shape[0]:
        d_e, _, g_e = segment_distance_grad(np.broadcast_to(x, (ES.shape[0], 3)), ES, EE, metric)
        k = int(np.argmin(d_e))
        if d_e[k] < best_d:
            best_d, best_g = float(d_e[k]), g_e[k]

    # best_g is the gradient of the distance itself, which grows away from the
    # boundary on either side; outside, s = -distance.
    return (best_d, best_g) if inside else (-best_d, -best_g)


_INRADIUS_CACHE: Dict[Tuple[str, str], float] = {}


def inradius(container: Container, metric: str = "EUCLIDEAN",
             samples: int = 64, seed: int = 0) -> float:
    """Radius of the largest ball that fits inside the container.

    This is exactly the p = 1 optimum -- one ball, no dispersion constraint --
    and since D_opt(p) is non-increasing in p it upper-bounds the optimum at
    every p. That makes it the right cap on an initial radius: a volume-based
    heuristic knows only how much space the container holds, not how it is
    shaped, and over-estimates badly on non-convex containers (1.5x on the
    H-box, 2.1x on the cage).

    Computed by sampling interior points for the one with the most clearance,
    then maximising clearance from there. The clearance function is concave on
    a convex container but only piecewise-smooth in general, so the refinement
    is derivative-free. Cached per container: the result depends on geometry
    alone.
    """
    key = (str(getattr(container, "description", container)), metric.upper())
    if key in _INRADIUS_CACHE:
        return _INRADIUS_CACHE[key]

    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    best_r, best_x = 0.0, None
    for _ in range(samples):
        x = np.asarray(container.generate_initial_solution(1, rng), dtype=float).reshape(1, 3)
        r = max_feasible_D(x, container, metric)
        if r > best_r:
            best_r, best_x = r, x.reshape(3)

    if best_x is not None:
        res = minimize(lambda v: -max_feasible_D(v.reshape(1, 3), container, metric),
                       best_x, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-10, "fatol": 1e-14})
        best_r = max(best_r, -float(res.fun))

    _INRADIUS_CACHE[key] = best_r
    return best_r


def _ball_volume_coeff(metric: str) -> float:
    """k in vol(ball of radius D) = k * D^3, for the metric's own ball.

    The heuristic places p balls filling `density` of the container, so it has
    to know how much volume a ball of radius D occupies -- and that is a
    property of the metric, not a constant. Using the Euclidean 4*pi/3 for every
    metric made the starting radius 1.24x too large under the sup norm, whose
    ball is a cube of volume (2D)^3, and 0.68x too small under l_1, whose ball
    is a cross-polytope of volume (2D)^3/6.

    Neither error is harmless. Too large: an initial radius above what any
    configuration supports sends the run down the recovery path of
    adjust_distance, which snaps D to the bound of whatever poor configuration
    it is holding and never climbs back. Too small: the search climbs, but can
    settle in an arrangement of too many small balls -- on the unit cube at
    p = 20 under the sup norm every start below the estimate ends at 1/8, every
    start at 1.3x the estimate at the optimum 1/6. The ladder in
    experiments/batch.py spreads the start over both sides. On the unit cube at p = 2
    under the sup norm this cost 8.7% of the radius, and it is the source of
    every substantive monotonicity violation in the sweep, all of which are sup
    norm and none of which are l_1 -- exactly the asymmetry these constants
    predict.
    """
    m = str(metric).upper()
    if m in ("SUP_NORM", "LINF"):
        return 8.0                      # cube of side 2D
    if m in ("TAXICAB", "L1"):
        return 8.0 / 6.0                # cross-polytope, (2D)^3 / 3!
    return 4.0 * np.pi / 3.0            # l_2, and the default


def initial_radius(container: Container, p: int, packing_density: float,
                   metric: str = "EUCLIDEAN") -> float:
    """Starting radius for a run. The single definition -- Config and the batch
    runner both call this, so the two cannot drift apart.

    p balls of radius D fill volume * density, so
    p * (4/3) * pi * D^3 = volume * density, capped at the inradius because no
    D above it is feasible at any p.
    """
    volume = float(container.volume)
    D = ((volume * packing_density) / (p * _ball_volume_coeff(metric))) ** (1.0 / 3.0)
    r_in = inradius(container, metric)
    return float(min(D, r_in)) if r_in > 0.0 else float(D)


def _dists_point_to_rows_pairwise(A: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    """Row-wise metric distance between two (n, 3) arrays."""
    diffs = np.asarray(A, dtype=float) - np.asarray(B, dtype=float)
    m = metric.upper()
    if m == "EUCLIDEAN":
        return np.sqrt(np.sum(diffs * diffs, axis=1))
    if m in ("TAXICAB", "L1"):
        return np.sum(np.abs(diffs), axis=1)
    if m in ("SUP_NORM", "LINF"):
        return np.max(np.abs(diffs), axis=1)
    raise ValueError(f"Unknown metric '{metric}'.")


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
    rng: Optional[np.random.Generator] = None,
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
    random_points = [container._sample_random_point(rng) for _ in range(10 * len(X))]
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

# ============================================================
# Analytic energy gradient
# ============================================================
# E is piecewise smooth: every term is a hinge (max(0, .))^2 composed with a
# metric distance to a fixed geometric feature -- a vertex, a segment, or a
# face plane. Each piece has a closed-form (sub)gradient, so the whole
# gradient is computable in one pass instead of scipy's 3p+1 finite-difference
# evaluations per L-BFGS iteration. Fully general: iterates the container's
# faces, edges and vertices; no container-specific logic.
#
# Feature gradients with respect to the query point x (Danskin's theorem for
# the segment minimum; s = signed plane offset (x - P0).n with unit Euclidean
# normal n and m(.) the metric norm):
#   vertex v          d = m(x - v)          grad d = g_m(x - v)
#   segment, argmin c d = m(x - c)          grad d = g_m(x - c)
#   face plane        d = |s| * m(n)        grad d = sign(s) * m(n) * n
# where g_m is the metric's (sub)gradient: x/||x|| for Euclidean, sign(x) for
# taxicab, and sign(x_k) e_k on the largest component for the sup norm.

from tsgo.container.geometry import point_to_line_closest


def _metric_subgrad(diff: np.ndarray, metric: str) -> np.ndarray:
    """(Sub)gradient of the metric norm at each row of diff, shape preserved."""
    m = metric.upper()
    diff = np.atleast_2d(diff)
    if m == "EUCLIDEAN":
        d = np.sqrt(np.sum(diff * diff, axis=1))
        out = np.zeros_like(diff)
        nz = d > 0.0
        out[nz] = diff[nz] / d[nz, None]
        return out
    if m == "TAXICAB":
        return np.sign(diff)
    if m == "SUP_NORM":
        out = np.zeros_like(diff)
        k = np.argmax(np.abs(diff), axis=1)
        rows = np.arange(diff.shape[0])
        out[rows, k] = np.sign(diff[rows, k])
        return out
    raise ValueError(f"Unknown metric '{metric}'")


def _metric_norm_of(v: np.ndarray, metric: str) -> float:
    m = metric.upper()
    if m == "EUCLIDEAN":
        return float(np.linalg.norm(v))
    if m == "TAXICAB":
        return float(np.sum(np.abs(v)))
    if m == "SUP_NORM":
        return float(np.max(np.abs(v)))
    raise ValueError(f"Unknown metric '{metric}'")



def _edge_dist_grad(x: np.ndarray, edge, metric: str) -> Tuple[float, np.ndarray]:
    """Distance from x to a segment with the gradient of that distance.

    Both the distance and its gradient come from the single convex
    formulation in tsgo.container.norms: the previous version derived the
    sup-norm tie case by hand and fell back to holding the closest point
    fixed everywhere else, which is wrong whenever the minimiser moves with x
    -- it disagreed with finite differences on 58% of taxicab samples.
    """
    a = np.array((edge[0].x, edge[0].y, edge[0].z), dtype=float) if hasattr(edge[0], "x") else np.asarray(edge[0], float)
    b = np.array((edge[1].x, edge[1].y, edge[1].z), dtype=float) if hasattr(edge[1], "x") else np.asarray(edge[1], float)
    d, _, g = segment_distance_grad(np.asarray(x, float), a, b, metric)
    return float(d[0]), g[0]


# Evaluate the container term for all points at once (_E_and_grad_batched)
# rather than point by point (_E_and_grad_loop). Same energy and gradient to
# the last bit; the per-point form spends most of its time in fixed numpy call
# overhead. On by default; CPDP_BATCHED_ENERGY=0 restores the loop. Read at
# import so multiprocessing workers inherit it.
BATCHED_ENERGY = os.environ.get("CPDP_BATCHED_ENERGY", "1") != "0"


def E_and_grad(
    X: np.ndarray,
    D: float,
    container: Container,
    metric: str = "EUCLIDEAN",
) -> Tuple[float, np.ndarray, float]:
    """Energy with its analytic gradient.

    Returns (E, dE/dX flattened to (3p,), dE/dD). Honors BOUNDARY_MODE.
    Dispatches on BATCHED_ENERGY; both paths compute the same function.
    """
    if BATCHED_ENERGY:
        return _E_and_grad_batched(X, D, container, metric)
    return _E_and_grad_loop(X, D, container, metric)


def _E_and_grad_loop(
    X: np.ndarray,
    D: float,
    container: Container,
    metric: str = "EUCLIDEAN",
) -> Tuple[float, np.ndarray, float]:
    """Energy with its analytic gradient, one point at a time.

    Returns (E, dE/dX flattened to (3p,), dE/dD). Honors BOUNDARY_MODE.
    The value reproduces dispersion_penalty + ALPHA * container_penalty on the
    general path (no cube shortcut; the geometry is recomputed exactly, not
    read from the rounded caches).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 3)
    p = X.shape[0]
    grad = np.zeros_like(X)
    dE_dD = 0.0
    total = 0.0

    # ---- dispersion term ----
    if p > 1:
        iu, ju = _pair_indices(p)
        diffs = X[iu] - X[ju]
        m = metric.upper()
        if m == "EUCLIDEAN":
            d = np.sqrt(np.sum(diffs * diffs, axis=1))
        elif m == "TAXICAB":
            d = np.sum(np.abs(diffs), axis=1)
        elif m == "SUP_NORM":
            d = np.max(np.abs(diffs), axis=1)
        else:
            raise ValueError(f"Unknown metric '{metric}'")
        u = (2.0 * D) - d
        u[u < 0.0] = 0.0
        total += float(np.dot(u, u))
        act = u > 0.0
        if np.any(act):
            g = _metric_subgrad(diffs[act], metric)
            coeff = (-2.0 * u[act])[:, None] * g
            np.add.at(grad, iu[act], coeff)
            np.add.at(grad, ju[act], -coeff)
            dE_dD += float(np.sum(4.0 * u[act]))

    # ---- container term ----
    use_min = BOUNDARY_MODE == "min"
    V = _get_verts(container)
    emin, emax, edges = _get_edges_bbox(container)

    for i in range(p):
        x = X[i]
        x_pt = Point(float(x[0]), float(x[1]), float(x[2]))
        inside = container.contains(x_pt)

        # Active footpoints, with the face each one belongs to. Batched over
        # every face: this loop was the largest remaining cost in the solver,
        # 140k Face.contains calls in a six-second profile. Inclusive to
        # EPS_PLANE on the inside, matching Container._is_active_plane -- a
        # point on, or a few 1e-9 outside, a face must keep that face active,
        # or its own wall drops out and the penalty reads zero.
        fa = _face_arrays(container)
        hs_all = np.einsum("fj,fj->f", fa["N"], x[None, :] - fa["P0"])
        act = (hs_all >= -_EPS_PLANE) if inside else (hs_all < 0.0)
        fp_feats = []  # (dist, grad_d) per active footpoint
        if act.any():
            # The same routine Container.get_active_footpoints calls. It used to
            # be a separate expression here that tested the *Euclidean* foot for
            # face membership while scoring it at the *metric* distance, so this
            # path and E disagreed by up to 2e-2 under the sup norm -- on the
            # gradient side, which is the one the optimiser follows.
            _, keep, _ = _active_feet(container, x, act, metric)
            idx = np.nonzero(keep)[0]
            if idx.size:
                mn = _face_normal_norms(container, metric)[idx]   # ||N||_*
                dd = np.abs(hs_all[idx]) / mn
                gg = (np.sign(hs_all[idx]) / mn)[:, None] * fa["N"][idx]
                fp_feats = list(zip(dd.tolist(), gg))

        if inside:
            best = (float("inf"), None)  # for "min" aggregation

            def _hinge(dist, gd):
                nonlocal total, dE_dD
                t = D - dist
                if t > 0.0:
                    total += ALPHA * t * t
                    grad[i] -= ALPHA * 2.0 * t * gd
                    dE_dD += ALPHA * 2.0 * t

            if V.size:
                dv = _dists_point_to_rows(x, V, metric)
                if use_min:
                    k = int(np.argmin(dv))
                    if dv[k] < best[0]:
                        best = (float(dv[k]), _metric_subgrad(x - V[k], metric)[0])
                else:
                    for k in np.nonzero(dv < D)[0]:
                        _hinge(float(dv[k]), _metric_subgrad(x - V[k], metric)[0])

            for dist, gd in fp_feats:
                if use_min:
                    if dist < best[0]:
                        best = (dist, gd)
                else:
                    _hinge(dist, gd)

            if edges and emin.shape[0]:
                mask = np.all((x[None, :] >= (emin - D)) & (x[None, :] <= (emax + D)), axis=1)
                idxs = np.nonzero(mask)[0]
                if idxs.size:
                    ES, EE = _get_edge_arrays(container)
                    dists, _, gds = segment_distance_grad(
                        np.broadcast_to(x, (idxs.size, 3)), ES[idxs], EE[idxs], metric)
                    if use_min:
                        k = int(np.argmin(dists))
                        if dists[k] < best[0]:
                            best = (float(dists[k]), gds[k])
                    else:
                        for k in np.nonzero(dists < D)[0]:
                            _hinge(float(dists[k]), gds[k])

            if use_min and best[1] is not None:
                _hinge(best[0], best[1])
        else:
            # exterior: (D + EXTERIOR_SLOPE * dmin)^2 over vertices and
            # footpoints, matching container_penalty's exterior branch
            dmin, gmin = float("inf"), np.zeros(3)
            if V.size:
                dv = _dists_point_to_rows(x, V, metric)
                k = int(np.argmin(dv))
                dmin, gmin = float(dv[k]), _metric_subgrad(x - V[k], metric)[0]
            for dist, gd in fp_feats:
                if dist < dmin:
                    dmin, gmin = dist, gd
            if np.isfinite(dmin):
                base = D + EXTERIOR_SLOPE * dmin
                total += ALPHA * base * base
                grad[i] += ALPHA * 2.0 * base * EXTERIOR_SLOPE * gmin
                dE_dD += ALPHA * 2.0 * base

    return float(total), grad.ravel(), float(dE_dD)


# Verlet pair list for the dispersion term, per (p, metric): the pairs within
# PAIR_REACH * 2D of each other at the reference configuration X_ref.
PAIR_SKIN = 0.5
_PAIR_LIST: Dict[Tuple[int, str], Tuple[np.ndarray, float, np.ndarray]] = {}


def _metric_rows(diffs: np.ndarray, metric: str) -> np.ndarray:
    m = metric.upper()
    if m == "EUCLIDEAN":
        return np.sqrt(np.sum(diffs * diffs, axis=1))
    if m == "TAXICAB":
        return np.sum(np.abs(diffs), axis=1)
    if m == "SUP_NORM":
        return np.max(np.abs(diffs), axis=1)
    raise ValueError(f"Unknown metric '{metric}'")


def _candidate_pairs(X: np.ndarray, D: float, metric: str) -> np.ndarray:
    """Indices into the full pair list (_pair_indices order) of every pair
    that can be within 2D -- a superset, ascending.

    Rebuilt from all pairs when the points have moved too far: a pair left
    out was at least `reach` apart at X_ref, and by the triangle inequality is
    still at least reach - 2 * max displacement apart now, so while that
    exceeds 2D it cannot contribute.
    """
    p = X.shape[0]
    key = (p, metric.upper())
    hit = _PAIR_LIST.get(key)
    if hit is not None:
        X_ref, reach, sel = hit
        moved = float(np.max(_metric_rows(X - X_ref, metric)))
        if 2.0 * D + 2.0 * moved < reach * (1.0 - 1e-12):
            return sel
    iu, ju = _pair_indices(p)
    reach = 2.0 * D * (1.0 + PAIR_SKIN)
    sel = np.nonzero(_metric_rows(X[iu] - X[ju], metric) < reach)[0]
    _PAIR_LIST[key] = (X.copy(), reach, sel)
    return sel


def _dispersion_and_grad(X, D, metric, grad):
    """Dispersion term of E_and_grad: adds its gradient into grad, returns
    (value, dE/dD).

    Only candidate pairs are measured, but every quantity is formed as the
    full computation forms it -- the value as a dot product over the full,
    zero-filled pair vector -- so the result is identical to measuring all
    pairs.
    """
    p = X.shape[0]
    if p <= 1:
        return 0.0, 0.0
    m = metric.upper()
    if m not in ("EUCLIDEAN", "TAXICAB", "SUP_NORM"):
        raise ValueError(f"Unknown metric '{metric}'")
    iu, ju = _pair_indices(p)
    sel = _candidate_pairs(X, D, metric)
    ci, cj = iu[sel], ju[sel]
    diffs = X[ci] - X[cj]
    uc = (2.0 * D) - _metric_rows(diffs, metric)
    uc[uc < 0.0] = 0.0
    u = np.zeros(iu.shape[0])
    u[sel] = uc
    total = float(np.dot(u, u))
    dE_dD = 0.0
    act = uc > 0.0
    if np.any(act):
        g = _metric_subgrad(diffs[act], metric)
        coeff = (-2.0 * uc[act])[:, None] * g
        np.add.at(grad, ci[act], coeff)
        np.add.at(grad, cj[act], -coeff)
        dE_dD = float(np.sum(4.0 * uc[act]))
    return total, dE_dD


def _E_and_grad_batched(
    X: np.ndarray,
    D: float,
    container: Container,
    metric: str = "EUCLIDEAN",
) -> Tuple[float, np.ndarray, float]:
    """E_and_grad with the container term evaluated for every point at once.

    The same features, the same tie-breaking and the same rules as
    _E_and_grad_loop; only the order in which terms are summed differs. Each
    point's boundary features are laid out as one row of candidates in the
    loop's order -- vertices, then active footpoints by face, then edges -- so
    a first-occurrence argmin reproduces the loop's "min" and exterior choices.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 3)
    p = X.shape[0]
    grad = np.zeros_like(X)
    total, dE_dD = _dispersion_and_grad(X, D, metric, grad)
    if p == 0:
        return float(total), grad.ravel(), float(dE_dD)

    use_min = BOUNDARY_MODE == "min"
    V = _get_verts(container)
    emin, emax, edges = _get_edges_bbox(container)
    ES, EE = _get_edge_arrays(container)
    fa = _face_arrays(container)
    N, P0, F = fa["N"], fa["P0"], fa["F"]
    nV, nE = V.shape[0] if V.size else 0, (emin.shape[0] if edges else 0)

    inside = container.contains_many(X)                                    # (p,)

    # Candidate distances (p, nV + F + nE), +inf where a feature does not apply,
    # with the gradient of each distance alongside.
    dist = np.full((p, nV + F + nE), np.inf)
    gdist = np.zeros((p, nV + F + nE, 3))

    if nV:
        dvec = X[:, None, :] - V[None, :, :]                                # (p, nV, 3)
        dist[:, :nV] = _pairwise_dists(X, V, metric)
        gdist[:, :nV] = _metric_subgrad(dvec.reshape(-1, 3), metric).reshape(p, nV, 3)

    if F:
        hs = np.einsum("fj,pfj->pf", N, X[:, None, :] - P0[None, :, :])    # (p, F)
        act = np.where(inside[:, None], hs >= -_EPS_PLANE, hs < 0.0)
        _, keep, _ = _active_feet_many(container, X, act, metric, hs=hs)
        mn = _face_normal_norms(container, metric)                          # (F,)
        dist[:, nV:nV + F] = np.where(keep, np.abs(hs) / mn[None, :], np.inf)
        gdist[:, nV:nV + F] = (np.sign(hs) / mn[None, :])[:, :, None] * N[None, :, :]

    if nE and inside.any():
        # Edges count only for interior points, and only those whose box
        # grown by D contains the point.
        near = np.all((X[:, None, :] >= (emin - D)[None, :, :]) &
                      (X[:, None, :] <= (emax + D)[None, :, :]), axis=2) & inside[:, None]
        pi, ei = np.nonzero(near)
        if pi.size:
            d_e, _, g_e = segment_distance_grad(X[pi], ES[ei], EE[ei], metric)
            dist[pi, nV + F + ei] = d_e
            gdist[pi, nV + F + ei] = g_e

    # Each term is formed exactly as the loop forms it, and the terms are
    # summed one at a time in the loop's order -- point by point, features in
    # candidate order. Summing them any other way changes E in the last bit,
    # and L-BFGS amplifies that: same-seed runs of the two paths diverged
    # within ~100 evaluations and finished at different optima.
    t_rows, t_val, t_dD = [], [], []

    ins = np.nonzero(inside)[0]
    if ins.size:
        if use_min:
            k = np.argmin(dist[ins], axis=1)
            dsel = dist[ins, k]
            ok = np.isfinite(dsel)
            rows, k, dsel = ins[ok], k[ok], dsel[ok]
            t = D - dsel
            pos = t > 0.0
            rows, t, gsel = rows[pos], t[pos], gdist[rows[pos], k[pos]]
        else:
            sub = dist[ins]
            r, c = np.nonzero(sub < D)          # row-major: point, then feature
            rows = ins[r]
            t = D - sub[r, c]
            gsel = gdist[rows, c]
        if rows.size:
            np.add.at(grad, rows, -(ALPHA * 2.0 * t)[:, None] * gsel)
            t_rows.append(rows); t_val.append(ALPHA * t * t); t_dD.append(ALPHA * 2.0 * t)

    out = np.nonzero(~inside)[0]
    if out.size:
        # Exterior: nearest vertex or footpoint, never an edge.
        sub = dist[out][:, :nV + F]
        if sub.shape[1]:
            k = np.argmin(sub, axis=1)
            dmin = sub[np.arange(out.size), k]
            ok = np.isfinite(dmin)
            rows, k, dmin = out[ok], k[ok], dmin[ok]
            base = D + EXTERIOR_SLOPE * dmin
            np.add.at(grad, rows, (ALPHA * 2.0 * base * EXTERIOR_SLOPE)[:, None] * gdist[rows, k])
            t_rows.append(rows); t_val.append(ALPHA * base * base); t_dD.append(ALPHA * 2.0 * base)

    if t_rows:
        order = np.argsort(np.concatenate(t_rows), kind="stable")
        # cumsum accumulates strictly left to right, as `total +=` does.
        total = float(np.cumsum(np.concatenate([[total], np.concatenate(t_val)[order]]))[-1])
        dE_dD = float(np.cumsum(np.concatenate([[dE_dD], np.concatenate(t_dD)[order]]))[-1])

    return float(total), grad.ravel(), float(dE_dD)


def phi_and_grad(
    X: np.ndarray,
    D: float,
    mu: float,
    container: Container,
    metric: str = "EUCLIDEAN",
) -> Tuple[float, np.ndarray]:
    """phi = mu * E(X, D) - D^2 with its gradient over the joint (X, D) vector.

    Returns (phi, grad of length 3p + 1) for LocalOptPhi.
    """
    e, gX, gD = E_and_grad(X, D, container, metric)
    val = mu * e - D * D
    return float(val), np.append(mu * gX, mu * gD - 2.0 * D)
