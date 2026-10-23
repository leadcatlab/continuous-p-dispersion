"""Norms on R^3, and point-to-segment distance under any of them.

The distance from a point x to the segment [s, e] is

    min_{t in [0,1]} || (x - s) - t (e - s) ||,

a norm composed with an affine map, hence **convex in t for every norm**. So
this is always a one-dimensional convex minimisation, never a per-metric
derivation. What differs between metrics is only how the minimum is located.

A norm is *polyhedral* when its unit ball is a polytope. Equivalently its dual
unit ball is a polytope with finitely many vertices g_1..g_K, and

    ||z|| = max_k <g_k, z>.

For such a norm the distance becomes

    d(t) = max_k ( <g_k, x-s> - t <g_k, e-s> ) = max_k ( a_k + t b_k ),

the upper envelope of K lines: convex and piecewise linear. Its minimum over
[0,1] is attained at an endpoint or where two of the lines cross, so the exact
minimiser comes from a finite candidate set with no iteration and no search.

This is the whole method. A metric enters only as its generator matrix:

    l_inf  dual ball is the l_1 ball     -> the 6 vectors +-e_i
    l_1    dual ball is the l_inf ball   -> the 8 vectors (+-1,+-1,+-1)

and a new polyhedral norm -- weighted, anisotropic, hexagonal -- is added by
supplying its generators, not by writing new geometry code. l_2 is the
classical smooth case: its dual ball is not a polytope, so it keeps the exact
closed-form projection instead. Any further smooth norm can fall back to
one-dimensional convex minimisation on the same d(t).

Everything here is vectorised over (point, segment) pairs, which is what makes
the polyhedral path fast: the per-pair work is a handful of small dense
products rather than a Python loop over candidate parameters.
"""

from typing import Optional, Tuple

import numpy as np

__all__ = ["Norm", "EUCLIDEAN", "TAXICAB", "SUP_NORM", "get_norm", "segment_distance"]


class Norm:
    """A norm on R^3.

    `generators` holds the vertices of the dual unit ball, so that
    ||z|| = max_k <g_k, z>. It is None for norms that are not polyhedral.
    """

    def __init__(self, name: str, generators: Optional[np.ndarray] = None):
        self.name = name
        self.generators = None if generators is None else np.asarray(generators, dtype=float)

    @property
    def is_polyhedral(self) -> bool:
        return self.generators is not None

    def value(self, Z: np.ndarray) -> np.ndarray:
        """||z|| for each row of Z, shape (n,)."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if self.is_polyhedral:
            return np.max(Z @ self.generators.T, axis=1)
        return np.sqrt(np.sum(Z * Z, axis=1))

    def subgradient(self, Z: np.ndarray) -> np.ndarray:
        """A subgradient of ||.|| at each row of Z, shape (n, 3).

        For a polyhedral norm the maximising generator is a subgradient; for
        l_2 it is the unit vector along z, and 0 at the origin.
        """
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if self.is_polyhedral:
            k = np.argmax(Z @ self.generators.T, axis=1)
            return self.generators[k]
        d = np.sqrt(np.sum(Z * Z, axis=1))
        out = np.zeros_like(Z)
        nz = d > 0.0
        out[nz] = Z[nz] / d[nz, None]
        return out


# l_inf: dual unit ball is the l_1 ball, whose vertices are +-e_i.
_SUP_GEN = np.array([[1.0, 0, 0], [-1.0, 0, 0],
                     [0, 1.0, 0], [0, -1.0, 0],
                     [0, 0, 1.0], [0, 0, -1.0]])

# l_1: dual unit ball is the l_inf ball, whose vertices are the sign vectors.
_TAX_GEN = np.array([[sx, sy, sz]
                     for sx in (1.0, -1.0)
                     for sy in (1.0, -1.0)
                     for sz in (1.0, -1.0)])

EUCLIDEAN = Norm("EUCLIDEAN", None)      # not polyhedral: closed-form projection
TAXICAB = Norm("TAXICAB", _TAX_GEN)
SUP_NORM = Norm("SUP_NORM", _SUP_GEN)

# Vertices of each norm's own unit ball. For a polyhedral norm the unit ball of
# one is the dual ball of the other, so these are just the generator sets
# swapped: the l_inf ball has the eight sign vectors as vertices, the l_1 ball
# the six +-e_i. Used to project onto a plane, where the minimiser lies along a
# ball vertex.
SUP_NORM.ball_vertices = _TAX_GEN
TAXICAB.ball_vertices = _SUP_GEN
EUCLIDEAN.ball_vertices = None

_BY_NAME = {
    "EUCLIDEAN": EUCLIDEAN, "L2": EUCLIDEAN,
    "TAXICAB": TAXICAB, "L1": TAXICAB,
    "SUP_NORM": SUP_NORM, "LINF": SUP_NORM,
}


def get_norm(metric) -> Norm:
    if isinstance(metric, Norm):
        return metric
    try:
        return _BY_NAME[str(metric).upper()]
    except KeyError:
        raise ValueError(f"Unknown metric '{metric}'. "
                         f"Known: {sorted(set(n.name for n in _BY_NAME.values()))}")


def segment_distance(X: np.ndarray, S: np.ndarray, E: np.ndarray,
                     metric="EUCLIDEAN") -> Tuple[np.ndarray, np.ndarray]:
    """Distance from each point to each segment, with the closest points.

    X, S, E are (n, 3) and broadcast against each other. Returns (d, C) with
    d shape (n,) and C shape (n, 3), C[i] being the point of segment
    [S[i], E[i]] closest to X[i] under the norm.
    """
    norm = get_norm(metric)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    S = np.atleast_2d(np.asarray(S, dtype=float))
    E = np.atleast_2d(np.asarray(E, dtype=float))
    X, S, E = np.broadcast_arrays(X, S, E)
    A = X - S                      # (n, 3)
    V = E - S                      # (n, 3)

    if not norm.is_polyhedral:
        # l_2: the minimiser of a smooth strictly convex quadratic in t.
        denom = np.einsum("ij,ij->i", V, V)
        t = np.zeros(A.shape[0])
        nz = denom > 0.0
        t[nz] = np.clip(np.einsum("ij,ij->i", A[nz], V[nz]) / denom[nz], 0.0, 1.0)
        C = S + t[:, None] * V
        return np.sqrt(np.sum((X - C) ** 2, axis=1)), C

    G = norm.generators                        # (K, 3)
    al = A @ G.T                               # (n, K)  intercepts
    be = -(V @ G.T)                            # (n, K)  slopes

    # Candidate minimisers: the two endpoints, plus every crossing of two of
    # the K lines. The envelope's kinks are a subset of those crossings, so
    # this finite set is guaranteed to contain the exact minimiser.
    K = G.shape[0]
    iu, ju = np.triu_indices(K, 1)
    den = be[:, iu] - be[:, ju]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cross = (al[:, ju] - al[:, iu]) / den
    bad = ~np.isfinite(t_cross) | (t_cross <= 0.0) | (t_cross >= 1.0)
    t_cross = np.where(bad, 0.0, t_cross)      # fold invalid onto t=0, already a candidate

    n = A.shape[0]
    T = np.concatenate([np.zeros((n, 1)), np.ones((n, 1)), t_cross], axis=1)   # (n, M)
    # d(t) = max_k (al_k + t be_k), evaluated at every candidate at once.
    D = np.max(al[:, None, :] + T[:, :, None] * be[:, None, :], axis=2)        # (n, M)

    best = np.argmin(D, axis=1)
    rows = np.arange(n)
    t_best = T[rows, best]
    return D[rows, best], S + t_best[:, None] * V


def segment_distance_grad(X: np.ndarray, S: np.ndarray, E: np.ndarray,
                          metric="EUCLIDEAN") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment distance, closest point, and the gradient with respect to X.

    Two regimes, both derived from d(t) = max_k (a_k + t b_k) with
    a_k = <g_k, x - s> and b_k = -<g_k, e - s>; only a_k depends on x, and
    da_k/dx = g_k.

    Minimum at an endpoint (t* = 0 or 1): t* cannot move, so Danskin gives

        dd/dx = g_{k*},   k* the generator attaining the max there.

    Minimum at a kink where lines i and j cross: t* moves with x, and
    t* = (a_j - a_i) / (b_i - b_j) gives dt*/dx = (g_j - g_i) / (b_i - b_j),
    so differentiating d = a_i + t* b_i,

        dd/dx = g_i + b_i (g_j - g_i) / (b_i - b_j).

    This is the general statement of the implicit differentiation that the
    sup norm needs at a two-component tie, and it holds for any polyhedral
    norm -- nothing here is specific to l_inf or l_1.
    """
    norm = get_norm(metric)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    S = np.atleast_2d(np.asarray(S, dtype=float))
    E = np.atleast_2d(np.asarray(E, dtype=float))
    X, S, E = np.broadcast_arrays(X, S, E)

    if not norm.is_polyhedral:
        d, C = segment_distance(X, S, E, norm)
        return d, C, norm.subgradient(X - C)

    A = X - S
    V = E - S
    G = norm.generators
    al = A @ G.T
    be = -(V @ G.T)

    K = G.shape[0]
    iu, ju = np.triu_indices(K, 1)
    den = be[:, iu] - be[:, ju]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_cross = (al[:, ju] - al[:, iu]) / den
    bad = ~np.isfinite(t_cross) | (t_cross <= 0.0) | (t_cross >= 1.0)
    t_cross = np.where(bad, 0.0, t_cross)

    n = A.shape[0]
    T = np.concatenate([np.zeros((n, 1)), np.ones((n, 1)), t_cross], axis=1)
    D = np.max(al[:, None, :] + T[:, :, None] * be[:, None, :], axis=2)

    best = np.argmin(D, axis=1)
    rows = np.arange(n)
    t_best = T[rows, best]
    d_best = D[rows, best]
    C = S + t_best[:, None] * V

    # Which lines actually form the envelope at t*. Several pairs can cross at
    # the same t, so the pair that produced the winning candidate need not be
    # the pair defining the kink -- the active set decides, not the candidate
    # index.
    vals = al + t_best[:, None] * be                    # (n, K)
    active = vals >= (d_best[:, None] - 1e-12)

    # At an interior kink the envelope descends into t* and climbs out of it,
    # so the active set holds a line of negative slope and one of positive
    # slope. i is the steepest descending, j the steepest ascending.
    NEG, POS = np.where(active, be, np.inf), np.where(active, be, -np.inf)
    i = np.argmin(NEG, axis=1)
    j = np.argmax(POS, axis=1)
    bi = be[rows, i]
    bj = be[rows, j]

    # A crossing only moves with x if t* is strictly interior and the two
    # bracketing slopes really do straddle zero; otherwise t* is pinned and
    # Danskin applies with the active generator.
    is_cross = (t_best > 0.0) & (t_best < 1.0) & (bi < 0.0) & (bj > 0.0) & (bi != bj)

    grad = G[np.argmax(vals, axis=1)].copy()
    if np.any(is_cross):
        k = np.nonzero(is_cross)[0]
        w = (bi[k] / (bi[k] - bj[k]))[:, None]
        grad[k] = G[i[k]] + w * (G[j[k]] - G[i[k]])

    return d_best, C, grad


def plane_nearest(x: np.ndarray, normal: np.ndarray, p0: np.ndarray,
                  metric="EUCLIDEAN") -> Tuple[float, np.ndarray]:
    """Nearest point of the plane {y : <N, y - p0> = 0} to x, and its distance.

    Under a general norm the nearest point of a plane is *not* the orthogonal
    foot. Writing h = <N, x - p0>, the distance is

        d(x, plane) = |h| / ||N||_* ,

    with ||.||_* the dual norm, and the minimiser is reached by travelling from
    x along a unit vector u maximising <N, u> (or minimising it, when h < 0).
    For a polyhedral norm that maximum is attained at a vertex of the unit
    ball, so the search is over a finite set; for l_2 it is N / ||N||_2 and the
    formula collapses to the familiar orthogonal projection.

    Returns (distance, foot).
    """
    norm = get_norm(metric)
    x = np.asarray(x, dtype=float)
    N = np.asarray(normal, dtype=float)
    h = float(np.dot(N, x - np.asarray(p0, dtype=float)))

    BV = norm.ball_vertices
    if BV is None:                       # l_2: u = N / ||N||_2
        nn = float(np.linalg.norm(N))
        if nn == 0.0:
            return 0.0, x.copy()
        u = N / nn
        s = h / float(np.dot(N, u))
        return abs(s), x - s * u

    vals = BV @ N
    k = int(np.argmax(vals)) if h >= 0.0 else int(np.argmin(vals))
    denom = float(vals[k])
    if denom == 0.0:                     # plane parallel to every ball vertex
        return abs(h), x.copy()
    s = h / denom
    return abs(s), x - s * BV[k]


def dual_norm(v: np.ndarray, metric="EUCLIDEAN") -> float:
    """||v||_* = max{ <v, u> : ||u|| <= 1 }.

    The distance from a point to a hyperplane with normal v is |h| / ||v||_*,
    so this is the quantity a face contributes through, not ||v|| itself.
    """
    norm = get_norm(metric)
    v = np.asarray(v, dtype=float)
    BV = norm.ball_vertices
    if BV is None:
        return float(np.linalg.norm(v))
    return float(np.max(BV @ v))
