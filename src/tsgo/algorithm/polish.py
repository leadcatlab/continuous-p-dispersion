"""Final tightening of a configuration: push D to the exact local maximum.

TSGO reports D = max_feasible_D(X), which is exact for the X it holds, but the
SUMT phase that produced X stops a few parts in 1e5 short of the configuration's
true local optimum (cube l2, p = 50 and 100: 2-4e-5 below Packomania's values
with the same structure). This closes that gap without any search.

The problem is

    maximise t   s.t.   ||c_i - c_j||   >= 2t   for every pair
                        d(c_i, feature) >= t    for every boundary feature,

and every distance in it is a convex function of the points: a norm of a
difference, or the distance to a convex set (a vertex, a segment, a face's
plane). A convex function lies above each of its tangent planes, so replacing
each distance by its linearisation at the current X,

    d(x) >= d(x0) + <g, x - x0>,   g a subgradient at x0,

gives constraints that are *sufficient*: any (X, t) satisfying the linearised
program satisfies the original one for the constraints it includes. Each step
is therefore a small LP, solved exactly. Constraints far from binding are left
out and a trust region keeps them far; the result of every step is certified
by max_feasible_D regardless, and kept only if the certified radius rises, so
the final radius is never below the input's.

General in container and metric: the subgradients are the ones the energy
already uses (_metric_subgrad, face normals over the dual norm,
segment_distance_grad).
"""
from typing import Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import tsgo.algorithm.energy_functions as ef
from tsgo.container.norms import segment_distance_grad

# Largest metric norm of a displacement with every coordinate within delta.
_BOX_TO_NORM = {"EUCLIDEAN": np.sqrt(3.0), "TAXICAB": 3.0, "SUP_NORM": 1.0}


def _constraints(X, D0, container, metric, delta):
    """Linearised constraints near binding at X, as rows of
    sum(coef * x[col]) + k_t * t <= rhs."""
    p = X.shape[0]
    c = _BOX_TO_NORM[metric.upper()] * delta
    rows, cols, vals, tcoef, rhs = [], [], [], [], []

    def add(point_terms, kt, b):
        r = len(rhs)
        for i, g, sign in point_terms:
            for k in range(3):
                if g[k] != 0.0:
                    rows.append(r); cols.append(3 * i + k); vals.append(sign * g[k])
        tcoef.append(kt); rhs.append(b)

    # Pairs: ||x_i - x_j|| >= <g, x_i - x_j> >= 2t, for pairs that could come
    # within 2 * (D0 + c) of each other after moving c each.
    if p > 1:
        iu, ju = ef._pair_indices(p)
        diffs = X[iu] - X[ju]
        d = ef._metric_rows(diffs, metric)
        near = np.nonzero(d < 2.0 * D0 + 4.0 * c)[0]
        G = ef._metric_subgrad(diffs[near], metric)
        for n, k in enumerate(near):
            # -<g, x_i> + <g, x_j> + 2t <= 0
            add([(iu[k], G[n], -1.0), (ju[k], G[n], 1.0)], 2.0, 0.0)

    # Boundary: d(x0) + <g, x - x0> >= t, i.e. -<g, x> + t <= d(x0) - <g, x0>.
    reach = D0 + 2.0 * c
    V = ef._get_verts(container)
    if V.size:
        diff = X[:, None, :] - V[None, :, :]
        dv = ef._pairwise_dists(X, V, metric)
        for i, v in zip(*np.nonzero(dv < reach)):
            g = ef._metric_subgrad(diff[i, v][None, :], metric)[0]
            add([(i, g, -1.0)], 1.0, dv[i, v] - g @ X[i])

    fa = ef._face_arrays(container)
    if fa["F"]:
        inside = container.contains_many(X)
        hs = np.einsum("fj,pfj->pf", fa["N"], X[:, None, :] - fa["P0"][None, :, :])
        act = np.where(inside[:, None], hs >= -ef._EPS_PLANE, hs < 0.0)
        _, keep, _ = ef._active_feet_many(container, X, act, metric, hs=hs)
        mn = ef._face_normal_norms(container, metric)
        df = np.abs(hs) / mn[None, :]
        for i, f in zip(*np.nonzero(keep & (df < reach))):
            g = (np.sign(hs[i, f]) / mn[f]) * fa["N"][f]
            add([(i, g, -1.0)], 1.0, df[i, f] - g @ X[i])

    ES, EE = ef._get_edge_arrays(container)
    if ES.shape[0]:
        emin, emax = np.minimum(ES, EE), np.maximum(ES, EE)
        box = np.all((X[:, None, :] >= (emin - reach)[None]) & (X[:, None, :] <= (emax + reach)[None]), axis=2)
        pi, ei = np.nonzero(box)
        if pi.size:
            de, _, ge = segment_distance_grad(X[pi], ES[ei], EE[ei], metric)
            for n in np.nonzero(de < reach)[0]:
                i = pi[n]
                add([(i, ge[n], -1.0)], 1.0, de[n] - ge[n] @ X[i])

    m = len(rhs)
    A = coo_matrix((vals, (rows, cols)), shape=(m, 3 * p)).tocsr()
    return A, np.asarray(tcoef), np.asarray(rhs), c


def polish(X: np.ndarray, container, metric: str = "EUCLIDEAN", max_iter: int = 200,
           delta0: float = 1e-2, rtol: float = 1e-13) -> Tuple[np.ndarray, float]:
    """Raise the certified radius of X to its local maximum by sequential LP.

    Returns (X, D) with D = max_feasible_D(X) * (1 - 1e-9), the same safety
    margin TSGO applies, and never below what X started with.
    """
    from scipy.sparse import hstack, csr_matrix
    metric = metric.upper()
    X = np.asarray(X, dtype=float).reshape(-1, 3).copy()
    p = X.shape[0]
    D = ef.max_feasible_D(X, container, metric)
    if D <= 0.0 or p == 0:
        return X, D * (1.0 - 1e-9)

    delta = delta0 * D
    stall = 0
    for _ in range(max_iter):
        A, kt, b, c = _constraints(X, D, container, metric, delta)
        A_full = hstack([A, csr_matrix(kt[:, None])]).tocsr()
        obj = np.zeros(3 * p + 1); obj[-1] = -1.0
        bounds = [(X[k // 3, k % 3] - delta, X[k // 3, k % 3] + delta) for k in range(3 * p)]
        bounds.append((D, D + c))
        res = linprog(obj, A_ub=A_full, b_ub=b, bounds=bounds, method="highs")
        if res.status != 0:
            delta *= 0.25
            if delta < 1e-14 * D:
                break
            continue
        X_new = res.x[:-1].reshape(p, 3)
        D_new = ef.max_feasible_D(X_new, container, metric)
        if D_new > D * (1.0 + rtol):
            gain_hit_trust = res.x[-1] >= D + 0.99 * c
            X, D = X_new, D_new
            stall = 0
            if gain_hit_trust:
                delta *= 2.0
        else:
            # No certified gain: either converged, or the step left the region
            # where the linearisation (or the constraint selection) holds.
            stall += 1
            delta *= 0.25
            if stall >= 3 or delta < 1e-14 * D:
                break
    return X, D * (1.0 - 1e-9)
