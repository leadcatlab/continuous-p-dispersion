"""Exact point-in-polyhedron test by signed tetrahedra (Feito and Torres, 1997).

Every face ring is fanned into triangles, oriented so the triangle normals of
an outer ring point out of the container and those of a hole ring point into
it. With a fixed origin O outside the container, each triangle (a, b, c)
spans a tetrahedron (O, a, b, c) whose signed volume has the sign s of
det[a - O, b - O, c - O]. The signed tetrahedra sum to the container as a
chain, so a point x is inside exactly when

    sum over triangles of s * [x lies in (O, a, b, c)]  !=  0.

Unlike ray casting there are no directions to choose and no vote: away from a
measure-zero set the count is exact. That set is where x lies on the boundary
of some tetrahedron -- a plane through O and a triangle edge, or a triangle's
own plane. Such points are flagged and counted again from a second origin;
a point flagged from both (in practice one on a face's plane, which every
origin sees on a tetrahedron's boundary) is left to the caller.

A fan of a non-convex or holed polygon covers some of its area with opposite
signs, so its triangles are signed rather than disjoint; the count is still
exact, since the signs cancel wherever the fan overlaps itself.
"""

import numpy as np

# How close a barycentric coordinate may come to a tetrahedron's boundary
# before the point is treated as ambiguous for that origin. Coordinates are
# dimensionless, so one tolerance serves every container.
AMBIGUOUS = 1e-10


def _ring(points) -> np.ndarray:
    return np.asarray([v.as_array() for v in points], dtype=float)


def _newell(R: np.ndarray) -> np.ndarray:
    return np.cross(R, np.roll(R, -1, axis=0)).sum(axis=0)


def _normal(face) -> np.ndarray:
    n = face.normal
    return np.asarray(n.as_array() if hasattr(n, 'as_array') else n, dtype=float)


def signed_triangles(container) -> np.ndarray:
    """(T, 3, 3): the fan triangles of every face ring, consistently oriented."""
    tris = []
    for f in container.faces:
        n_in = _normal(f)
        rings = [(_ring(f.vertices), False)] + [(_ring(h), True) for h in (f.holes or [])]
        for R, is_hole in rings:
            points_out = np.dot(_newell(R), n_in) < 0.0
            if points_out == is_hole:  # outer rings point out, holes point in
                R = R[::-1]
            for i in range(1, len(R) - 1):
                tris.append((R[0], R[i], R[i + 1]))
    return np.asarray(tris, dtype=float).reshape(-1, 3, 3)


def _tetrahedra(T: np.ndarray, O: np.ndarray) -> dict:
    M = np.stack([T[:, 0] - O, T[:, 1] - O, T[:, 2] - O], axis=2)  # columns a-O, b-O, c-O
    det = np.linalg.det(M)
    scale = np.abs(M).max(axis=(1, 2)) ** 3
    keep = np.abs(det) > 1e-12 * np.maximum(scale, 1e-300)  # collinear fan triangles add nothing
    return {'O': O, 'Minv': np.linalg.inv(M[keep]), 'sign': np.sign(det[keep]).astype(np.int64)}


def prepare(container) -> dict:
    """Signed tetrahedra from two fixed origins, built once per container.

    The origins lie outside the bounding box, off every axis and in different
    directions, so a plane through one origin and a triangle edge is unlikely
    to pass through the other. Both are fixed: the test is deterministic.
    """
    T = signed_triangles(container)
    V = T.reshape(-1, 3)
    lo, hi = V.min(axis=0), V.max(axis=0)
    span = float((hi - lo).max())
    O1 = lo - span * np.array([0.5137, 0.4231, 0.3719])
    O2 = hi + span * np.array([0.3613, 0.5297, 0.4471])
    return {'T': T.shape[0], 'origins': [_tetrahedra(T, O1), _tetrahedra(T, O2)]}


def _count(tet: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per point: (inside verdict, ambiguous) from one origin."""
    lam = np.einsum('tij,pj->tpi', tet['Minv'], X - tet['O'])  # (T, P, 3)
    s = lam.sum(axis=2)
    near = (lam >= -AMBIGUOUS).all(axis=2) & (s <= 1.0 + AMBIGUOUS)
    strict = (lam > AMBIGUOUS).all(axis=2) & (s < 1.0 - AMBIGUOUS)
    count = (tet['sign'][:, None] * strict).sum(axis=0)
    return count != 0, (near & ~strict).any(axis=0)


def classify(prep: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inside verdict for every row of X, and a mask of rows ambiguous from
    both origins, whose verdict the caller must supply."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    inside, unresolved = _count(prep['origins'][0], X)
    if unresolved.any():
        idx = np.nonzero(unresolved)[0]
        second, still = _count(prep['origins'][1], X[idx])
        inside[idx] = second
        unresolved[idx] = still
    return inside, unresolved
