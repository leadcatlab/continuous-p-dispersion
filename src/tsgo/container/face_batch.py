"""Vectorised point-in-face tests over every face of a container at once.

Container.contains casts rays and asks, for each face, whether the ray hits
inside it. Done face by face in Python that dominates the solver: profiling a
sup-norm optimisation put 58% of the time in contains -> _single_ray_parity ->
Face.contains, against 9% in the metric geometry. The work per face is a
handful of dot products and an even-odd crossing count, so it vectorises
cleanly once the ragged polygons are padded to a common width.

This is metric-independent: it speeds up every norm equally.

Tolerances are copied exactly from Face.contains and Face._even_odd_contains_2d
so that the batched result agrees with the scalar one point for point --
containment is a boolean that feeds the energy, and a single flipped value
changes a run's trajectory.

Faces carrying holes fall back to the scalar path; only the cage has any
(8 of its 54 faces), and they are a small minority of a container's work.
"""

from typing import Dict, Tuple

import numpy as np

from tsgo.container.face import EPS_PLANE

_CACHE: Dict[int, dict] = {}

EPS_BBOX = 1e-9
EPS_SEG = 1e-10
EPS_XING = 1e-12


def face_arrays(container) -> dict:
    """Padded array form of a container's faces, built once per container."""
    key = id(container)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    faces = list(container.faces)
    F = len(faces)
    for f in faces:                       # ensure the 2-D basis exists
        if not hasattr(f, "_outer2d"):
            f._build_2d_cache()

    vmax = max((f._outer2d.shape[0] for f in faces), default=0)
    N = np.zeros((F, 3)); P0 = np.zeros((F, 3))
    U = np.zeros((F, 3)); W = np.zeros((F, 3))
    poly = np.zeros((F, vmax, 2)); cnt = np.zeros(F, dtype=int)
    bbox = np.zeros((F, 4)); holed = np.zeros(F, dtype=bool)

    for i, f in enumerate(faces):
        N[i] = np.asarray(f.normal, dtype=float)
        P0[i] = np.asarray(f._P0, dtype=float)
        U[i] = f._u
        W[i] = f._w
        n = f._outer2d.shape[0]
        cnt[i] = n
        poly[i, :n] = f._outer2d
        bbox[i] = f._outer_bbox
        holed[i] = bool(getattr(f, "_holes2d", None))

    # nxt[i, k] is the polygon's next vertex, wrapping at that face's own count
    idx = np.arange(vmax)[None, :]
    nxt = (idx + 1) % np.maximum(cnt, 1)[:, None]
    edge_ok = idx < cnt[:, None]

    out = dict(faces=faces, F=F, N=N, P0=P0, U=U, W=W, poly=poly, cnt=cnt,
               nxt=nxt, edge_ok=edge_ok, bbox=bbox, holed=holed, vmax=vmax)
    _CACHE[key] = out
    return out


def faces_contain(container, Q3: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Is Q3[i, r] inside face i? Returns bool of Q3's leading shape.

    Q3 is (F, 3) -- one query point per face, as a single ray cast produces --
    or (F, R, 3) for R rays at once. Batching the rays matters: the per-call
    numpy overhead is fixed while the work is tiny for a container with only a
    few faces, so casting one ray at a time made the cube and the tetrahedron
    slower than the scalar loop they replaced. `mask` optionally restricts
    which faces are tested.
    """
    fa = face_arrays(container)
    F, vmax = fa["F"], fa["vmax"]
    Q3 = np.asarray(Q3, dtype=float)
    batched = Q3.ndim == 3
    R = Q3.shape[1] if batched else 1
    lead = (F, R)

    out = np.zeros(lead, dtype=bool)
    if F == 0 or vmax == 0:
        return out if batched else out[:, 0]

    Q = Q3 if batched else Q3[:, None, :]           # (F, R, 3)

    if mask is None:
        active = np.ones(lead, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        active = (mask if mask.ndim == 2 else mask[:, None]) & np.ones(lead, dtype=bool)
    if not active.any():
        return out if batched else out[:, 0]

    # 1. the point must lie on the face's plane
    V = Q - fa["P0"][:, None, :]                    # (F, R, 3)
    active &= np.abs(np.einsum("fj,frj->fr", fa["N"], V)) <= EPS_PLANE
    if not active.any():
        return out if batched else out[:, 0]

    # 2. project into the face's own 2-D basis
    qx = np.einsum("fj,frj->fr", fa["U"], V)
    qy = np.einsum("fj,frj->fr", fa["W"], V)

    bb = fa["bbox"]
    active &= ((qx >= bb[:, 0:1] - EPS_BBOX) & (qx <= bb[:, 1:2] + EPS_BBOX) &
               (qy >= bb[:, 2:3] - EPS_BBOX) & (qy <= bb[:, 3:4] + EPS_BBOX))
    if not active.any():
        return out if batched else out[:, 0]

    poly = fa["poly"]; nxt = fa["nxt"]; edge_ok = fa["edge_ok"]
    rows = np.arange(F)[:, None]
    # (F, 1, V) so they broadcast against (F, R, 1) queries
    ax = poly[:, :, 0][:, None, :]; ay = poly[:, :, 1][:, None, :]
    bx = poly[rows, nxt, 0][:, None, :]; by = poly[rows, nxt, 1][:, None, :]
    edge_ok = edge_ok[:, None, :]

    QX = qx[:, :, None]; QY = qy[:, :, None]

    # 3. on the boundary counts as inside, matching Face._point_on_segment
    abx = bx - ax; aby = by - ay
    aqx = QX - ax; aqy = QY - ay
    cross = abx * aqy - aby * aqx
    dot = aqx * abx + aqy * aby
    ab2 = abx * abx + aby * aby
    on_seg = (edge_ok & (np.abs(cross) <= EPS_SEG) &
              (dot >= -EPS_SEG) & (dot <= ab2 + EPS_SEG))
    boundary = on_seg.any(axis=2)

    # 4. even-odd crossing count
    straddle = edge_ok & ((ay > QY) != (by > QY))
    with np.errstate(divide="ignore", invalid="ignore"):
        x_int = ax + (bx - ax) * (QY - ay) / (by - ay)
    crossings = straddle & np.isfinite(x_int) & (x_int >= QX - EPS_XING)
    inside = (crossings.sum(axis=2) % 2) == 1

    out = active & (boundary | inside)

    # 5. holed faces are rare; defer to the scalar routine for exactness
    holed = fa["holed"][:, None] & active
    if holed.any():
        faces = fa["faces"]
        for i, r in zip(*np.nonzero(holed)):
            out[i, r] = faces[i].contains(Q[i, r])
    return out if batched else out[:, 0]


def active_feet(container, x, active, metric="EUCLIDEAN"):
    """Footpoint of every active face, and whether it lies on that face.

    Returns ``(feet, keep, hs)`` with ``feet`` shape (F, 3), ``keep`` a boolean
    mask of the faces contributing an *active footpoint*, and ``hs`` the signed
    plane offsets.

    Why this is not a one-liner. Under a polyhedral norm the nearest point of a
    plane is generally **not unique**: the minimising directions are
    ``U* = {u : ||u|| <= 1, <N,u> = ||N||_*}``, a whole face of the unit ball
    whenever the ball has a face parallel to the plane. Every ``u in U*`` gives a
    foot at the same distance ``|h| / ||N||_*``, but at a *different place* on the
    plane -- and the place decides whether the foot lands on the face, hence
    whether the face contributes to the penalty at all.

    Picking one minimiser by ``argmax`` therefore made the energy depend on the
    order the unit ball's vertices happen to be listed in. For the cube's wall
    ``x = -1/2`` under the sup norm, four of the eight vertices tie, and the
    first of them drags the foot diagonally in y and z -- off the face, for a
    point whose own wall is right there.

    So we search ``U*`` instead of picking from it. All tied candidates share one
    distance, so the *distance* is tie-break independent; what the search decides
    is the geometric question "does some nearest point of this plane lie on this
    face?", which has a single answer. The centroid of the tied vertices is tried
    first: it lies in ``U*`` too (a convex combination of points of one ball face,
    so still of norm 1 and still attaining the max), it is the most interior
    choice, and for an axis-aligned normal it is exactly the orthogonal
    direction -- so this reduces to the familiar foot in every case where the
    familiar foot is right.

    Both the scalar energy and the analytic-gradient path call this, which is the
    point: they previously disagreed about which faces were active under the sup
    norm -- ``E`` used the metric foot, ``E_and_grad`` the Euclidean one -- so the
    optimiser was descending a different function from the one reported.
    """
    fa = face_arrays(container)
    F = fa["F"]
    x = np.asarray(x, dtype=float)
    N, P0 = fa["N"], fa["P0"]
    hs = np.einsum("fj,fj->f", N, x[None, :] - P0)
    active = np.asarray(active, dtype=bool)

    if F == 0 or not active.any():
        return np.zeros((F, 3)), np.zeros(F, dtype=bool), hs

    from tsgo.container.norms import get_norm
    BV = get_norm(metric).ball_vertices

    if BV is None:
        # l_2: the minimiser is unique, u = N / ||N||_2, and N is already unit.
        feet = x[None, :] - hs[:, None] * N
        return feet, faces_contain(container, feet, mask=active) & active, hs

    vals = N @ BV.T                                        # (F, K)
    K = BV.shape[0]
    target = np.where(hs >= 0.0, vals.max(axis=1), vals.min(axis=1))   # (F,)
    tied = np.abs(vals - target[:, None]) <= 1e-12         # (F, K)

    # Centroid of each face's tied directions, prepended as candidate 0.
    cnt = tied.sum(axis=1, keepdims=True)
    centroid = (tied[:, :, None] * BV[None, :, :]).sum(axis=1) / np.maximum(cnt, 1)
    U = np.concatenate([centroid[:, None, :], np.broadcast_to(BV, (F, K, 3))], axis=1)
    ok = np.concatenate([cnt > 0, tied], axis=1)           # (F, K+1)

    # s is the same for every tied candidate, so all candidate feet of a face sit
    # at one distance; only their positions differ.
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(target != 0.0, hs / target, 0.0)
    cand = x[None, None, :] - s[:, None, None] * U         # (F, K+1, 3)

    # Two passes, because the first almost always settles it. The centroid is
    # the candidate we want anyway, and a face whose maximiser is unique -- every
    # slanted face, hence the whole of the tetrahedron and the star -- has no
    # other candidate to try. Testing all K+1 up front cost 1.4-1.9x the runtime
    # of the single test this replaced; testing the centroid alone and falling
    # back only where it misses *and* a genuine tie exists costs almost nothing.
    rows = np.arange(F)
    keep = faces_contain(container, cand[:, 0, :], mask=active) & active
    feet = cand[:, 0, :].copy()

    retry = active & ~keep & (cnt[:, 0] > 1)
    if retry.any():
        mask = ok[:, 1:] & retry[:, None]
        hit = faces_contain(container, cand[:, 1:, :], mask=mask) & mask   # (F, K)
        found = hit.any(axis=1)
        idx = np.nonzero(found)[0]
        if idx.size:
            feet[idx] = cand[idx, 1 + np.argmax(hit[idx], axis=1)]
            keep[idx] = True

    return feet, keep, hs


def active_feet_many(container, X, active, metric="EUCLIDEAN", hs=None):
    """active_feet for every row of X at once.

    X is (P, 3) and ``active`` (P, F). Returns ``(feet, keep, hs)`` with shapes
    (P, F, 3), (P, F) and (P, F); row i is what active_feet(container, X[i],
    active[i], metric) returns. Pass ``hs`` when the caller already has the
    plane offsets.
    """
    fa = face_arrays(container)
    F = fa["F"]
    X = np.atleast_2d(np.asarray(X, dtype=float))
    P = X.shape[0]
    N, P0 = fa["N"], fa["P0"]
    if hs is None:
        hs = np.einsum("fj,pfj->pf", N, X[:, None, :] - P0[None, :, :])
    active = np.asarray(active, dtype=bool)

    if F == 0 or P == 0 or not active.any():
        return np.zeros((P, F, 3)), np.zeros((P, F), dtype=bool), hs

    from tsgo.container.norms import get_norm
    BV = get_norm(metric).ball_vertices

    if BV is None:
        feet = X[:, None, :] - hs[:, :, None] * N[None, :, :]               # (P, F, 3)
        keep = faces_contain(container, feet.transpose(1, 0, 2),
                             mask=active.T).T & active
        return feet, keep, hs

    vals = N @ BV.T                                                         # (F, K)
    K = BV.shape[0]
    target = np.where(hs >= 0.0, vals.max(axis=1)[None, :], vals.min(axis=1)[None, :])  # (P, F)
    tied = np.abs(vals[None, :, :] - target[:, :, None]) <= 1e-12           # (P, F, K)

    cnt = tied.sum(axis=2, keepdims=True)                                   # (P, F, 1)
    centroid = (tied[..., None] * BV[None, None, :, :]).sum(axis=2) / np.maximum(cnt, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(target != 0.0, hs / target, 0.0)                       # (P, F)

    # First pass: the centroid candidate only, as in active_feet.
    feet = X[:, None, :] - s[:, :, None] * centroid                         # (P, F, 3)
    keep = faces_contain(container, feet.transpose(1, 0, 2), mask=active.T).T & active

    retry = active & ~keep & (cnt[:, :, 0] > 1)
    if retry.any():
        cand = X[:, None, None, :] - s[:, :, None, None] * BV[None, None, :, :]   # (P, F, K, 3)
        mask = tied & retry[:, :, None]                                     # (P, F, K)
        hit = faces_contain(container,
                            cand.transpose(1, 0, 2, 3).reshape(F, P * K, 3),
                            mask=mask.transpose(1, 0, 2).reshape(F, P * K))
        hit = hit.reshape(F, P, K).transpose(1, 0, 2) & mask
        pi, fi = np.nonzero(hit.any(axis=2))
        if pi.size:
            feet[pi, fi] = cand[pi, fi, np.argmax(hit[pi, fi], axis=1)]
            keep[pi, fi] = True

    return feet, keep, hs
