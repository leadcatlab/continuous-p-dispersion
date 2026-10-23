"""The eight containers of Stoyan & Yaskov (2013), rebuilt as face lists.

Yu. G. Stoyan and G. N. Yaskov, "Packing congruent spheres into a
multi-connected polyhedral domain", Intl. Trans. in Op. Res. 20 (2013) 79-99,
section 4. Each container is a convex polytope with convex pieces removed,

    A = cl(P \\ (P_1 u ... u C_1 u ... u D_1 u ...)),

where P is the truncated box of their Table 1 (examples 1-5) or the cuboid
9.39 x 10.43 x 7.83 (examples 6-8), P_j are through-prisms, C_l polyhedral
cones and D_t dihedral angles, all given in the paper by a few points. They
pack spheres of radius 1 (the centres of their Table 2 sit exactly 1 from the
walls) and report the largest number packed:

    example   1   2   3   4   5   6   7   8
    spheres  93  91  86  81  79  30  58  73
    minutes  41  34  47  54  60   2   7  12   (AMD Athlon 64 X2 6000+, Delphi)

The boundary of A is computed here and written as data/sy_ex<k>.json, the
format every container uses, so the solver and the exact radius certificate
treat these containers like any other. A face of P keeps what lies outside
every removed piece; a face of a removed piece keeps what lies inside P and
outside the other pieces. Unbounded pieces are clipped to a box far larger
than P, which changes nothing inside P. Needs shapely, for the polygon
differences, only to regenerate the files:

    python -m tsgo.container.stoyan_yaskov
"""
import json
import os
from typing import Dict, List, Tuple

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAR = 100.0          # half-width of the box that bounds the unbounded pieces
MIN_AREA = 1e-9

# Table 1: the 24 vertices of P, a 9.39 x 10.43 x 7.83 box with every corner cut.
TABLE1 = [(3.13, 0, 0), (0, 3.65, 0), (0, 0, 2.61), (6.26, 0, 0), (9.39, 4.7, 0),
          (9.39, 0, 3.65), (9.39, 6.26, 0), (6.26, 10.43, 0), (9.39, 10.43, 2.61),
          (0, 6.78, 0), (3.65, 10.43, 0), (0, 10.43, 2.61), (0, 0, 5.22), (3.65, 0, 7.83),
          (0, 4.7, 7.83), (5.74, 0, 7.83), (9.39, 4.7, 7.83), (9.39, 0, 5.22),
          (9.39, 6.78, 7.83), (7.3, 10.43, 7.83), (9.39, 10.43, 5.22), (3.13, 10.43, 7.83),
          (0, 5.74, 7.83), (0, 10.43, 5.22)]
BOX = (9.39, 10.43, 7.83)

PUBLISHED = {1: (93, 41), 2: (91, 34), 3: (86, 47), 4: (81, 54), 5: (79, 60),
             6: (30, 2), 7: (58, 7), 8: (73, 12)}

HalfSpaces = List[Tuple[np.ndarray, float]]      # n . x <= b, n pointing out of the set


def hull_halfspaces(points) -> HalfSpaces:
    from scipy.spatial import ConvexHull
    pts = np.asarray(points, dtype=float)
    hull = ConvexHull(pts)
    out: HalfSpaces = []
    for eq in hull.equations:                       # n . x + d <= 0 inside
        n, d = eq[:3], eq[3]
        if not any(np.allclose(n, m, atol=1e-9) and abs(-d - b) < 1e-9 for m, b in out):
            out.append((n / np.linalg.norm(n), -d / np.linalg.norm(n)))
    return out


def cuboid(lo, hi) -> HalfSpaces:
    out: HalfSpaces = []
    for k in range(3):
        e = np.eye(3)[k]
        out += [(e, float(hi[k])), (-e, float(-lo[k]))]
    return out


def far_box() -> HalfSpaces:
    return cuboid((-FAR,) * 3, (FAR,) * 3)


def cone(apex, edge_points) -> HalfSpaces:
    """Convex cone with vertex `apex` whose edges pass through `edge_points`
    (in order around the cone), clipped to the far box."""
    a = np.asarray(apex, dtype=float)
    rays = [np.asarray(q, dtype=float) - a for q in edge_points]
    axis = sum(r / np.linalg.norm(r) for r in rays)
    out: HalfSpaces = []
    for r1, r2 in zip(rays, rays[1:] + rays[:1]):
        n = np.cross(r1, r2)
        if np.dot(n, axis) > 0:                     # outward: away from the axis
            n = -n
        n /= np.linalg.norm(n)
        out.append((n, float(n @ a)))
    return out + far_box()


def prism(axis: int, section) -> HalfSpaces:
    """Through-prism parallel to coordinate `axis` whose cross-section is the
    convex polygon `section`, given in the other two coordinates in order."""
    others = [k for k in range(3) if k != axis]
    S = np.asarray(section, dtype=float)
    c = S.mean(axis=0)
    out: HalfSpaces = []
    for p, q in zip(S, np.roll(S, -1, axis=0)):
        t = q - p
        n2 = np.array([t[1], -t[0]])
        if n2 @ (c - p) > 0:
            n2 = -n2
        n = np.zeros(3); n[others] = n2
        n /= np.linalg.norm(n)
        x = np.zeros(3); x[others] = p
        out.append((n, float(n @ x)))
    return out + far_box()


def dihedral(edge_point, edge_axis: int, facet_points) -> HalfSpaces:
    """Convex dihedral angle whose edge passes through `edge_point` parallel to
    coordinate `edge_axis` and whose two facets pass through `facet_points`."""
    e = np.eye(3)[edge_axis]
    v = np.asarray(edge_point, dtype=float)
    q1, q2 = (np.asarray(q, dtype=float) for q in facet_points)
    inside = v + 0.5 * ((q1 - v) + (q2 - v))
    out: HalfSpaces = []
    for q in (q1, q2):
        n = np.cross(e, q - v)
        if n @ (inside - v) > 0:
            n = -n
        n /= np.linalg.norm(n)
        out.append((n, float(n @ v)))
    return out + far_box()


def _basis(n):
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _section(H: HalfSpaces, n, b):
    """The convex polygon (shapely) where the plane n . x = b meets the set H,
    in the plane's (u, w) coordinates; None if empty or degenerate."""
    from shapely.geometry import Polygon, box
    u, w = _basis(n)
    o = n * b
    poly = box(-4 * FAR, -4 * FAR, 4 * FAR, 4 * FAR)
    for m, c in H:
        # m . (o + s u + t w) <= c  ->  (m.u) s + (m.w) t <= c - m.o
        a1, a2, rhs = m @ u, m @ w, c - m @ o
        if abs(a1) < 1e-14 and abs(a2) < 1e-14:
            if rhs < -1e-12:
                return None
            continue
        # clip by the half-plane: intersect with a huge polygon on its side
        L = 16 * FAR
        d = np.array([a1, a2]) / np.hypot(a1, a2)
        p0 = d * rhs / np.hypot(a1, a2)
        tvec = np.array([-d[1], d[0]])
        half = Polygon([p0 + L * tvec, p0 - L * tvec, p0 - L * tvec - L * d, p0 + L * tvec - L * d])
        poly = poly.intersection(half)
        if poly.is_empty:
            return None
    return poly if poly.area > MIN_AREA else None


def _faces_of(H: HalfSpaces):
    """(n, b) for every half-space of H that contributes a 2D face."""
    return [(n, b) for n, b in H if _section(H, n, b) is not None]


def _ring3d(coords, n, b) -> List[List[float]]:
    u, w = _basis(n)
    o = n * b
    pts = [np.round(o + s * u + t * w, 12) for s, t in list(coords)[:-1]]
    # Drop repeated and collinear vertices: a zero-length edge breaks the
    # face's edge geometry, and shapely leaves near-duplicates that rounding
    # merges. Repeat until stable, since removing one can expose another.
    changed = True
    while changed and len(pts) > 3:
        changed = False
        k = len(pts)
        for i in range(k):
            a, c, d = pts[i - 1], pts[i], pts[(i + 1) % k]
            if (np.linalg.norm(c - a) < 1e-9
                    or np.linalg.norm(np.cross(c - a, d - c)) < 1e-9 * max(1.0, np.linalg.norm(d - a))):
                pts.pop(i)
                changed = True
                break
    return [[float(x) for x in p] for p in pts]


def domain_faces(P: HalfSpaces, pieces: List[HalfSpaces]) -> List[Dict]:
    """Faces of cl(P minus the union of `pieces`), normals pointing into it."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    faces = []

    def without_holes(g):
        """Split a polygon with holes into hole-free pieces by cutting through
        each hole. The solver tests holed faces one point at a time, far more
        slowly than plain ones; the cut lines lie on the wall, so distances to
        the boundary, and with them every certified radius, are unchanged."""
        from shapely.geometry import LineString
        from shapely.ops import split
        if g.geom_type != "Polygon" or not g.interiors:
            return [g]
        cx = Polygon(g.interiors[0]).centroid.x
        line = LineString([(cx, -8 * FAR), (cx, 8 * FAR)])
        return [q for part in split(g, line).geoms for q in without_holes(part)]

    def emit(region, n_into, n_plane, b_plane, label):
        region = region.buffer(0)
        geoms = [q for g in getattr(region, "geoms", [region]) for q in without_holes(g)]
        for g in geoms:
            if g.geom_type != "Polygon" or g.area <= MIN_AREA:
                continue
            g = g.simplify(1e-10, preserve_topology=True)
            faces.append({
                "description": label,
                "vertices": _ring3d(g.exterior.coords, n_plane, b_plane),
                "holes": [_ring3d(r.coords, n_plane, b_plane) for r in g.interiors
                          if Polygon(r).area > MIN_AREA],
                "normal": [round(float(x), 15) for x in n_into],
            })

    # Faces of P, minus what the pieces remove from them.
    for i, (n, b) in enumerate(_faces_of(P)):
        region = _section(P, n, b)
        cuts = [s for K in pieces for s in [_section(K, n, b)] if s is not None]
        if cuts:
            region = region.difference(unary_union(cuts))
        emit(region, -n, n, b, f"P face {i}")

    # Faces of each piece inside P, minus the other pieces.
    for j, K in enumerate(pieces):
        for i, (n, b) in enumerate(_faces_of(K)):
            region = _section(K, n, b)
            inP = _section(P, n, b)
            if inP is None:
                continue
            region = region.intersection(inP)
            others = [s for k, K2 in enumerate(pieces) if k != j
                      for s in [_section(K2, n, b)] if s is not None]
            if others:
                region = region.difference(unary_union(others))
            emit(region, n, n, b, f"piece {j} face {i}")
    for f in faces:
        if not f["holes"]:
            del f["holes"]
    return faces


def example(k: int) -> Tuple[HalfSpaces, List[HalfSpaces]]:
    P = hull_halfspaces(TABLE1)
    C1 = cone((7, 7, 3), [(9.39, 3.5, 2), (9.39, 7.5, 2), (9.39, 7.5, 4), (9.39, 3.5, 4)])
    P1 = prism(1, [(2.5, 2.6), (3.5, 3.6), (4.5, 2.1)])                 # along y, (x, z)
    D1 = dihedral((0, 5.2, 5.83), 0, [(0, 3.7, 7.83), (0, 6.7, 7.83)])
    P2 = prism(0, [(3.5, 2.5), (4, 3.8), (4.5, 2.8), (4.5, 2.3)])      # along x, (y, z)
    rect = [(9.39, 1, 1), (9.39, 9.43, 1), (9.39, 9.43, 6.83), (9.39, 1, 6.83)]
    P0 = cuboid((0, 0, 0), BOX)
    return {1: (P, []), 2: (P, [C1]), 3: (P, [P1, C1]), 4: (P, [P1, C1, D1]),
            5: (P, [P1, P2, C1, D1]), 6: (P0, [cone((-10, 7, 3), rect)]),
            7: (P0, [cone((-1, 7, 3), rect)]), 8: (P0, [cone((4, 7, 3), rect)])}[k]


def write(k: int) -> str:
    P, pieces = example(k)
    spheres, minutes = PUBLISHED[k]
    spec = {"description": f"stoyan-yaskov-{k}",
            "note": (f"Stoyan & Yaskov (2013), ITOR 20:79-99, example {k}: {spheres} spheres "
                     f"of radius 1 in {minutes} min. Generated by "
                     f"tsgo.container.stoyan_yaskov; do not edit by hand."),
            "faces": domain_faces(P, pieces)}
    path = os.path.join(DATA_DIR, f"stoyan_yaskov_{k}.json")
    with open(path, "w") as fh:
        json.dump(spec, fh, indent=1)
    return path


def main() -> None:
    for k in range(1, 9):
        path = write(k)
        print(f"example {k}: {len(json.load(open(path))['faces'])} faces -> {path}")


if __name__ == "__main__":
    main()
