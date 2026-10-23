#!/usr/bin/env python3
"""Check a reported packing from its centres alone.

    python verify_packing.py configurations/stoyan-yaskov-8/euclidean/p80.csv
    python verify_packing.py configurations            # every file below a directory

Needs only numpy, and shares no code with the solver. Each configuration file
names its container (a face list in containers/), its metric and the radius
claimed for it, then lists the centres. For each file this script computes,
from the geometry alone, the largest radius r at which the configuration is
feasible,

    r = min( min_{i<j} ||c_i - c_j|| / 2 ,  min_i dist(c_i, boundary) ),

with every centre required to lie inside the container, and reports it beside
the claimed radius. The norm is l2, l1 or l-inf as the file says; balls of
radius r in that norm then have disjoint interiors and lie in the container.

Containers are closed polyhedral domains given by their boundary faces: planar
polygons, possibly non-convex and with holes, each with a unit normal pointing
into the domain. The distance from a point to a face is exact in all three
norms: the nearest point lies either inside the face, where the norm's ball
first touches the face's plane, or on one of the face's edges, and both cases
are computed in closed form. Inside-ness is a ray-crossing parity test, taken
along five directions that must all agree.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ZERO = 1e-12
EDGE_TOL = 1e-12
# Generic directions: no two components equal in size or in a simple ratio, so
# a ray from a symmetric centre does not run along a container's edge.
RAYS = np.array([[0.5463, 0.6158, 0.5677], [-0.6829, 0.2914, 0.6697], [0.2381, -0.8163, 0.5263],
                 [-0.3907, -0.5412, -0.7449], [0.8513, -0.2246, -0.4741]])
RAYS = RAYS / np.linalg.norm(RAYS, axis=1, keepdims=True)


# ---------------------------------------------------------------- geometry ---

class Face:
    def __init__(self, spec):
        self.rings = [np.asarray(spec["vertices"], float)] + \
                     [np.asarray(h, float) for h in spec.get("holes", [])]
        n = np.asarray(spec["normal"], float)
        self.n = n / np.linalg.norm(n)                  # into the domain
        self.o = self.rings[0][0]
        a = self.rings[0][1] - self.rings[0][0]
        self.a = a / np.linalg.norm(a)
        self.b = np.cross(self.n, self.a)
        self.rings2 = [self.to2(R) for R in self.rings]
        self.edges = np.concatenate([np.stack([R, np.roll(R, -1, axis=0)], axis=1)
                                     for R in self.rings])          # (m, 2, 3)

    def to2(self, P):
        P = np.asarray(P, float) - self.o
        return np.stack([P @ self.a, P @ self.b], axis=-1)


def load_container(path):
    return [Face(f) for f in json.load(open(path))["faces"]]


def on_segment2(q, p0, p1):
    """q (k, 2) on the segment p0-p1."""
    d = p1 - p0
    L = np.hypot(*d)
    if L <= ZERO:                                      # a point, not a segment
        return np.hypot(*(q - p0).T) <= EDGE_TOL
    cross = (d[0] * (q[:, 1] - p0[1]) - d[1] * (q[:, 0] - p0[0])) / max(L, ZERO)
    t = ((q - p0) @ d) / max(L * L, ZERO)
    return (np.abs(cross) <= EDGE_TOL * max(1.0, L)) & (t >= -EDGE_TOL) & (t <= 1 + EDGE_TOL)


def in_polygon2(q, rings2):
    """Points q (k, 2) in the closed polygon with these rings (even-odd)."""
    q = np.atleast_2d(q)
    inside = np.zeros(len(q), bool)
    on = np.zeros(len(q), bool)
    for R in rings2:
        for p0, p1 in zip(R, np.roll(R, -1, axis=0)):
            on |= on_segment2(q, p0, p1)
            crosses = (p0[1] > q[:, 1]) != (p1[1] > q[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                x = p0[0] + (q[:, 1] - p0[1]) * (p1[0] - p0[0]) / (p1[1] - p0[1])
            inside ^= crosses & (q[:, 0] < x)
    return inside | on


def segments_meet2(p0, p1, q0, q1):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1, o2 = orient(p0, p1, q0), orient(p0, p1, q1)
    o3, o4 = orient(q0, q1, p0), orient(q0, q1, p1)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and min(map(abs, (o1, o2, o3, o4))) > 0:
        return True
    pts = np.array([q0, q1])
    return bool(on_segment2(pts, p0, p1).any() or on_segment2(np.array([p0, p1]), q0, q1).any())


def convex_meets_polygon(C2, rings2):
    """Does the convex set with vertices C2 (1 to 4, in order) meet the polygon?"""
    if in_polygon2(C2, rings2).any():
        return True
    if len(C2) == 1:
        return False
    V = np.concatenate(rings2)
    if len(C2) == 2:
        if on_segment2(V, C2[0], C2[1]).any():
            return True
    else:
        if in_polygon2(V, [C2]).any():
            return True
    cedges = list(zip(C2, np.roll(C2, -1, axis=0)))[: (1 if len(C2) == 2 else len(C2))]
    for R in rings2:
        for q0, q1 in zip(R, np.roll(R, -1, axis=0)):
            if any(segments_meet2(p0, p1, q0, q1) for p0, p1 in cedges):
                return True
    return False


# ------------------------------------------------------------------- norms ---

def norm(v, metric):
    if metric == "EUCLIDEAN":
        return np.sqrt((v * v).sum(-1))
    if metric == "TAXICAB":
        return np.abs(v).sum(-1)
    if metric == "SUP_NORM":
        return np.abs(v).max(-1)
    raise ValueError(metric)


def dual_norm(n, metric):
    return {"EUCLIDEAN": np.linalg.norm(n), "TAXICAB": np.abs(n).max(),
            "SUP_NORM": np.abs(n).sum()}[metric]


def exposed_face(n, metric):
    """Vertices, in order, of the face of the unit ball that maximises n . v."""
    if metric == "EUCLIDEAN":
        return np.array([n / np.linalg.norm(n)])
    if metric == "SUP_NORM":
        free = [i for i in range(3) if abs(n[i]) <= ZERO]
        base = np.sign(n) * (np.abs(n) > ZERO)
        if not free:
            return np.array([base])
        if len(free) == 1:
            out = []
            for s in (-1, 1):
                v = base.copy(); v[free[0]] = s; out.append(v)
            return np.array(out)
        i, j = free
        out = []
        for si, sj in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            v = base.copy(); v[i] = si; v[j] = sj; out.append(v)
        return np.array(out)
    if metric == "TAXICAB":
        m = np.abs(n).max()
        return np.array([np.sign(n[i]) * np.eye(3)[i] for i in range(3) if np.abs(n[i]) >= m - ZERO])
    raise ValueError(metric)


def point_segment(C, a, b, metric):
    """Distance in the norm from each point of C (k, 3) to the segment a-b."""
    d = b - a
    E = a - C                                          # g(s) = E + s d
    if metric == "EUCLIDEAN":
        s = np.clip(-(E @ d) / max(d @ d, ZERO), 0, 1)
        return norm(E + s[:, None] * d, metric)
    cands = [np.zeros(len(C)), np.ones(len(C))]
    nz = [i for i in range(3) if abs(d[i]) > ZERO]
    for i in nz:                                       # a coordinate of g crosses zero
        cands.append(-E[:, i] / d[i])
    if metric == "SUP_NORM":                           # two coordinates of |g| meet
        for i in range(3):
            for j in range(i + 1, 3):
                for sg in (1, -1):
                    den = d[i] - sg * d[j]
                    if abs(den) > ZERO:
                        cands.append(-(E[:, i] - sg * E[:, j]) / den)
    S = np.clip(np.stack(cands, axis=1), 0, 1)        # (k, m)
    G = E[:, None, :] + S[..., None] * d
    return norm(G, metric).min(axis=1)


def face_distance(C, f, metric):
    """Exact distance in the norm from each point of C to the closed face f."""
    h = (C - f.o) @ f.n                                # signed, + into the domain
    plane = np.abs(h) / dual_norm(f.n, metric)
    # Where the norm ball first touches the plane: c - sign(h) t v, v on the
    # face of the unit ball exposed by n.
    F = exposed_face(f.n, metric)
    touch = np.zeros(len(C), bool)
    if len(F) == 1:
        Q = C - (np.sign(h) * plane)[:, None] * F[0]
        touch = in_polygon2(f.to2(Q), f.rings2)
    else:
        for k, c in enumerate(C):
            Q = c - np.sign(h[k]) * plane[k] * F
            # A centre on the plane shrinks the contact set to the centre.
            Q2 = f.to2(Q[:1] if plane[k] <= ZERO else Q)
            touch[k] = convex_meets_polygon(Q2, f.rings2)
    edge = np.full(len(C), np.inf)
    for a, b in f.edges:
        edge = np.minimum(edge, point_segment(C, a, b, metric))
    return np.where(touch, plane, edge)


def inside(C, faces):
    """Ray-crossing parity along each of RAYS; all must agree."""
    votes = []
    for u in RAYS:
        k = np.zeros(len(C), int)
        for f in faces:
            nu = f.n @ u
            if abs(nu) < ZERO:
                continue
            t = ((f.o - C) @ f.n) / nu
            Q = C + t[:, None] * u
            k += (t > 0) & in_polygon2(f.to2(Q), f.rings2)
        votes.append(k % 2 == 1)
    votes = np.array(votes)
    if not (votes.all(axis=0) | ~votes.any(axis=0)).all():
        raise RuntimeError("rays disagree on inside-ness: a centre lies on the boundary")
    return votes[0]


def certified_radius(C, faces, metric):
    C = np.asarray(C, float)
    ins = inside(C, faces)
    pair = np.inf
    if len(C) > 1:
        i, j = np.triu_indices(len(C), 1)
        pair = 0.5 * norm(C[i] - C[j], metric).min()
    wall = min(face_distance(C, f, metric).min() for f in faces)
    r = 0.0 if not ins.all() else min(pair, wall)
    return r, pair, wall, int((~ins).sum())


# -------------------------------------------------------------------- files ---

def read_configuration(path):
    meta, rows = {}, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if ":" in line:
                    k, v = line[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
            else:
                rows.append(line)
    C = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in csv.DictReader(rows)])
    return meta, C


def check(path):
    meta, C = read_configuration(path)
    geom = os.path.join(os.path.dirname(os.path.abspath(path)), meta["geometry"]) \
        if not os.path.isabs(meta["geometry"]) else meta["geometry"]
    faces = load_container(os.path.normpath(geom))
    r, pair, wall, outside = certified_radius(C, faces, meta["metric"])
    claimed = float(meta["radius"])
    return {"file": path, "container": meta["container"], "metric": meta["metric"], "p": len(C),
            "claimed": claimed, "verified": r, "half_min_pair": pair, "min_wall": wall,
            "outside": outside, "ok": r >= claimed * (1 - 1e-9)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="Configuration files, or directories to search.")
    ap.add_argument("--csv", help="Also write the results to this CSV.")
    args = ap.parse_args()
    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs if f.endswith(".csv"))
        else:
            files.append(p)
    out, bad = [], 0
    for f in files:
        r = check(f)
        out.append(r)
        bad += not r["ok"]
        print(f"{'ok  ' if r['ok'] else 'FAIL'} {r['container']:<16} {r['metric']:<9} p={r['p']:<4} "
              f"claimed {r['claimed']:.12f}  verified {r['verified']:.12f}"
              + (f"  ({r['outside']} centres outside)" if r["outside"] else ""))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader(); w.writerows(out)
    print(f"{len(out) - bad} of {len(out)} configurations verified at their claimed radius"
          " (to a relative 1e-9)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
