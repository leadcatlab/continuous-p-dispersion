import os

import numpy as np
from typing import Dict, List, Optional, Union

from tsgo.container.face import Face, EPS_PLANE
from tsgo.container.point import Point

_DEFAULT_RNG: Optional[np.random.Generator] = None


def _default_rng() -> np.random.Generator:
    """Fallback generator for callers that pass no rng. Not reproducible."""
    global _DEFAULT_RNG
    if _DEFAULT_RNG is None:
        _DEFAULT_RNG = np.random.default_rng()
    return _DEFAULT_RNG



# Entries per memoisation cache before it is emptied (see Container.contains).
CACHE_LIMIT = 100_000

# Answer contains() from the inside/outside grid where it decides (see
# Container._grid_lookup). Same switch as the batched energy: both are exact
# fast paths, so they are on or off together. Read at import so
# multiprocessing workers inherit it.
USE_GRID = os.environ.get("CPDP_BATCHED_ENERGY", "1") != "0"

class Container:
    def __init__(self, faces: List[Face], description: str = ""):
        self.faces = faces
        self.vertices = self._get_vertices()
        self.edges = self._get_edges()
        self.description = description

        # caches
        self.active_footpoints: Dict[tuple, List[Point]] = {}
        self._contains_cache: Dict[tuple, bool] = {}

        # deterministic ray direction (avoid random per call)
        d = np.array([1.0, 1.41421356237, 3.14159265359], dtype=float)
        self._ray_dir = d / np.linalg.norm(d)

        # Ray directions for point-in-polyhedron testing. Fixed by default;
        # set_ray_directions() draws a fresh set per run from the run's seed.
        self._ray_dirs = self._DEFAULT_RAY_DIRS

    # Eight fixed unit directions, chosen off-axis to avoid the degeneracies
    # (rays parallel to a face, or through an edge or vertex) that make a
    # parity count ambiguous.
    _DEFAULT_RAY_DIRS = np.asarray([
        [ 0.5773502691896258,  0.5773502691896258,  0.5773502691896258],
        [ 0.8506508083520399,  0.5257311121191336,  0.0000000000000000],
        [ 0.5257311121191336,  0.0000000000000000,  0.8506508083520399],
        [ 0.0000000000000000,  0.8506508083520399,  0.5257311121191336],
        [-0.8506508083520399,  0.5257311121191336,  0.0000000000000000],
        [ 0.5257311121191336,  0.0000000000000000, -0.8506508083520399],
        [ 0.0000000000000000, -0.8506508083520399,  0.5257311121191336],
        [-0.5773502691896258,  0.5773502691896258, -0.5773502691896258],
    ], dtype=float)

    def set_ray_directions(self, rng: Optional[np.random.Generator] = None,
                           k: int = 8) -> np.ndarray:
        """Draw k uniform directions on the sphere for containment testing.

        Called once per run so the direction set is fixed for the whole run --
        the energy stays deterministic in X, which is what L-BFGS requires --
        while differing between runs, so a degenerate alignment cannot bias
        every run the same way. Pass the run's Generator to keep it
        reproducible; pass None to restore the fixed default set.

        k need not scale with container complexity. For a query point at
        distance r from the nearest edge, the directions that resolve a
        crossing ambiguously occupy an angular band of width ~eps/r about a
        measure-zero arc, so a single ray misclassifies with probability
        p ~ c*|E|*eps/r and a k-ray majority vote fails with probability
        ~(4p)^(k/2). |E| enters only inside a logarithm, giving k = O(log|E|);
        at eps ~ 1e-10 and |E| ~ 200, k = 8 is already far past sufficient.
        The bound degrades as r -> 0, but that regime is handled by the
        on-boundary face test in contains(), not by adding rays.
        """
        if rng is None:
            self._ray_dirs = self._DEFAULT_RAY_DIRS
        else:
            v = rng.normal(size=(k, 3))
            self._ray_dirs = v / np.linalg.norm(v, axis=1, keepdims=True)
        self._contains_cache.clear()
        self.active_footpoints.clear()
        return self._ray_dirs

    def _get_vertices(self) -> List[Point]:
        vertices = set()
        for face in self.faces:
            for vertex in face.vertices:
                vertices.add(vertex)
        return list(vertices)

    def _get_edges(self) -> List[tuple[Point, Point]]:
        edges = set()
        for face in self.faces:
            for e in face.edges:
                edges.add(e)
        return list(edges)

    def contains(self, point: Union["Point", np.ndarray]) -> bool:
        # Tolerances
        TOL = 1e-10          # for ray-plane denom / t comparisons
        T_DEDUP = 1e-9       # for t dedup (a bit looser helps with seams)

        # Normalize input to Point
        if isinstance(point, (list, tuple)):
            point = Point(float(point[0]), float(point[1]), float(point[2]))
        elif isinstance(point, np.ndarray):
            point = Point.from_array(point)

        # Exact coordinates, not rounded ones. A key rounded to 6 decimals
        # merged a point 5e-7 outside a wall with one just inside it, so the
        # outside point inherited "inside" -- and with it an energy of zero,
        # since its own wall then drops out of the active set. The optimiser
        # found that pocket: 201 of the 250 infeasible seeds in the 2026-09-28
        # sweep ended with one point parked in it. Any tolerance belongs in
        # the geometry (face.EPS_PLANE), where every caller sees the same rule.
        key = (point.x, point.y, point.z)
        hit = self._contains_cache.get(key)
        if hit is not None:
            return hit
        # Bounded: with exact keys a long run never repeats most of them, and
        # an unbounded cache grew ~2.4 GB an hour on p = 100 runs -- eleven of
        # them exhausted 32 GB and froze the machine. It only memoises an
        # exact answer, so emptying it can cost time but never change a result.
        if len(self._contains_cache) >= CACHE_LIMIT:
            self._contains_cache.clear()

        P = point.as_array().astype(float)

        if USE_GRID:
            v = self._grid_lookup(P[None, :])[0]
            if v >= 0:
                self._contains_cache[key] = bool(v)
                return bool(v)

        from tsgo.container.face_batch import faces_contain, face_arrays
        fa = face_arrays(self)
        F = fa["F"]

        # Boundary => inside. One batched test against every face rather than
        # a Python loop calling Face.contains F times.
        if F and faces_contain(self, np.broadcast_to(P, (F, 3))).any():
            self._contains_cache[key] = True
            return True

        RAY_DIRS = self._ray_dirs
        N = fa["N"]
        P0 = fa["P0"]

        def _parity_for_dirs(dirs: np.ndarray) -> np.ndarray:
            """Parity of each ray direction, as an object array of
            True / False / None (None = degenerate, ignore that direction).

            All directions are intersected against all faces in one pass: the
            per-call numpy cost is fixed, so casting rays one at a time left
            the small containers slower than the scalar loop this replaces.
            """
            dirs = np.atleast_2d(np.asarray(dirs, dtype=float))
            norms = np.linalg.norm(dirs, axis=1)
            res = np.empty(dirs.shape[0], dtype=object)
            res[:] = None
            good = norms > 0.0
            if F == 0 or not good.any():
                res[good] = False
                return res

            Dg = dirs[good] / norms[good, None]              # (Rg, 3)
            denom = N @ Dg.T                                 # (F, Rg)
            num = np.einsum("fj,fj->f", N, P0 - P[None, :])  # (F,)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = num[:, None] / denom
            ok = (np.abs(denom) > TOL) & np.isfinite(t) & (t > TOL)
            # Zero the parallel/behind entries before forming hit points: they
            # are masked out anyway, but inf * d would propagate NaN.
            t = np.where(ok, t, 0.0)
            hits = P[None, None, :] + t[:, :, None] * Dg[None, :, :]   # (F, Rg, 3)
            hit_in = faces_contain(self, hits, mask=ok) & ok

            parity = np.zeros(Dg.shape[0], dtype=bool)
            for r in range(Dg.shape[0]):
                ts = np.sort(t[:, r][hit_in[:, r]])
                if ts.size == 0:
                    parity[r] = False        # 0 intersections: outside, not degenerate
                    continue
                # Dedup intersections at nearly the same t (shared edges /
                # seams), same relative+absolute rule as before.
                kept = 1
                last = ts[0]
                for k in range(1, ts.size):
                    if abs(ts[k] - last) > (T_DEDUP * max(1.0, abs(ts[k]), abs(last))):
                        kept += 1
                        last = ts[k]
                parity[r] = (kept % 2) == 1

            res[np.nonzero(good)[0]] = parity
            return res

        votes = _parity_for_dirs(RAY_DIRS)
        used = int(sum(1 for v in votes if v is not None))
        votes_true = int(sum(1 for v in votes if v is True))
        votes_false = used - votes_true

        # If all rays were weird (rare), do deterministic jitter fallback
        if used == 0:
            base = np.asarray(self._ray_dir, dtype=float)
            jitter = np.asarray([base + (k + 1) * 1e-7 * np.array([1.0, 2.0, 3.0])
                                 for k in range(8)], dtype=float)
            for v in _parity_for_dirs(jitter):
                if v is not None:
                    self._contains_cache[key] = bool(v)
                    return bool(v)

            # last resort
            self._contains_cache[key] = False
            return False

        inside = votes_true > votes_false
        self._contains_cache[key] = inside
        return inside

    def contains_many(self, X: np.ndarray) -> np.ndarray:
        """contains() for every row of X: the grid where it decides, rays elsewhere."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not USE_GRID:
            return self._contains_many_raycast(X)
        v = self._grid_lookup(X)
        out = v == 1
        rest = np.nonzero(v < 0)[0]
        if rest.size:
            out[rest] = self._contains_many_raycast(X[rest])
        return out

    # ------------------------------------------------------------------
    # Inside/outside grid
    # ------------------------------------------------------------------
    # Cells along the longest side of the (padded) bounding box. 128 gives
    # 2 MB of int8 state at most; cells are classified only when first visited.
    GRID_CELLS = 128
    # How far a cell's centre must clear the boundary beyond its half-diagonal.
    # contains() calls a point within EPS_PLANE (1e-8) of a face inside, so the
    # margin must exceed that tolerance band by a wide berth.
    GRID_MARGIN = 1e-6

    def _grid(self) -> dict:
        g = getattr(self, "_grid_state", None)
        if g is None:
            bb = self._bounding_box()
            lo = np.array([bb["min_x"], bb["min_y"], bb["min_z"]], dtype=float)
            hi = np.array([bb["max_x"], bb["max_y"], bb["max_z"]], dtype=float)
            pad = 0.05 * float((hi - lo).max())
            lo, hi = lo - pad, hi + pad
            h = float((hi - lo).max()) / self.GRID_CELLS
            n = np.maximum(1, np.ceil((hi - lo) / h).astype(np.int64))
            g = dict(lo=lo, h=h, n=n, half_diag=0.5 * h * np.sqrt(3.0),
                     state=np.full(tuple(n), -1, dtype=np.int8))
            self._grid_state = g
        return g

    def _grid_lookup(self, X: np.ndarray) -> np.ndarray:
        """Per row of X: 1 or 0 where the point's whole grid cell lies inside
        or outside the container, -1 where only ray casting can tell.

        A cell is decided when its centre is further from the boundary than
        the cell's half-diagonal plus GRID_MARGIN: then no point of the cell is
        on, or within contains()'s tolerance of, the boundary, so every point
        of it gets the verdict of the centre. The centre's verdict is the ray
        cast's; away from the boundary that is the truth, so the grid returns
        what contains() would. Feasible points are at least D from the
        boundary, which is why nearly every query lands in a decided cell.

        State per cell: -1 not yet visited, 0 outside, 1 inside, 2 too close
        to the boundary to decide.
        """
        g = self._grid()
        out = np.full(X.shape[0], -1, dtype=np.int8)
        f = (X - g["lo"]) / g["h"]
        ok = np.all(np.isfinite(f) & (f >= 0.0) & (f < g["n"]), axis=1)
        if not ok.any():
            return out
        idx = f[ok].astype(np.int64)
        st = g["state"]
        s = st[idx[:, 0], idx[:, 1], idx[:, 2]]
        new = s == -1
        if new.any():
            cells = np.unique(idx[new], axis=0)
            centres = g["lo"] + (cells + 0.5) * g["h"]
            clear = self._boundary_distance(centres)
            decided = clear > g["half_diag"] + self.GRID_MARGIN
            verdict = np.full(cells.shape[0], 2, dtype=np.int8)
            if decided.any():
                verdict[decided] = self._contains_many_raycast(centres[decided]).astype(np.int8)
            st[cells[:, 0], cells[:, 1], cells[:, 2]] = verdict
            s = st[idx[:, 0], idx[:, 1], idx[:, 2]]
        out[ok] = np.where(s <= 1, s, -1)
        return out

    def _boundary_distance(self, Y: np.ndarray) -> np.ndarray:
        """Euclidean distance from each row of Y to the container's surface.

        The nearest point of a planar polygon is either the orthogonal foot,
        when that foot lies on the polygon, or a point of one of its edges --
        outer or hole -- so faces' feet and all edges cover every case.
        """
        from tsgo.container.face_batch import faces_contain, face_arrays
        from tsgo.container.norms import segment_distance
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        n = Y.shape[0]
        best = np.full(n, np.inf)
        fa = face_arrays(self)
        F = fa["F"]
        if F:
            N, P0 = fa["N"], fa["P0"]
            hs = np.einsum("fj,nfj->nf", N, Y[:, None, :] - P0[None, :, :])   # (n, F)
            feet = Y[:, None, :] - hs[:, :, None] * N[None, :, :]
            on = faces_contain(self, feet.transpose(1, 0, 2)).T
            best = np.minimum(best, np.where(on, np.abs(hs), np.inf).min(axis=1))
        if self.edges:
            ES = np.asarray([e[0].as_array() for e in self.edges], dtype=float)
            EE = np.asarray([e[1].as_array() for e in self.edges], dtype=float)
            E = ES.shape[0]
            d, _ = segment_distance(np.repeat(Y, E, axis=0), np.tile(ES, (n, 1)),
                                    np.tile(EE, (n, 1)), "EUCLIDEAN")
            best = np.minimum(best, d.reshape(n, E).min(axis=1))
        return best

    def _contains_many_raycast(self, X: np.ndarray) -> np.ndarray:
        """contains() for every row of X, in one pass over all points at once.

        Same rules as contains() -- on a face counts as inside, otherwise a
        majority vote of the run's ray directions, with intersections at nearly
        the same t merged by the same sequential rule -- but the rays of every
        point are cast against every face together. Per point, contains()
        spends most of its time in fixed numpy call overhead; batched, that is
        paid once per configuration instead of once per point. Not memoised:
        the exact-coordinate cache rarely hits during an optimisation.
        """
        TOL = 1e-10
        T_DEDUP = 1e-9

        X = np.atleast_2d(np.asarray(X, dtype=float))
        P = X.shape[0]
        out = np.zeros(P, dtype=bool)

        from tsgo.container.face_batch import faces_contain, face_arrays
        fa = face_arrays(self)
        F, N, P0 = fa["F"], fa["N"], fa["P0"]
        if F == 0 or P == 0:
            return out

        # Boundary => inside, as in contains().
        out |= faces_contain(self, np.broadcast_to(X[None, :, :], (F, P, 3))).any(axis=0)
        rest = np.nonzero(~out)[0]
        if rest.size == 0:
            return out

        dirs = np.atleast_2d(np.asarray(self._ray_dirs, dtype=float))
        norms = np.linalg.norm(dirs, axis=1)
        good = norms > 0.0
        if not good.any():
            # Degenerate direction set: defer to the scalar routine's fallback.
            out[rest] = [self.contains(X[i]) for i in rest]
            return out
        Dg = dirs[good] / norms[good, None]                          # (R, 3)
        R = Dg.shape[0]
        Xr = X[rest]
        Pr = Xr.shape[0]

        denom = N @ Dg.T                                             # (F, R)
        num = np.einsum("fj,fpj->fp", N, P0[:, None, :] - Xr[None, :, :])   # (F, Pr)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num[:, :, None] / denom[:, None, :]                  # (F, Pr, R)
        ok = (np.abs(denom)[:, None, :] > TOL) & np.isfinite(t) & (t > TOL)
        t = np.where(ok, t, 0.0)
        hits = Xr[None, :, None, :] + t[..., None] * Dg[None, None, :, :]  # (F, Pr, R, 3)

        ok = ok.reshape(F, Pr * R)
        t = t.reshape(F, Pr * R)
        hit_in = faces_contain(self, hits.reshape(F, Pr * R, 3), mask=ok) & ok

        # Count distinct crossings per ray. Sorted along the face axis with
        # misses at +inf; a crossing counts when it is further than T_DEDUP
        # (relative) from the last one counted -- contains()'s rule exactly.
        ts = np.sort(np.where(hit_in, t, np.inf), axis=0)            # (F, Pr*R)
        last = ts[0].copy()
        kept = np.isfinite(last).astype(int)
        for k in range(1, F):
            cur = ts[k]
            fin = np.isfinite(cur)
            if not fin.any():
                break                                                # rest are +inf too
            with np.errstate(invalid="ignore"):
                new = fin & (np.abs(cur - last) >
                             T_DEDUP * np.maximum(1.0, np.maximum(np.abs(cur), np.abs(last))))
            kept += new
            last = np.where(new, cur, last)

        parity = (kept % 2 == 1).reshape(Pr, R)
        votes_true = parity.sum(axis=1)
        out[rest] = votes_true > (R - votes_true)
        return out


    def _is_active_plane(self, face: Face, point: Point, inside: bool) -> bool:
        # Active Face, per the model: F is active w.r.t. c if either c is inside
        # and N_F . (c - P0) >= 0, or c is outside and N_F . (c - P0) < 0.
        #
        # The inside test must be inclusive, and inclusive to the *same
        # tolerance* that decided `inside`. A point lying exactly on a face has
        # half_space == 0 there, and a strict > dropped that face from its own
        # active set, so the point's boundary penalty vanished and E read 0 -- a
        # point sitting on the container wall scored as perfectly feasible.
        # Exact equality is not enough either: contains() calls a point inside
        # when it is within EPS_PLANE of a face, so a point a few 1e-9 *outside*
        # a wall is inside by that test while its own wall has hs < 0. Testing
        # hs >= -EPS_PLANE closes the gap between the two rules.
        #
        # Only the inside branch is widened. A dropped face there removes a term
        # from a penalty whose floor is zero, which manufactures feasibility; on
        # the outside branch the penalty is bounded below by D^2 > 0, so a
        # dropped face can only over-penalise, which is safe.
        hs = face.half_space(point)
        if inside and hs >= -EPS_PLANE:
            return True
        if (not inside) and hs < 0:
            return True
        return False

    def get_active_footpoints(self, point, metric: str = "EUCLIDEAN") -> List[Point]:
        if isinstance(point, list):
            point = Point(float(point[0]), float(point[1]), float(point[2]))
        elif isinstance(point, np.ndarray):
            point = Point(float(point[0]), float(point[1]), float(point[2]))

        # The metric belongs in the key: the foot of a point on a face depends
        # on it, so a cache shared across metrics would return the wrong one.
        # Exact coordinates for the same reason as contains(): a rounded key
        # hands a point the feet, and the inside/outside verdict, of a
        # neighbour on the other side of a wall.
        key = (point.x, point.y, point.z, metric.upper())
        cached = self.active_footpoints.get(key)
        if cached is not None:
            return cached
        if len(self.active_footpoints) >= CACHE_LIMIT:     # bounded; see contains()
            self.active_footpoints.clear()

        inside = self.contains(point)

        from tsgo.container.face_batch import face_arrays, active_feet
        fa = face_arrays(self)
        F = fa["F"]
        if F == 0:
            self.active_footpoints[key] = []
            return []

        P = point.as_array().astype(float)

        # Active Face, per the model: F is active w.r.t. c if either c is
        # inside and N_F . (c - P0) >= 0, or c is outside and N_F . (c - P0)
        # < 0. Evaluated for every face at once; _is_active_plane remains the
        # single-face statement of the same rule.
        hs = np.einsum("fj,fj->f", fa["N"], P[None, :] - fa["P0"])
        active = (hs >= -EPS_PLANE) if inside else (hs < 0.0)

        # Metric-nearest point of each active face's plane, kept when it lands
        # on the face. Shared with the analytic-gradient path so the two cannot
        # drift apart again; see face_batch.active_feet for why the nearest
        # point of a plane needs a search rather than an argmax.
        feet, keep, _ = active_feet(self, P, active, metric)

        out: List[Point] = [Point(float(feet[i, 0]), float(feet[i, 1]), float(feet[i, 2]))
                            for i in np.nonzero(keep)[0]]

        self.active_footpoints[key] = out
        return out

    def _bounding_box(self) -> Dict[str, float]:
        min_x = min(v.x for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        min_z = min(v.z for v in self.vertices)
        max_z = max(v.z for v in self.vertices)
        return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "min_z": min_z, "max_z": max_z}

    def _sample_random_point(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Rejection-sample a point inside the container.

        Pass an explicit Generator for reproducibility; omitting it draws from a
        module-level generator seeded from OS entropy, which is not reproducible.
        """
        if rng is None:
            rng = _default_rng()
        bb = self._bounding_box()
        lo = (bb["min_x"], bb["min_y"], bb["min_z"])
        hi = (bb["max_x"], bb["max_y"], bb["max_z"])
        # Iterative rather than recursive: a thin container can need many draws,
        # and the recursive form risked a RecursionError.
        while True:
            x, y, z = (float(rng.uniform(lo[k], hi[k])) for k in range(3))
            if self.contains(Point(x, y, z)):
                return np.array([x, y, z], dtype=float)

    def generate_initial_solution(self, p: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = _default_rng()
        return np.array([self._sample_random_point(rng) for _ in range(p)], dtype=float)
    
    @property
    def volume(self) -> float:
        # using divergence theorem: V = (1/3) * sum(face_area * face_normal . face_centroid)
        volume = 0.0
        for face in self.faces:
            area = face.area
            normal = np.asarray(face.normal, dtype=float)
            centroid = np.mean([v.as_array() for v in face.vertices], axis=0)
            volume += area * np.dot(normal, centroid)
        return abs(volume) / 3.0

