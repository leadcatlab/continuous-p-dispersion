import numpy as np
from typing import Dict, List, Union

from tsgo.container.face import Face
from tsgo.container.point import Point


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

        key = (round(point.x, 6), round(point.y, 6), round(point.z, 6))
        hit = self._contains_cache.get(key)
        if hit is not None:
            return hit

        P = point.as_array().astype(float)

        # Boundary => inside
        for face in self.faces:
            if face.contains(point):
                self._contains_cache[key] = True
                return True

        # Deterministic set of ray directions (avoid axis-aligned rays; reduce degeneracy)
        RAY_DIRS = np.asarray([
            [ 0.5773502691896258,  0.5773502691896258,  0.5773502691896258],
            [ 0.8506508083520399,  0.5257311121191336,  0.0000000000000000],
            [ 0.5257311121191336,  0.0000000000000000,  0.8506508083520399],
            [ 0.0000000000000000,  0.8506508083520399,  0.5257311121191336],
            [-0.8506508083520399,  0.5257311121191336,  0.0000000000000000],
            [ 0.5257311121191336,  0.0000000000000000, -0.8506508083520399],
            [ 0.0000000000000000, -0.8506508083520399,  0.5257311121191336],
            [-0.5773502691896258,  0.5773502691896258, -0.5773502691896258],
        ], dtype=float)

        def _single_ray_parity(d: np.ndarray) -> Union[bool, None]:
            """
            Returns:
            True/False => inside/outside
            None => degenerate ray (too parallel / too ambiguous), ignore this direction
            """
            # ensure unit direction
            dn = np.linalg.norm(d)
            if dn == 0:
                return None
            d = d / dn

            ts = []
            for face in self.faces:
                nvec = np.asarray(face.normal, dtype=float)
                P0 = face.vertices[0].as_array().astype(float)

                denom = float(np.dot(nvec, d))
                if abs(denom) <= TOL:
                    continue

                t = float(np.dot(nvec, (P0 - P)) / denom)
                if t <= TOL:
                    continue

                hit3 = P + t * d

                # IMPORTANT: call face.contains with a Point for consistency
                hit_pt = Point(float(hit3[0]), float(hit3[1]), float(hit3[2]))
                if face.contains(hit_pt):
                    ts.append(t)

            if not ts:
                # This direction says "outside" (0 intersections); not degenerate.
                return False

            ts.sort()

            # Dedup intersections at nearly the same t (shared edges / seams)
            unique_ts = []
            last = None
            for t in ts:
                if last is None:
                    unique_ts.append(t)
                    last = t
                    continue

                # relative + absolute tolerance
                if abs(t - last) > (T_DEDUP * max(1.0, abs(t), abs(last))):
                    unique_ts.append(t)
                    last = t

            return (len(unique_ts) % 2) == 1

        # Majority vote across ray directions
        votes_true = 0
        votes_false = 0
        used = 0

        for d in RAY_DIRS:
            res = _single_ray_parity(d)
            if res is None:
                continue
            used += 1
            if res:
                votes_true += 1
            else:
                votes_false += 1

            # early exit once majority is impossible to overturn
            if used >= 5:
                if votes_true >= 3:
                    self._contains_cache[key] = True
                    return True
                if votes_false >= 3:
                    self._contains_cache[key] = False
                    return False

        # If all rays were weird (rare), do deterministic jitter fallback
        if used == 0:
            base = np.asarray(self._ray_dir, dtype=float)
            for k in range(8):
                d = base + (k + 1) * 1e-7 * np.array([1.0, 2.0, 3.0])
                res = _single_ray_parity(d)
                if res is not None:
                    self._contains_cache[key] = res
                    return res

            # last resort
            self._contains_cache[key] = False
            return False

        inside = votes_true > votes_false
        self._contains_cache[key] = inside
        return inside


    def _is_active_plane(self, face: Face, point: Point, inside: bool) -> bool:
        # Same logic as your old version, but WITHOUT calling contains repeatedly
        hs = face.half_space(point)
        if inside and hs > 0:
            return True
        if (not inside) and hs < 0:
            return True
        return False

    def get_active_footpoints(self, point) -> List[Point]:
        if isinstance(point, list):
            point = Point(float(point[0]), float(point[1]), float(point[2]))
        elif isinstance(point, np.ndarray):
            point = Point(float(point[0]), float(point[1]), float(point[2]))

        key = (round(point.x, 6), round(point.y, 6), round(point.z, 6))
        cached = self.active_footpoints.get(key)
        if cached is not None:
            return cached

        inside = self.contains(point)

        out: List[Point] = []
        for face in self.faces:
            if not self._is_active_plane(face, point, inside):
                continue

            fp = face.project_point(point)
            if face.contains(fp):
                out.append(fp)

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

    def _sample_random_point(self) -> np.ndarray:
        bb = self._bounding_box()
        x = np.random.uniform(bb["min_x"], bb["max_x"])
        y = np.random.uniform(bb["min_y"], bb["max_y"])
        z = np.random.uniform(bb["min_z"], bb["max_z"])
        pt = Point(x, y, z)
        return np.array([x, y, z], dtype=float) if self.contains(pt) else self._sample_random_point()

    def generate_initial_solution(self, p: int) -> np.ndarray:
        return np.array([self._sample_random_point() for _ in range(p)], dtype=float)
    
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

