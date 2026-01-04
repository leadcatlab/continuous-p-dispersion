import numpy as np
from typing import List, Union

from tsgo.container.point import Point


class Face:
    def __init__(self, vertices: List[Point], normal: np.ndarray, description: str = "", holes: List[List[Point]] = None):
        if holes is None:
            holes = []
        self.vertices = vertices
        self.holes = holes
        self.normal = np.array(normal, dtype=float)
        self._adjust_normal()
        self.description = description
        self.edges = self._get_edges()

        # Cached geometry for fast contains()
        self._P0 = self.vertices[0].as_array().astype(float)
        self._build_2d_cache()

    def _adjust_normal(self):
        v1 = self.vertices[1].as_array() - self.vertices[0].as_array()
        v2 = self.vertices[2].as_array() - self.vertices[0].as_array()
        computed_normal = np.cross(v1, v2).astype(float)
        norm = np.linalg.norm(computed_normal)
        if norm > 0:
            computed_normal /= norm
        if np.dot(computed_normal, self.normal) < 0:
            computed_normal = -computed_normal
        self.normal = computed_normal

    def _get_edges(self) -> List[tuple[Point, Point]]:
        edges = []
        for i in range(len(self.vertices)):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % len(self.vertices)]
            edges.append((a, b))
        for hole in self.holes:
            for i in range(len(hole)):
                a = hole[i]
                b = hole[(i + 1) % len(hole)]
                edges.append((a, b))
        return edges

    def half_space(self, point: Point) -> float:
        point_vector = point.as_array()
        return float(np.dot(self.normal, point_vector - self._P0))

    def project_point(self, point: Point) -> Point:
        d = self.half_space(point)
        return Point.from_array(point.as_array() - d * self.normal)

    # ----------------------------
    # Fast cached point-in-face
    # ----------------------------
    def _build_2d_cache(self):
        # Build orthonormal basis (u,w) on the face plane
        P0 = self._P0
        n = self.normal

        # choose u from an edge direction not parallel to normal
        e = (self.vertices[1].as_array() - P0).astype(float)
        e_norm = np.linalg.norm(e)
        if e_norm == 0:
            e = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            e = e / e_norm

        # ensure u is in plane
        e = e - np.dot(e, n) * n
        e_norm = np.linalg.norm(e)
        if e_norm == 0:
            # pick an arbitrary vector not parallel to n
            tmp = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(np.dot(tmp, n)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=float)
            e = tmp - np.dot(tmp, n) * n
            e = e / np.linalg.norm(e)
        else:
            e = e / e_norm

        u = e
        w = np.cross(n, u)
        w = w / np.linalg.norm(w)

        self._u = u
        self._w = w

        def to2d(P3: np.ndarray) -> np.ndarray:
            v = P3 - P0
            return np.array([np.dot(u, v), np.dot(w, v)], dtype=float)

        # outer polygon
        outer2d = np.vstack([to2d(v.as_array().astype(float)) for v in self.vertices])
        self._outer2d = outer2d
        self._outer_bbox = (
            float(np.min(outer2d[:, 0])), float(np.max(outer2d[:, 0])),
            float(np.min(outer2d[:, 1])), float(np.max(outer2d[:, 1])),
        )

        # holes
        self._holes2d = []
        self._holes_bbox = []
        for hole in self.holes:
            h2d = np.vstack([to2d(v.as_array().astype(float)) for v in hole])
            self._holes2d.append(h2d)
            self._holes_bbox.append((
                float(np.min(h2d[:, 0])), float(np.max(h2d[:, 0])),
                float(np.min(h2d[:, 1])), float(np.max(h2d[:, 1])),
            ))

    @staticmethod
    def _point_on_segment(q, a, b, eps=1e-10) -> bool:
        ax, ay = a
        bx, by = b
        qx, qy = q

        abx = bx - ax
        aby = by - ay
        aqx = qx - ax
        aqy = qy - ay

        cross = abx * aqy - aby * aqx
        if abs(cross) > eps:
            return False

        dot = aqx * abx + aqy * aby
        if dot < -eps:
            return False

        ab2 = abx * abx + aby * aby
        if dot > ab2 + eps:
            return False

        return True

    @staticmethod
    def _even_odd_contains_2d(q, poly2d: np.ndarray, bbox, eps=1e-12) -> bool:
        qx, qy = float(q[0]), float(q[1])
        minx, maxx, miny, maxy = bbox
        if qx < minx - 1e-9 or qx > maxx + 1e-9 or qy < miny - 1e-9 or qy > maxy + 1e-9:
            return False

        n = poly2d.shape[0]

        # boundary check
        for i in range(n):
            a = poly2d[i]
            b = poly2d[(i + 1) % n]
            if Face._point_on_segment((qx, qy), a, b, eps=1e-10):
                return True

        inside = False
        for i in range(n):
            x1, y1 = float(poly2d[i][0]), float(poly2d[i][1])
            x2, y2 = float(poly2d[(i + 1) % n][0]), float(poly2d[(i + 1) % n][1])

            if (y1 > qy) != (y2 > qy):
                x_int = x1 + (x2 - x1) * (qy - y1) / (y2 - y1)
                if x_int >= qx - eps:
                    inside = not inside

        return inside

    def contains(self, point: Union[Point, np.ndarray], eps_plane: float = 1e-8) -> bool:
        if isinstance(point, np.ndarray):
            Q3 = point.astype(float)
        else:
            Q3 = point.as_array().astype(float)

        # point must lie on plane
        if abs(np.dot(self.normal, Q3 - self._P0)) > eps_plane:
            return False

        v = Q3 - self._P0
        q2 = np.array([np.dot(self._u, v), np.dot(self._w, v)], dtype=float)

        if not self._even_odd_contains_2d(q2, self._outer2d, self._outer_bbox):
            return False

        # inside a hole => not contained
        for h2d, hb in zip(self._holes2d, self._holes_bbox):
            if self._even_odd_contains_2d(q2, h2d, hb):
                return False

        return True
