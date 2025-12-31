import numpy as np
from typing import List

from tsgo.container.point import Point

class Face:
    def __init__(self, vertices: List[Point], normal: np.ndarray, description: str = "", holes: List[List[Point]] = []):
        """
        Initializes a Plane object with vertices and a normal vector.   
        
        :param vertices: List of Point objects defining the plane.
        :param normal: Normal vector of the plane as a numpy array to define its orientation.
        :param description: Optional description of the plane. (Ex. Left, Right, Top, Bottom, etc.)
        """
        self.vertices = vertices
        self.holes = holes  # List of holes, each hole is a list of Point objects listed in order
        self.normal = np.array(normal)
        self._adjust_normal()
        self.description = description
        self.edges = self._get_edges()

    def _adjust_normal(self):
        """
        Adjust the normal vector to ensure it is perpendicular to the face.
        """
        # adjust the normal vector to be perpendicular to the face
        # first compute the cross product
        v1 = self.vertices[1].as_array() - self.vertices[0].as_array()
        v2 = self.vertices[2].as_array() - self.vertices[0].as_array()
        computed_normal = np.cross(v1, v2).astype(float)
        norm = np.linalg.norm(computed_normal)
        if norm > 0:
            computed_normal /= norm

        # if the direction is opposite to the existing self.normal, then flip the resultant cross product
        if np.dot(computed_normal, self.normal) < 0:
            computed_normal = -computed_normal
        self.normal = computed_normal

    def _get_edges(self) -> List[tuple[Point, Point]]:
        """
        Get the edges of the face.

        Returns:
            list: A list of edges defined by pairs of vertices.
        """
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
        """
        Determines which half-space the point is in relative to the plane.
        
        :param point: Point object to check.
        :return: Positive if the point is in line with the normal, negative if not.
        """
        point_vector = point.as_array()
        plane_point = self.vertices[0].as_array()
        return np.dot(self.normal, point_vector - plane_point)

    def project_point(self, point: Point) -> Point:
        """
        Projects a point onto the plane defined by this face.
        
        :param point: Point object to project onto the plane.
        :return: The projected Point object.
        """
        d = self.half_space(point)
        projection = Point.from_array(point.as_array() - d * self.normal)
        return projection

    @staticmethod
    def _contains(point: Point, vertices: List[Point]) -> bool:
        """
        Returns True if the point is located within (or on the boundary of) the face
        defined by the vertices.
        """
        import numpy as np

        EPS = 1e-8

        V3 = [v.as_array() for v in vertices]
        n = len(V3)
        assert n >= 3, "Face must have at least 3 vertices."

        P0 = V3[0]

        # Find a valid face normal from any non-degenerate triangle fan
        nvec = None
        for i in range(1, n - 1):
            c = np.cross(V3[i] - P0, V3[i + 1] - P0)
            nc = np.linalg.norm(c)
            if nc > EPS:
                nvec = c / nc
                break
        if nvec is None:
            return False  # degenerate face

        # Planarity check for the polygon vertices
        if max(abs(np.dot(nvec, P - P0)) for P in V3) > EPS:
            return False

        Q3 = point.as_array()

        # Point-on-plane check
        if abs(np.dot(nvec, Q3 - P0)) > EPS:
            return False

        # Build orthonormal basis (u, w) for the plane
        e0 = None
        for i in range(n):
            e = V3[(i + 1) % n] - V3[i]
            if np.linalg.norm(e) > EPS:
                e0 = e
                break
        if e0 is None:
            return False

        u = e0 - np.dot(e0, nvec) * nvec
        nu = np.linalg.norm(u)
        if nu <= EPS:
            # pick any axis not parallel to nvec
            axis = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(axis, nvec)) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            u = axis - np.dot(axis, nvec) * nvec
            nu = np.linalg.norm(u)
            if nu <= EPS:
                axis = np.array([0.0, 0.0, 1.0])
                u = axis - np.dot(axis, nvec) * nvec
                nu = np.linalg.norm(u)
                if nu <= EPS:
                    return False

        u = u / nu
        w = np.cross(nvec, u)  # second in-plane axis

        def proj2d(X3: np.ndarray) -> np.ndarray:
            d = X3 - P0
            return np.array([np.dot(d, u), np.dot(d, w)], dtype=float)

        poly2d = [proj2d(P) for P in V3]
        q = proj2d(Q3)

        # ---- Boundary check in 2D: point on any edge => inside ----
        def point_on_segment_2d(p, a, b) -> bool:
            ab = b - a
            ap = p - a
            denom = np.dot(ab, ab)
            if denom <= EPS:
                return np.linalg.norm(p - a) <= EPS
            t = np.dot(ap, ab) / denom
            if t < -EPS or t > 1.0 + EPS:
                return False
            closest = a + t * ab
            # distance-to-segment
            return np.linalg.norm(p - closest) <= EPS

        for i in range(n):
            a = poly2d[i]
            b = poly2d[(i + 1) % n]
            if point_on_segment_2d(q, a, b):
                return True

        # ---- Standard even–odd rule (ray to +x), stable crossing test ----
        inside = False
        qx, qy = float(q[0]), float(q[1])

        for i in range(n):
            x1, y1 = float(poly2d[i][0]), float(poly2d[i][1])
            x2, y2 = float(poly2d[(i + 1) % n][0]), float(poly2d[(i + 1) % n][1])

            # Check if edge straddles the horizontal ray at y=qy
            # using half-open interval to avoid double-counting vertices.
            if (y1 > qy) != (y2 > qy):
                x_int = x1 + (x2 - x1) * (qy - y1) / (y2 - y1)
                if x_int >= qx - EPS:
                    inside = not inside

        return inside
    
        # @param point: a coordinate (ex. point = [0.0, 0.0]) to check.
    # @param edge: a line segment defined by two points (start and end).
    # @return: True if the point lies on the segment, False otherwise.
    @staticmethod
    def _point_on_segment(point: Point, edge: tuple[Point, Point]) -> bool:
        # The cross-product of (point - A) and (B - A) is zero and dot product is positive iff point is on the line.
        a, b = edge[0].as_array(), edge[1].as_array()
        point = point.as_array() 

        first = point - a
        second = b - a
        
        EPS = 1e-8
        cross = np.cross(first, second)
        
        # Check if point is collinear (cross product ~= 0) and within segment bounds
        if np.linalg.norm(cross) > EPS:
            return False
        
        # Check if point is between the segment endpoints
        dot_prod = np.dot(first, second)
        len_sq = np.dot(second, second)
        
        return bool(dot_prod > -EPS and dot_prod < len_sq + EPS)

    def contains(self, point: Point) -> bool:
        """
        Determines if the point is located within the feasible region of the face.

        Args:
            point (Point): Point to check.

        Returns:
            bool: True if the point is inside the face, False otherwise.
        """
        for hole in self.holes:
            if Face._contains(point, hole):
                return False
        
        # Check if the point lies on any vertex or edge
        for vertex in self.vertices:
            if np.linalg.norm(point.as_array() - vertex.as_array()) < 1e-8:
                return True
        for edge in self.edges:
            if Face._point_on_segment(point, edge):
                return True
            
        return Face._contains(point, self.vertices)
    
    def area(self) -> float:
        """
        Compute the (planar) polygon area via
          A = 1/2 * | n · (Σ v_i x v_{i+1}) |
        where n is the unit normal.
        """
        # pull out all vertex‐vectors
        vertices = [v.as_array() for v in self.vertices]
        # accumulate the cross‐sum
        S = np.zeros(3)
        n = len(vertices)
        for i in range(n):
            S += np.cross(vertices[i], vertices[(i + 1) % n])
        
        # project onto normal and take absolute half
        return 0.5 * abs(np.dot(S, self.normal))
        

    def __repr__(self):
        return f"Face with {len(self.vertices)} vertices and normal {self.normal}. {self.description if self.description else ''}"