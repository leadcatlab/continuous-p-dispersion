import numpy as np
from typing import List

from tsgo.container.point import Point

class Face:
    def __init__(self, vertices: List[Point], normal: np.ndarray, description: str = ""):
        """
        Initializes a Plane object with vertices and a normal vector.
        
        :param vertices: List of Point objects defining the plane.
        :param normal: Normal vector of the plane as a numpy array to define its orientation.
        :param description: Optional description of the plane. (Ex. Left, Right, Top, Bottom, etc.)
        """
        self.vertices = vertices
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


    def contains(self, point: Point) -> bool:
        """
        True iff `point` lies inside or on the boundary of the planar polygon `face` in 3D.
        Assumes: face.vertices is an ordered loop of Point; Point has .x/.y/.z or .as_array().
        """
        TOL = 1e-8

        V = [v.as_array() for v in self.vertices]
        n = len(V)
        assert n >= 3, "Face must have at least 3 vertices."

        # Plane normal from a non-collinear triple
        P0 = V[0]
        nvec = None
        for i in range(1, n - 1):
            c = np.cross(V[i] - P0, V[i + 1] - P0)
            nc = np.linalg.norm(c)
            if nc > TOL:
                nvec = c / nc
                break
        if nvec is None:
            return False  # degenerate (collinear)

        # Planarity + point-on-plane check
        if max(abs(np.dot(nvec, v - P0)) for v in V) > TOL:
            return False
        Q = point.as_array()
        if abs(np.dot(nvec, Q - P0)) > TOL:
            return False

        # Orthonormal basis (u, v) on the plane
        e0 = None
        for i in range(n):
            e = V[(i + 1) % n] - V[i]
            if np.linalg.norm(e) > TOL:
                e0 = e
                break
        if e0 is None:
            return False

        u = e0 - np.dot(e0, nvec) * nvec
        nu = np.linalg.norm(u)
        if nu <= TOL:
            axis = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(axis, nvec)) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            u = axis - np.dot(axis, nvec) * nvec
            nu = np.linalg.norm(u)
            if nu <= TOL:
                axis = np.array([0.0, 0.0, 1.0])
                u = axis - np.dot(axis, nvec) * nvec
                nu = np.linalg.norm(u)
                if nu <= TOL:
                    return False
        u /= nu
        v = np.cross(nvec, u)

        def proj2d(X):
            d = X - P0
            return float(np.dot(d, u)), float(np.dot(d, v))

        poly2d = [proj2d(P) for P in V]
        qx, qy = proj2d(Q)

        # Boundary check (edge or vertex)
        def on_seg(ax, ay, bx, by):
            cross = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
            if abs(cross) > TOL:
                return False
            return (min(ax, bx) - TOL <= qx <= max(ax, bx) + TOL and
                    min(ay, by) - TOL <= qy <= max(ay, by) + TOL)

        for i in range(n):
            x1, y1 = poly2d[i]
            x2, y2 = poly2d[(i + 1) % n]
            if on_seg(x1, y1, x2, y2):
                return True

        # Even–odd ray cast (rightward), upper endpoint excluded
        inside = False
        for i in range(n):
            x1, y1 = poly2d[i]
            x2, y2 = poly2d[(i + 1) % n]
            if abs(y1 - y2) <= TOL:
                continue
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            if y1 - TOL <= qy < y2 - TOL:
                x_int = x1 + (x2 - x1) * (qy - y1) / (y2 - y1)
                if x_int >= qx - TOL:
                    inside = not inside

        return inside


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