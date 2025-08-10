from typing import List
import numpy as np

class Container:
    
    # @return: list of edges defined by pairs of vertices; vertices are pairs of floats (coordinates)
    def _get_edges(self) -> List[tuple[List[float], List[float]]]:
        edges = []
        for i in range(len(self.vertices_as_list)):
            a = self.vertices_as_list[i]
            b = self.vertices_as_list[(i + 1) % len(self.vertices_as_list)]
            edges.append((a, b))
        return edges
    
    # Constructor
    # @param vertices: list of dimensions (n, 3) of vertices listed in counter-clockwise order.
    # @param description: description of the container.
    def __init__(self, vertices: List[List[float]], description: str = ""):
        self.vertices_as_list = vertices
        self.vertices = np.array(vertices)  # vertices must be listed in counter-clockwise order
        self.edges = self._get_edges()
        self.description = description

    # @param point: a coordinate (ex. point = [0.0, 0.0]) to check.
    # @param edge: a line segment defined by two points (start and end).
    # @return: True if the point lies on the segment, False otherwise.
    def _point_on_segment(self, point: List[float], edge: List[tuple[List[float], List[float]]]) -> bool:
        # The cross-product of (point - A) and (B - A) is zero iff point is on the line and dot product is positive.
        a, b = edge
        point = np.array(point)
        a, b = np.array(a), np.array(b)

        first = point - a
        second = b - a
        return np.cross(first, second) == 0 and np.dot(first, second) > 0


    # @param point: a coordinate (ex. point = [0.0, 0.0]) to check.
    # @return: True if the point is inside the container, False otherwise.
    def contains(self, point: List[float]) -> bool:
        # Check if the point is a vertex of the container
        if point in self.vertices_as_list:
            return True
        
        # Check if the point is on any of the edges
        for edge in self.edges:
            if self._point_on_segment(point, edge):
                return True

        edges = np.array(self.edges)
        
        # Check if the point lies in the interior or boundary
        # Ray casting algorithm for non-convex polygons
        x, y = point
        inside = False
        n = len(self.vertices_as_list)
        for i in range(n):
            xi, yi = self.vertices_as_list[i]
            xj, yj = self.vertices_as_list[(i + 1) % n]
            if ((yi > y) != (yj > y)):
                x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
                if x < x_intersect:
                    inside = not inside
        return inside