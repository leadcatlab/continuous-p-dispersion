from typing import List
import numpy as np


def is_counter_clockwise(p1, p2, p3):
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

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
        # The cross-product of (point - A) and (B - A) is zero and dot product is positive iff point is on the line.
        a, b = edge
        point = np.array(point)
        a, b = np.array(a), np.array(b)

        first = point - a
        second = b - a
        
        # make first and second into three-dimensional vectors for the cross product
        first = np.append(first, 0)
        second = np.append(second, 0)

        return np.linalg.norm(np.cross(first, second)) == 0 and np.dot(first, second) > 0


    # @param point: a coordinate (ex. point = [0.0, 0.0]) to check.
    # @return: True if the point is inside the container, False otherwise.
    def contains(self, point: List[float]) -> bool:
        # Check if the point is a vertex of the container
        for vertex in self.vertices_as_list:
            if np.array_equal(vertex, point):
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

    # @param point: the coordinate to retrieve all active footpoints.
    # @return: a list of active footpoints with respect to `point`.
    def get_active_footpoints(self, point: List[float], as_numpy: bool = False) -> List[List[float]]:

        def _get_active_footpoint(point: List[float], edge: tuple[List[float], List[float]]) -> List[float]:
            a, b = edge
            a = np.array(a)
            b = np.array(b)
            point = np.array(point)
            
            # if the point does not have an active footpoint wrt the edge, return None
            if self.contains(point) and not is_counter_clockwise(a, b, point):
                return None
            elif not self.contains(point) and is_counter_clockwise(a, b, point):
                return None

            # Project point onto the line defined by edge (a, b)
            ab = b - a
            ap = point - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            if t < 0 or t > 1:
                return None  # Projection does not lie on the edge
            return (a + t * ab).tolist()

        active_footpoints = []
        for edge in self.edges:
            footpoint = _get_active_footpoint(point, edge)
            if footpoint is not None:
                active_footpoints.append(np.array(footpoint))
        active_footpoints.sort(key=lambda x: (x[0], x[1]))  # Sort by x, then y
        return active_footpoints if not as_numpy else np.array(active_footpoints)

    # @return: the area of the container using the shoelace formula
    def area(self) -> float:
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        return abs(area) / 2.0

    # @return: the dimensions of the minimal bounding box for the polygon container
    def _bounding_box(self) -> List[float]:
        min_x = min(vertex[0] for vertex in self.vertices_as_list)
        max_x = max(vertex[0] for vertex in self.vertices_as_list)
        min_y = min(vertex[1] for vertex in self.vertices_as_list)
        max_y = max(vertex[1] for vertex in self.vertices_as_list)
        return [min_x, min_y, max_x, max_y]

    def _sample_random_point(self) -> np.ndarray:
        bounding_box = self._bounding_box()
        x = np.random.uniform(bounding_box[0], bounding_box[2])
        y = np.random.uniform(bounding_box[1], bounding_box[3])
        return np.array([x, y])

    # @param p: the number of points to generate
    # @return: an array of shape (p, 2) containing the generated points inside the container
    def generate_initial_solution(self, p: int) -> np.ndarray:
        points = []
        while len(points) < p:
            point = self._sample_random_point()
            points.append(point)
        return np.array(points)