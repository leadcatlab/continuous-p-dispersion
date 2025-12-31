import numpy as np
from typing import Dict, List, Union

from tsgo.container.face import Face
from tsgo.container.point import Point

class Container:

    # Constructor
    # @param vertices: list of dimensions (n, 3) of vertices listed in counter-clockwise order.
    # @param description: description of the container.
    def __init__(self, faces: List[Face], description: str = ""):
        self.faces = faces
        self.vertices = self._get_vertices() # list of point objects
        self.edges = self._get_edges()
        self.description = description
        self.active_footpoints = {} # cache that stores active footpoints for each point

    # @return: list of edges defined by pairs of vertices; vertices are pairs of floats (coordinates)
    def _get_edges(self) -> List[tuple[Point, Point]]:
        edges = set()
        for face in self.faces:
            for i in range(len(face.vertices)):
                a = face.vertices[i]
                b = face.vertices[(i + 1) % len(face.vertices)]
                edges.add((a, b))
        return list(edges)

    # @return: list of vertices defined by the faces.
    def _get_vertices(self) -> List[Point]:
        vertices = set()
        for face in self.faces:
            for vertex in face.vertices:
                vertices.add(vertex)
        vertices_as_list = list(vertices)
        return vertices_as_list

    def contains(self, point: Union[Point, np.ndarray]) -> bool:
        """
        Ray-casting test in 3D: cast a single ray from `point` in a generic direction,
        count intersections with faces (using Face.contains for the in-face check).
        Odd count => inside. Boundary counts as inside.

        :param point: a coordinate (ex. point = [0.0, 0.0, 0.0]) to check.
        :return: True if the point is inside the container, False otherwise.
        """
        TOL = 1e-8
        
        if isinstance(point, list) or isinstance(point, tuple):
            point = Point(point[0], point[1], point[2])
        elif isinstance(point, np.ndarray):
            point = Point.from_array(point)

        P = point.as_array()

        # Quick boundary check: if the point lies on any face, it's inside.
        for face in self.faces:
            if face.contains(point):
                return True

        # A generic (non-axis-aligned) ray direction to reduce parallel degeneracies.
        d = np.random.normal(size=3)
        d /= np.linalg.norm(d)

        ts = []  # parametric distances along the ray for valid intersections

        for face in self.faces:
            V = [v.as_array() for v in face.vertices]

            # Plane normal from a non-collinear triple
            P0 = V[0]
            nvec = None
            for i in range(1, len(V) - 1):
                c = np.cross(V[i] - P0, V[i + 1] - P0)
                nc = np.linalg.norm(c)
                if nc > TOL:
                    nvec = c / nc
                    break
            if nvec is None:
                continue  # degenerate face

            denom = float(np.dot(nvec, d))
            if abs(denom) <= TOL:
                continue  # ray parallel to this face's plane

            t = float(np.dot(nvec, (P0 - P)) / denom)
            if t <= TOL:
                continue  # behind the start point or too close

            hit = P + t * d

            # Count the hit only if the intersection point lies within the polygon
            if face.contains(Point.from_array(hit)):
                ts.append(t)

        if not ts:
            return False

        # Deduplicate intersections that are essentially the same (e.g., hitting an edge shared by two faces)
        ts.sort()
        unique_ts = []
        last = None
        for t in ts:
            if last is None or abs(t - last) > TOL * max(1.0, abs(t), abs(last)):
                unique_ts.append(t)
                last = t

        # Odd-even rule along the ray
        return (len(unique_ts) % 2) == 1

    def _is_active_plane(self, face: Face, point: Point) -> bool:
        """
        Check if a face is an active plane for a given point.
        An active plane is one where the point is on the same side as the face normal.
        """
        if face is None or face not in self.faces:
            return False
        
        # if container contains point and point is in direction of inward normal, then True
        if self.contains(point) and face.half_space(point) > 0:
            return True
        
        # if container doesn't contain point and point is not in direction of inward normal, then True
        elif not self.contains(point) and face.half_space(point) < 0:
            return True
        
        # otherwise, false
        return False

    def get_active_footpoints(self, point) -> List[Point]:
        # normalize input
        if isinstance(point, list):
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            point = Point(x, y, z)
        elif isinstance(point, np.ndarray):
            # avoid Point.from_array overhead if it's simple
            point = Point(float(point[0]), float(point[1]), float(point[2]))
        # else assume Point

        # quantized key so cache actually hits during local optimization
        key = (round(point.x, 6), round(point.y, 6), round(point.z, 6))
        cached = self.active_footpoints.get(key)
        if cached is not None:
            return cached

        out: List[Point] = []
        for face in self.faces:
            if not self._is_active_plane(face, point):
                continue

            fp = face.project_point(point)

            # If you add cached 2D bbox inside Face.contains, this is already fast.
            if face.contains(fp):
                out.append(fp)

        self.active_footpoints[key] = out
        return out

    def _bounding_box(self) -> Dict[str, float]:
        """
        Calculate the bounding box of the polygon container.

        Returns:
            Dict[str, float]: The dimensions of the minimal bounding box for the polygon container.
        """
        min_x = min(vertex.x for vertex in self.vertices)
        max_x = max(vertex.x for vertex in self.vertices)
        min_y = min(vertex.y for vertex in self.vertices)
        max_y = max(vertex.y for vertex in self.vertices)
        min_z = min(vertex.z for vertex in self.vertices)
        max_z = max(vertex.z for vertex in self.vertices)
        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "min_z": min_z,
            "max_z": max_z
        }

    def _sample_random_point(self) -> np.ndarray:
        """
        Sample a random point inside the container.

        Returns:
            Point: A random point inside the container.
        """
        
        bounding_box = self._bounding_box()
        x = np.random.uniform(bounding_box["min_x"], bounding_box["max_x"])
        y = np.random.uniform(bounding_box["min_y"], bounding_box["max_y"])
        z = np.random.uniform(bounding_box["min_z"], bounding_box["max_z"])

        return np.array([x, y, z]) if self.contains(Point(x, y, z)) else self._sample_random_point()

    def generate_initial_solution(self, p: int) -> np.ndarray:
        """
        Generate an initial solution by sampling points inside the container.

        Args:
            p (int): The number of points to generate.

        Returns:
            np.ndarray: A list of points sampled from the container.
        """
        return np.array([self._sample_random_point() for _ in range(p)])

    def volume(self) -> float:
        """
        Calculate the volume of the container.

        Returns:
            float: The volume of the container.
        """
        volume_summation = 0.0
        for face in self.faces:
            volume_summation += face.area() * np.dot(face.normal, face.vertices[0].as_array())
        return abs(volume_summation) / 3.0