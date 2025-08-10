import numpy as np

def is_counter_clockwise(p1, p2, p3):
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

class Container:
    """
    Represents a container, which can be used to define a multiply connected domain.
    Each face is a Face object with vertices and a description.
    """
    def __init__(self, vertices, description: str = ""):
        self.vertices = vertices # vertices must be listed in counter-clockwise order
        if (len(self.vertices) < 3):
            raise ValueError("Container must have at least 3 vertices.")

        self.edges = self.get_edges()
        self.description = description

    # @return: list of edges, each edge is a tuple of two vertices, each vertex is a length-2 list of coordinates (numbers)
    # Ex. [([-1, -1], [1, -1]), ([1, -1], [1, 1]), ([1, 1], [-1, 1]), ([-1, 1], [-1, -1])]
    def get_edges(self):
        edges = []
        for i in range(len(self.vertices)):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % len(self.vertices)]
            edge = (a, b)
            edges.append(edge)
        return edges

    # @description: Check if a point is inside the container using ray-casting algorithm.
    # @param point: coordinate pair (length-2 list) of the point to check
    # @param num_rays: number of rays to cast for the ray-casting algorithm
    # @return: True if the point is inside the container, False otherwise
    def contains(self, point, num_rays: int = 100) -> bool:
        
        # First, determine if a point is a vertex of the container
        for vertex in self.vertices:
            if vertex == point:
                return True

        # Second, determine if the point lies on any edge of the container
        for edge in self.edges:
            a, b = edge
            edge_vector = b - a
            point_vector = point - a

            edge_vector_normalized = edge_vector / np.linalg.norm(edge_vector)
            point_vector_normalized = point_vector / np.linalg.norm(point_vector)

            if np.allclose(edge_vector_normalized, point_vector_normalized, atol=1e-8):
                return True

        def ray_intersects_segment(ray_start, ray_end, a, b):
            """
            Check if a ray intersects a line segment with endpoints a and b.
            """            
            ccw1 = is_counter_clockwise(ray_start, a, b)
            ccw2 = is_counter_clockwise(ray_end, a, b)
            ccw3 = is_counter_clockwise(ray_start, ray_end, a)
            ccw4 = is_counter_clockwise(ray_start, ray_end, b)
            
            return (ccw1 != ccw2) and (ccw3 != ccw4)

        # Third, use the ray-casting algorithm to determine if the point is inside the container
        rays_inside = 0
        
        for _ in range(num_rays):
            angle = np.random.uniform(0, 2 * np.pi)
            rand_vector = np.array([np.cos(angle), np.sin(angle)])
            ray_start = np.array(point)
            ray_end = ray_start + rand_vector * 1e6  # Extend the ray
            
            intersections = 0
            for edge in self.edges:
                a, b = edge
                if ray_intersects_segment(ray_start, ray_end, a, b):
                    intersections += 1
            if intersections % 2 == 1: rays_inside += 1
        
        rays_outside = num_rays - rays_inside
        # print(f"Rays inside: {rays_inside}, Rays outside: {rays_outside}")
        if rays_inside == num_rays:
            # print(f"Point {point} is inside the container ({self.description}).")
            pass
        return rays_inside == num_rays

    def is_active_edge(self, point, edge):
        """
        Check if an edge is active for a given point.
        An active edge is one where the point is on the same side as the edge normal.
        """
        
        # return True if point is in container and lies to the left of the edge
        # return True if point is not in container and lies to the right of the edge
        
        if self.contains(point):
            # Point is inside the container, check if it lies to the left of the edge
            a, b = edge
            return is_counter_clockwise(a, b, point)
        else:
            # Point is outside the container, check if it lies to the right of the edge
            a, b = edge
            return not is_counter_clockwise(a, b, point)


    def get_active_footpoint(self, point, edge):
        """
        Check if a point has an active footpoint for a given edge.
        A footpoint is active if the point is on the same side as the edge normal.
        """

        if not self.is_active_edge(point, edge): return None

        a, b = edge
        
        # project point onto edge via a vector projection
        edge_vector = b - a
        point_vector = point - a
        edge_length_squared = np.dot(edge_vector, edge_vector)

        footpoint = (((np.dot(point_vector, edge_vector) / edge_length_squared)) * edge_vector) + a

        # return footpoint if it lies on the edge segment
        if 0 <= np.dot(footpoint - a, edge_vector) <= edge_length_squared:
            return footpoint

        return None

    def get_active_footpoints(self, point) -> list:
        """
        Get all active footpoints for a given point.
        An active footpoint is one where the point is on the same side as the face normal.
        """
        active_edges = [edge for edge in self.edges if self.is_active_edge(point, edge)]
        footpoints = [self.get_active_footpoint(point, edge) for edge in active_edges]
        return [footpoint for footpoint in footpoints if footpoint is not None]

    def area(self):
        # compute the area of the container using the shoelace formula
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) / 2.0

    def bounding_box(self):
        pts_all = np.array([v.as_array() for v in self.vertices])
        x_min, x_max = pts_all[:, 0].min(), pts_all[:, 0].max()
        y_min, y_max = pts_all[:, 1].min(), pts_all[:, 1].max()
        
        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max
        }
        
    def generate_initial_solution(self, p: int) -> np.ndarray:
        """
        Generate an initial random solution of points within the bounding box of the container.
        :param p: Number of points to generate.
        :return: Array of shape (p, 2) representing the initial solution points.
        """
        bounding_box = self.bounding_box()
        x_min, x_max = bounding_box['x_min'], bounding_box['x_max']
        y_min, y_max = bounding_box['y_min'], bounding_box['y_max']
        
        points = []
        for _ in range(p):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            points.append([x, y])

        return np.array(points).reshape(-1, 2)

    def random_point(self):
        """
        Generate a random point within the bounding box of the container.
        :return: A Point object representing the random point.
        """
        bounding_box = self.bounding_box()
        while True:
            x = np.random.uniform(bounding_box['x_min'], bounding_box['x_max'])
            y = np.random.uniform(bounding_box['y_min'], bounding_box['y_max'])
            point = np.array([x, y])
            if self.contains(point):
                return point

    def __repr__(self):
        desc = f" {self.description}" if self.description else ""
        return f"Container with {len(self.vertices)} vertices. " + desc