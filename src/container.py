from typing import List
import numpy as np

class Container:
    
    # Constructor
    # @param vertices: list of dimensions (n, 3) of vertices listed in counter-clockwise order.
    # @param description: description of the container.
    def __init__(self, vertices: List[List[float]], description: str = ""):
        self.vertices_as_list = vertices
        self.vertices = np.array(vertices)  # vertices must be listed in counter-clockwise order
        self.edges = self._get_edges()
        self.description = description
        
    def _get_edges(self) -> List[tuple[List[float], List[float]]]:
        edges = []
        for i in range(len(self.vertices_as_list)):
            a = self.vertices_as_list[i]
            b = self.vertices_as_list[(i + 1) % len(self.vertices_as_list)]
            edges.append((a, b))
        return edges