import numpy as np
from src.point import Point

def wedge(v1, v2):
    return 

class Face:
    def __init__(self, vertices, normal, description=""):
        """
        Initializes a Plane object with vertices and a normal vector.
        
        :param vertices: List of Point objects defining the plane.
        :param normal: Normal vector of the plane as a numpy array to define its orientation.
        :param description: Optional description of the plane. (Ex. Left, Right, Top, Bottom, etc.)
        """
        self.vertices = vertices
        self.normal = np.array(normal)
        self.description = description
        
    def half_space(self, point):
        """
        Determines which half-space the point is in relative to the plane.
        
        :param point: Point object to check.
        :return: Positive if the point is above the plane, negative if below, and zero if on the plane.
        """
        point_vector = np.array(point.as_array())
        plane_point = np.array(self.vertices[0].as_array())
        return np.dot(self.normal, point_vector - plane_point)
    
    def project_point(self, point: Point):
        """
        Projects a point onto the plane defined by this face.
        
        :param point: Point object to project onto the plane.
        :return: The projected Point object.
        """
        d = self.half_space(point)
        return Point(*(np.array(point.as_array()) - d * self.normal))
    
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