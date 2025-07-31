import numpy as np

class Point:
    def __init__(self, x, y, z):
        """Initialize a 3D point.

        :param x: x-coordinate of the point.
        :param y: y-coordinate of the point.
        :param z: z-coordinate of the point."""
        self.x = x
        self.y = y
        self.z = z
            
    def dist_to(self, point):
        return ((self.x - point.x) ** 2 + (self.y - point.y) ** 2 + (self.z - point.z) ** 2) ** 0.5
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __lt__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def as_array(self):
        return np.array([self.x, self.y, self.z])
    
    def __repr__(self):
        return f"Point({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"