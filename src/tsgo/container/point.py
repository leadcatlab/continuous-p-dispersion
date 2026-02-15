import numpy as np

from typing import Union

class Point:
    def __init__(self, x, y, z):
        """Initialize a 3D point.

        :param x: x-coordinate of the point.
        :param y: y-coordinate of the point.
        :param z: z-coordinate of the point."""
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_array(cls, array):
        """Create a Point from a numpy array."""
        if len(array) != 3:
            raise ValueError(f"Array ({array}) must be of shape (3,), current shape: {len(array)}")
        return cls(array[0], array[1], array[2])

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def __lt__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)
    
    def __le__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y, self.z) <= (other.x, other.y, other.z)
    
    def __gt__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y, self.z) > (other.x, other.y, other.z)
    
    def __ge__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y, self.z) >= (other.x, other.y, other.z)

    def __add__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y, self.z - other.z)
        if isinstance(other, np.ndarray):
            if other.shape == (3,):
                return Point(self.x - other[0], self.y - other[1], self.z - other[2])
            else:
                raise ValueError(f"Numpy array must be of shape (3,), current shape: {other.shape}")
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Point):
            return Point(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (int, float)):
            return Point(self.x * other, self.y * other, self.z * other)
        return NotImplemented
    
    def __rmul__(self, other):
        return self * other

    def __abs__(self):
        return Point(abs(self.x), abs(self.y), abs(self.z))
    
    def as_array(self):
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f"Point({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"
    
def unwrap_point(point: Union[Point, np.ndarray]) -> np.ndarray:
    """Convert a Point or numpy array to a numpy array.

    :param point: Point or numpy array to convert.
    :return: Numpy array representation of the point."""
    if isinstance(point, Point):
        return point.x, point.y, point.z
    
    point_arr = np.asarray(point, dtype=float)
    return point_arr[0], point_arr[1], point_arr[2]