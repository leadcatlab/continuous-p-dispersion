from typing import Optional, Union, Tuple
from tsgo.metric_enums import MetricType
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

    @classmethod
    def from_array(cls, array):
        """Create a Point from a numpy array."""
        if len(array) != 3:
            raise ValueError(f"Array ({array}) must be of shape (3,), current shape: {array.shape}")
        return cls(array[0], array[1], array[2])

    def dist_to(self, point, metric: callable = MetricType['EUCLIDEAN']):
        if metric == MetricType['EUCLIDEAN']:
            return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2 + (self.z - point.z) ** 2)
        elif metric == MetricType['TAXICAB']:
            return np.abs(self.x - point.x) + np.abs(self.y - point.y) + np.abs(self.z - point.z)
        elif metric == MetricType['SUP_NORM']:
            return np.max([np.abs(self.x - point.x), np.abs(self.y - point.y), np.abs(self.z - point.z)])
        elif callable(metric):
            return metric(self, point)
        else:
            raise ValueError(f"Unknown metric: {metric}")

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

    def __add__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x * other.x, self.y * other.y, self.z * other.z)

    def as_array(self):
        return np.array([self.x, self.y, self.z])

    def __repr__(self):
        return f"Point({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"