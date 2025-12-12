import numpy as np
import pytest

from tsgo.container.point import Point
from tsgo.container import geometry

def test_point_to_line_dist_within_range():
    # Test case where point is closest to a point a + tb for t in (0, 1)
    point = Point(5, 0, 1)
    line = (Point(0, 0, 0), Point(10, 0, 0))
    
    assert geometry.point_to_line_dist(point, line) == 1.0
    
def test_point_to_line_dist_outside_range():
    # Test case where point is closest to a point on the line segment
    point = Point(15, 0, 0)
    line = (Point(0, 0, 0), Point(10, 0, 0))
    
    assert geometry.point_to_line_dist(point, line) == 5.0

def test_point_to_line_dist_endpoints_coincide():
    # Test case where the endpoints of the line coincide
    point = Point(0, 0, 0)
    line = (Point(10, 0, 0), Point(10, 0, 0))
    
    assert geometry.point_to_line_dist(point, line) == 10.0
    assert geometry.point_to_line_dist(point, line) == 10.0
    
def test_is_counter_clockwise():
    # Test case where points are in counter-clockwise order
    point1 = Point(0, 0, 0)
    point2 = Point(1, 0, 0)
    point3 = Point(0, 1, 0)
    
    assert geometry.is_counter_clockwise(point1, point2, point3) == True
    
def test_is_counter_clockwise_clockwise():    # Test case where points are in clockwise order
    point1 = Point(0, 0, 0)
    point2 = Point(1, 0, 0)
    point3 = Point(0, -1, 0)
    
    assert geometry.is_counter_clockwise(point1, point2, point3) == False
    
def test_is_counter_clockwise_collinear():    # Test case where points are collinear
    point1 = Point(0, 0, 0)
    point2 = Point(1, 0, 0)
    point3 = Point(2, 0, 0)
    
    assert geometry.is_counter_clockwise(point1, point2, point3) == False
    
def test_is_counter_clockwise_unsupported_dimensions():
    point1 = np.array([0, 0, 0, 0])
    point2 = np.array([1, 0, 0, 0])
    point3 = np.array([0, 1, 0, 0])
    
    with pytest.raises(ValueError):
        _ = geometry.is_counter_clockwise(point1, point2, point3)