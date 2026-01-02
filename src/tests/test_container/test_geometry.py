import numpy as np
import pytest

from tsgo.container.point import Point
from tsgo.container import geometry

# ----------------------------
# Distance Tests
# ----------------------------
def test_distance_euclidean():
    p1 = Point(0, 0, 0)
    p2 = Point(3, 4, 0)
    
    assert geometry.distance(p1, p2, metric="EUCLIDEAN") == 5.0
    assert geometry.distance(p1, p2, metric="L2") == 5.0
    
def test_distance_taxicab():
    p1 = Point(0, 0, 0)
    p2 = Point(3, 4, 5)
    
    assert geometry.distance(p1, p2, metric="TAXICAB") == 12.0
    assert geometry.distance(p1, p2, metric="L1") == 12.0
    
def test_distance_sup_norm():
    p1 = Point(0, 0, 0)
    p2 = Point(3, -4, 5)
    
    assert geometry.distance(p1, p2, metric="SUP_NORM") == 5.0
    assert geometry.distance(p1, p2, metric="LINF") == 5.0

# ----------------------------
# Point to Line Distance Tests
# ----------------------------
def test_point_to_line_dist_numpy_inputs():
    point = np.array([1.0, 2.0, 3.0])
    line = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 5.0]))
    
    assert geometry.point_to_line_dist(point, line) == np.sqrt(5)
    assert geometry.point_to_line_dist(point, line, metric="L1") == 3.0
    assert geometry.point_to_line_dist(point, line, metric="LINF") == 2.0

def test_point_to_line_dist_within_range():
    # Test case where point is closest to a point a + tb for t in (0, 1)
    point = Point(5, 0, 1)
    line = (Point(0, 0, 0), Point(10, 0, 0))
    
    assert geometry.point_to_line_dist(point, line) == 1.0
    assert geometry.point_to_line_dist(point, line, metric="L1") == 1.0
    assert geometry.point_to_line_dist(point, line, metric="LINF") == 1.0
    
def test_point_to_line_dist_outside_range_t_less_than_0():
    # Test case where point is closest to a point on the line segment
    point = Point(-5, 0, 0)
    line = (Point(0, 0, 0), Point(10, 0, 0))
    
    assert geometry.point_to_line_dist(point, line) == 5.0
    assert geometry.point_to_line_dist(point, line, metric="L1") == 5.0
    assert geometry.point_to_line_dist(point, line, metric="LINF") == 5.0
    
def test_point_to_line_dist_outside_range_t_more_than_1():
    # Test case where point is closest to a point on the line segment
    point = Point(15, 0, 0)
    line = (Point(0, 0, 0), Point(10, 0, 0))
    
    assert geometry.point_to_line_dist(point, line) == 5.0
    assert geometry.point_to_line_dist(point, line, metric="L1") == 5.0
    assert geometry.point_to_line_dist(point, line, metric="LINF") == 5.0

def test_point_to_line_dist_endpoints_coincide():
    # Test case where the endpoints of the line coincide
    point = Point(0, 0, 0)
    line = (Point(10, 0, 0), Point(10, 0, 0))
    
    assert geometry.point_to_line_dist(point, line) == 10.0
    assert geometry.point_to_line_dist(point, line, metric="L1") == 10.0
    assert geometry.point_to_line_dist(point, line, metric="LINF") == 10
