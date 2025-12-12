import numpy as np
import pytest

from tsgo.container.point import Point
from tsgo.utils.metric_enums import MetricType

point1 = Point(1, 2, 3)
point2 = Point(4, 5, 6)
point3 = Point(2, 2, 3)
point4 = Point(2, 3, -1)

def test_dist_to_euclidean():
    assert point1.dist_to(point2) == (9 + 9 + 9) ** 0.5
    assert point1.dist_to(point3) == 1.0
    assert point1.dist_to(point4) == (1 + 1 + 16) ** 0.5
    assert point2.dist_to(point3) == (4 + 9 + 9) ** 0.5
    assert point2.dist_to(point4) == (4 + 4 + 49) ** 0.5
    assert point3.dist_to(point4) == (0 + 1 + 16) ** 0.5
    
def test_dist_to_taxicab():
    assert point1.dist_to(point2, metric=MetricType['TAXICAB']) == 9
    assert point1.dist_to(point3, metric=MetricType['TAXICAB']) == 1
    assert point1.dist_to(point4, metric=MetricType['TAXICAB']) == 6
    assert point2.dist_to(point3, metric=MetricType['TAXICAB']) == 8
    assert point2.dist_to(point4, metric=MetricType['TAXICAB']) == 11
    assert point3.dist_to(point4, metric=MetricType['TAXICAB']) == 5

def test_dist_to_callable():
    assert point1.dist_to(point2, metric=lambda p1, p2: np.linalg.norm(p1 - p2)) == (9 + 9 + 9) ** 0.5
    assert point1.dist_to(point3, metric=lambda p1, p2: np.linalg.norm(p1 - p2)) == 1.0
    assert point1.dist_to(point4, metric=lambda p1, p2: np.linalg.norm(p1 - p2)) == (1 + 1 + 16) ** 0.5
    assert point2.dist_to(point3, metric=lambda p1, p2: np.linalg.norm(p1 - p2)) == (4 + 9 + 9) ** 0.5
    assert point2.dist_to(point4, metric=lambda p1, p2: np.linalg.norm(p1 - p2)) == (4 + 4 + 49) ** 0.5
    assert point3.dist_to(point4, metric=lambda p1, p2: np.linalg.norm(p1 - p2)) == (0 + 1 + 16) ** 0.5
    
def test_dist_to_sup():
    assert point1.dist_to(point2, metric=MetricType['SUP_NORM']) == 3
    assert point1.dist_to(point3, metric=MetricType['SUP_NORM']) == 1
    assert point1.dist_to(point4, metric=MetricType['SUP_NORM']) == 4
    assert point2.dist_to(point3, metric=MetricType['SUP_NORM']) == 3
    assert point2.dist_to(point4, metric=MetricType['SUP_NORM']) == 7
    assert point3.dist_to(point4, metric=MetricType['SUP_NORM']) == 4

def test_constructor_from_array():
    array = np.array([1, 2, 3])
    point = Point.from_array(array)
    assert point.x == 1
    assert point.y == 2
    assert point.z == 3
    
def test_constructor_from_tuple_invalid():
    with pytest.raises(ValueError):
        _ = Point.from_array((1, 2))
        
    with pytest.raises(ValueError):
        _ = Point.from_array((1, 2, 3, 4))
        
def test_eq():
    point1 = Point(1, 2, 3)
    point2 = Point(1, 2, 3)
    point3 = Point(4, 5, 6)
    
    assert point1 == point2
    assert point1 != point3
    assert point2 != point3
    
def test_eq_not_implemented():
    point1 = Point(1, 2, 3)
    
    assert not point1 == 1
    
def test_hash():
    point1 = Point(1, 2, 3)
    point2 = Point(1, 2, 3)
    point3 = Point(4, 5, 6)
    
    assert hash(point1) == hash(point2)
    assert hash(point1) != hash(point3)

def test_lt():
    point1 = Point(1, 2, 3)
    point2 = Point(4, 5, 6)
    
    assert point1 < point2
    assert not point2 < point1
    
def test_lt_not_implemented():
    point1 = Point(1, 2, 3)
    
    with pytest.raises(TypeError):
        _ = point1 < 1

def test_le():
    point1 = Point(1, 2, 3)
    point2 = Point(1, 2, 6)
    
    assert point1 <= point2
    assert not point2 <= point1
    
def test_le_not_implemented():
    point1 = Point(1, 2, 3)
    
    with pytest.raises(TypeError):
        _ = point1 <= 1

def test_gt():
    point1 = Point(1, 2, 3)
    point2 = Point(4, 5, 6)
    
    assert point2 > point1
    assert not point1 > point2
    
def test_gt_not_implemented():
    point1 = Point(1, 2, 3)
    
    with pytest.raises(TypeError):
        _ = point1 > 1
        
def test_ge():
    point1 = Point(1, 2, 3)
    point2 = Point(1, 2, 6)
    
    assert point2 >= point1
    assert not point1 >= point2
    
def test_ge_not_implemented():
    point1 = Point(1, 2, 3)
    
    with pytest.raises(TypeError):
        _ = point1 >= 1

def test_add():
    point1 = Point(1, 2, 3)
    point2 = Point(4, 5, 6)
    point3 = point1 + point2
    
    assert point3 == Point(5, 7, 9)
    
def test_add_not_implemented():
    point1 = Point(1, 2, 3)
    
    with pytest.raises(TypeError):
        _ = point1 + "hello"
    
def test_sub():
    point1 = Point(1, 2, 3)
    point2 = Point(4, 5, 6)
    point3 = point1 - point2
    
    assert point3 == Point(-3, -3, -3)
    
def test_sub_numpy_array():
    point1 = Point(1, 2, 3)
    point2 = np.array([4, 5, 6])
    point3 = point1 - point2
    
    assert point3 == Point(-3, -3, -3)
    
def test_sub_numpy_array_invalid():
    point1 = Point(1, 2, 3)
    point2 = np.array([4, 5])
    
    with pytest.raises(ValueError):
        _ = point1 - point2

def test_sub_not_implemented():
    point1 = Point(1, 2, 3)
    
    with pytest.raises(TypeError):
        _ = point1 - "hello"
    
def test_mul():
    point1 = Point(1, 2, 3)
    point2 = point1 * 2
    
    assert point2 == Point(2, 4, 6)
    assert point1 * point2 == Point(2, 8, 18)
    
    assert point1 * 2 == Point(2, 4, 6)
    assert 2.0 * point1 == Point(2, 4, 6)
    
def test_mul_not_implemented():
    point1 = Point(1, 2, 3)
    
    with pytest.raises(TypeError):
        _ = point1 * "hello"
    
    with pytest.raises(TypeError):
        _ = "hello" * point1
    
def test_abs():
    point1 = Point(1, -2, -3)
    assert abs(point1) == Point(1, 2, 3)

def test_repr():
    point1 = Point(1, 2, 3)
    assert str(point1) == "Point(1.000, 2.000, 3.000)"
    assert repr(point1) == "Point(1.000, 2.000, 3.000)"