from src.metric_enums import MetricType
from src.point import Point
import numpy as np

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
    assert point1.dist_to(point2, metric=lambda p1, p2: np.linalg.norm(p1.as_array() - p2.as_array())) == (9 + 9 + 9) ** 0.5
    assert point1.dist_to(point3, metric=lambda p1, p2: np.linalg.norm(p1.as_array() - p2.as_array())) == 1.0
    assert point1.dist_to(point4, metric=lambda p1, p2: np.linalg.norm(p1.as_array() - p2.as_array())) == (1 + 1 + 16) ** 0.5
    assert point2.dist_to(point3, metric=lambda p1, p2: np.linalg.norm(p1.as_array() - p2.as_array())) == (4 + 9 + 9) ** 0.5
    assert point2.dist_to(point4, metric=lambda p1, p2: np.linalg.norm(p1.as_array() - p2.as_array())) == (4 + 4 + 49) ** 0.5
    assert point3.dist_to(point4, metric=lambda p1, p2: np.linalg.norm(p1.as_array() - p2.as_array())) == (0 + 1 + 16) ** 0.5
    
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
