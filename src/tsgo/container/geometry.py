import numpy as np
from typing import Tuple, Union

from tsgo.container.point import Point
from tsgo.utils.metric_enums import MetricType


def point_to_line_dist(
    point: Point, 
    line: Tuple[Point, Point], 
    metric: callable = MetricType['EUCLIDEAN']
) -> float:
    """
    Computes the distance from a point to a line.

    Args:
        point (Point): The point.
        line (Tuple[Point, Point]): The line defined by two endpoints.
        metric (callable, optional): The distance metric to use. Defaults to MetricType['EUCLIDEAN'].

    Returns:
        float: The distance from the point to the line.
    """
    p = point.as_array() if isinstance(point, Point) else np.array(point)
    a = line[0].as_array() if isinstance(line[0], Point) else np.array(line[0])
    b = line[1].as_array() if isinstance(line[1], Point) else np.array(line[1])

    line_vec = b - a
    point_vec = p - a
    denom = np.dot(line_vec, line_vec)
    
    # If the line endpoints coincide, treat as distance to a point
    if np.isclose(a, b, atol=1e-16).all():
        return metric(p, a)

    # projection parameter t along the segment (0 -> a, 1 -> b)
    t = np.dot(point_vec, line_vec) / denom

    # If projection falls outside the segment, return distance to nearest endpoint using the metric
    if t < 0 or t > 1:
        d0 = metric(p, a)
        d1 = metric(p, b)
        return float(min(d0, d1))

    # Closest point on the segment
    closest = a + t * line_vec
    return float(metric(p, closest))

def is_counter_clockwise(p1: Union[Point, np.ndarray], p2: Union[Point, np.ndarray], p3: Union[Point, np.ndarray]) -> bool:
    """
    Three points p1, p2, p3 are in counter-clockwise order if 
    the triple product (p2 - p1) x (p3 - p1) has positive z-component
    
    :param p1, p2, p3: Three points in 3D space.
    :return: True if the points are in counter-clockwise order.
    """
    p1 = p1.as_array() if isinstance(p1, Point) else np.array(p1)
    p2 = p2.as_array() if isinstance(p2, Point) else np.array(p2)
    p3 = p3.as_array() if isinstance(p3, Point) else np.array(p3)
        
    # 3D case: use the sign of the scalar triple product
    if p1.shape[0] == 3:
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        # You may want to specify a reference normal direction for the polygon's plane.
        # Here, we just check if the cross product is not zero (i.e., not collinear).
        return cross[2] > 0
    
    # Other dimensions not supported
    else:
        raise ValueError("Points must be 2D or 3D coordinates.")
