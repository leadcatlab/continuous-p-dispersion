import numpy as np
from typing import Tuple, Union

from tsgo.container.point import Point
from tsgo.utils.metric_enums import MetricType


def point_to_line_dist(point, line, metric=MetricType["EUCLIDEAN"]) -> float:
    p = point.as_array() if isinstance(point, Point) else np.asarray(point, dtype=float)
    a = line[0].as_array() if isinstance(line[0], Point) else np.asarray(line[0], dtype=float)
    b = line[1].as_array() if isinstance(line[1], Point) else np.asarray(line[1], dtype=float)

    ab = b - a
    ap = p - a
    denom = float(np.dot(ab, ab))

    # Degenerate segment
    if denom <= 1e-30:
        if metric is MetricType["EUCLIDEAN"] or metric is np.linalg.norm:
            return float(np.linalg.norm(p - a))
        return float(metric(p, a))

    t = float(np.dot(ap, ab) / denom)

    # Clamp projection to segment [0, 1] (avoids two metric calls)
    if t <= 0.0:
        closest = a
    elif t >= 1.0:
        closest = b
    else:
        closest = a + t * ab

    if metric is MetricType["EUCLIDEAN"] or metric is np.linalg.norm:
        return float(np.linalg.norm(p - closest))
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
