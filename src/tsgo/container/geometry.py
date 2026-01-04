import numpy as np
from typing import Tuple, Union

from tsgo.container.point import Point


def distance(point1: Union[Point, np.ndarray], point2: Union[Point, np.ndarray], metric: str = "EUCLIDEAN") -> float:
    p1 = point1.as_array() if isinstance(point1, Point) else np.asarray(point1, dtype=float)
    p2 = point2.as_array() if isinstance(point2, Point) else np.asarray(point2, dtype=float)

    metric = metric.upper()
    dx, dy, dz = float(p1[0] - p2[0]), float(p1[1] - p2[1]), float(p1[2] - p2[2])

    if metric in ["TAXICAB", "L1"]:
        return float(abs(dx) + abs(dy) + abs(dz))
    if metric in ["SUP_NORM", "LINF"]:
        return float(max(abs(dx), abs(dy), abs(dz)))

    return float((dx * dx + dy * dy + dz * dz) ** 0.5)

def point_to_line_dist(point: Union[Point, np.ndarray],
                       line_segment: Tuple[Union[Point, np.ndarray], Union[Point, np.ndarray]],
                       metric: str = "EUCLIDEAN") -> float:
    # Extract coords without allocating big arrays
    if isinstance(point, Point):
        px, py, pz = point.x, point.y, point.z
    else:
        point = np.asarray(point, dtype=float)
        px, py, pz = float(point[0]), float(point[1]), float(point[2])

    a, b = line_segment
    if isinstance(a, Point):
        ax, ay, az = a.x, a.y, a.z
    else:
        a = np.asarray(a, dtype=float)
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])

    if isinstance(b, Point):
        bx, by, bz = b.x, b.y, b.z
    else:
        b = np.asarray(b, dtype=float)
        bx, by, bz = float(b[0]), float(b[1]), float(b[2])

    abx, aby, abz = (bx - ax), (by - ay), (bz - az)
    apx, apy, apz = (px - ax), (py - ay), (pz - az)

    denom = abx * abx + aby * aby + abz * abz
    if denom == 0.0:
        # a == b
        cx, cy, cz = ax, ay, az
    else:
        t = (apx * abx + apy * aby + apz * abz) / denom
        if t <= 0.0:
            cx, cy, cz = ax, ay, az
        elif t >= 1.0:
            cx, cy, cz = bx, by, bz
        else:
            cx, cy, cz = (ax + t * abx), (ay + t * aby), (az + t * abz)

    dx, dy, dz = (px - cx), (py - cy), (pz - cz)
    metric = metric.upper()
    if metric in ["TAXICAB", "L1"]:
        return float(abs(dx) + abs(dy) + abs(dz))
    if metric in ["SUP_NORM", "LINF"]:
        return float(max(abs(dx), abs(dy), abs(dz)))

    return float((dx * dx + dy * dy + dz * dz) ** 0.5)
