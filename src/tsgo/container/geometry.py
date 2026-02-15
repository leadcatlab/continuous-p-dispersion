import numpy as np
from typing import Tuple, Union

from tsgo.container.point import Point, unwrap_point


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

def point_to_line_dist_euclidean(point: Union[Point, np.ndarray],
                       line_segment: Tuple[Union[Point, np.ndarray], Union[Point, np.ndarray]]) -> float:
    # Extract coords without allocating big arrays
    px, py, pz = unwrap_point(point)
    
    a, b = line_segment
    ax, ay, az = unwrap_point(a)
    bx, by, bz = unwrap_point(b)

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
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)

import numpy as np

def point_to_line_dist_taxicab(point, start, end):
    """
    Calculates the minimum L1 (Taxicab) distance from a point to a line segment.
    Instead of generic optimization, this exploits the fact that the minimum
    must occur at t=0, t=1, or where one coordinate difference becomes zero (a 'kink').
    """
    p = np.array(point)
    s = np.array(start)
    e = np.array(end)
    
    vec = e - s
    rel_p = p - s  # Vector from Start to Point
    
    # Critical values for 't' are 0, 1, and where (p_i - (s_i + t*v_i)) = 0
    # Solving for t: t = (p_i - s_i) / v_i
    candidates = [0.0, 1.0]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate kinks for all dimensions at once
        t_kinks = rel_p / vec
        # Only keep kinks that actually lie within the segment (0 < t < 1)
        # We ignore cases where vec is 0 (parallel to axis) as those don't generate unique kinks
        valid_kinks = t_kinks[np.isfinite(t_kinks) & (t_kinks > 0) & (t_kinks < 1)]
        candidates.extend(valid_kinks)
        
    # Evaluate distance at all critical candidates to find the winner
    min_dist = float('inf')
    
    for t in candidates:
        current_p_on_line = s + t * vec
        dist = np.sum(np.abs(p - current_p_on_line))
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

def point_to_line_dist_sup_norm(point, start, end):
    """
    Calculates the minimum L_inf (Supremum) distance.
    The minimum occurs at endpoints, or where the error magnitude in two dimensions
    is equal (intersection of error functions), or where an error term is zero.
    """
    point = unwrap_point(point)
    start = unwrap_point(start)
    end = unwrap_point(end)
    
    p = np.array(point)
    s = np.array(start)
    e = np.array(end)
    
    vec = e - s
    diff = s - p 
    # We want to minimize f(t) = max_i |diff_i + t * vec_i|
    
    candidates = [0.0, 1.0]
    n = len(p)
    
    # 1. Check intersections where |term_i| == |term_j|
    # This corresponds to the 'creases' of the L_inf unit ball
    for i in range(n):
        for j in range(i + 1, n):
            # Case A: (diff_i + t*v_i) = (diff_j + t*v_j)
            denom_a = vec[i] - vec[j]
            if abs(denom_a) > 1e-9:
                t_a = (diff[j] - diff[i]) / denom_a
                if 0 < t_a < 1: candidates.append(t_a)
            
            # Case B: (diff_i + t*v_i) = -(diff_j + t*v_j)
            denom_b = vec[i] + vec[j]
            if abs(denom_b) > 1e-9:
                t_b = -(diff[j] + diff[i]) / denom_b
                if 0 < t_b < 1: candidates.append(t_b)

    # 2. Check simple zeros (where a term hits 0) just in case
    with np.errstate(divide='ignore'):
         t_zeros = -diff / vec
         valid_zeros = t_zeros[np.isfinite(t_zeros) & (t_zeros > 0) & (t_zeros < 1)]
         candidates.extend(valid_zeros)

    # Evaluate all candidates
    min_dist = float('inf')
    
    for t in candidates:
        current_p_on_line = s + t * vec
        dist = np.max(np.abs(p - current_p_on_line))
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

def point_to_line_dist(point: Union[Point, np.ndarray],
                       line_segment: Tuple[Union[Point, np.ndarray], Union[Point, np.ndarray]],
                       metric: str = "EUCLIDEAN") -> float:
    metric = metric.upper()
    if metric == "EUCLIDEAN":
        return point_to_line_dist_euclidean(point, line_segment)
    elif metric in ["TAXICAB", "L1"]:
        return point_to_line_dist_taxicab(point, line_segment[0], line_segment[1])
    elif metric in ["SUP_NORM", "LINF"]:
        return point_to_line_dist_sup_norm(point, line_segment[0], line_segment[1])
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    