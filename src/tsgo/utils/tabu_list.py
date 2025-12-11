from typing import List

import numpy as np

from tsgo.container.point import Point

def move_point(X: List[Point], i: int, low_energy_point: Point) -> List[Point]:
    """
    Move the high energy point to the low energy point.
    
    Args:
        X (List[Point]): The array of points.
        high_energy_point (int): The index of the point in X considered to have high energy.
        low_energy_point (Point): The point to which the high energy point will be moved.

    Returns
    -------
    List[Point]
        A copy of X with the high energy point replaced by the low energy point.
    """
    
    X_copy = X.copy()
    X_copy[i] = low_energy_point
    return X_copy

def forbidden(high_energy_point: Point, tabu_list: list, iterations: int):
    # Check if the high energy point is in the tabu list
    # 0th index: movement (high energy point, low energy point)
    # 1st index: iteration when the move becomes valid again

    for entry in tabu_list:
        if entry is None:
            continue
        if entry[0][0] == high_energy_point and entry[1] > iterations:
            return True
    return False

def update_tabu_list(I: tuple, tabu_list: list, iterations: int, max_size: int = 100):
    if I is None or I[0] is None or I[1] is None:
        return tabu_list
    # Add a new entry to the tabu list
    tabu_list.append((I, iterations + 10))
    # Remove the oldest entry if the list exceeds max_size
    while len(tabu_list) > max_size:
        tabu_list.pop(0)
    return tabu_list