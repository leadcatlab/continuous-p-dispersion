from typing import Any, List, Tuple

import numpy as np

from tsgo.container.point import Point

def quantize(v: np.ndarray, delta) -> Tuple[int, int, int]:
    """
    Quantize a continuous vector into a discrete cell identifier.

    This function converts a continuous n-dimensional vector into a stable discrete
    representation by dividing each component by a delta value and flooring the result
    to obtain integer cell coordinates.

    Args:
        v (np.ndarray): A continuous vector to be quantized.
        delta: The quantization step size. Each component of the vector is divided
               by this value before discretization.

    Returns:
        Tuple[int, int, int]: A tuple of integers representing the discrete cell
                              identifier corresponding to the quantized vector.
    """
    # Convert continuous dest to a stable discrete "cell id"
    return tuple(np.floor(v / delta).astype(int).tolist())

def is_tabu(tabu_list: List[Tuple[Any, ...]], key: Tuple[Any, ...], it: int) -> bool:
    """
    Determines whether a given key is considered 'tabu' based on the current iteration.
    A key is deemed 'tabu' if it exists in the tabu list and the current iteration 
    is less than the associated expiration iteration.
    Args:
        key (Tuple[Any, ...]): The key to check against the tabu list.
        it (int): The current iteration number.
    Returns:
        bool: True if the key is tabu (exists in the tabu list and is not expired), 
               False otherwise.
    """
    exp = tabu_list.get(key)
    return (exp is not None) and (it < exp)

def tabu_add(tabu_list: List[Tuple[Any, ...]], key: Tuple[Any, ...], it: int, tabu_tenure: int, tabu_max_size: int) -> None:
    """
    Adds a key to the tabu list with an expiration iteration based on the current
    iteration and the predefined tabu tenure. If the tabu list exceeds its maximum
    size, the oldest entries are removed to maintain the size constraint.
    
    Args:
        key (Tuple[Any, ...]): The key to add to the tabu list.
        it (int): The current iteration number.
    """
    tabu_list[key] = it + tabu_tenure
    # Enforce max size by dropping oldest keys
    while len(tabu_list) > tabu_max_size:
        oldest_key = next(iter(tabu_list))
        tabu_list.pop(oldest_key, None)