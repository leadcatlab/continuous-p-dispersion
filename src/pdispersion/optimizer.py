import numpy as np
from .geometry import Sphere

def max_min_dispersion(centers: np.ndarray, radii: float, p: float):
    """
    Given initial centers (n×3) and uniform radius,
    maximize the min_{i<j} ‖c_i - c_j‖_p via simple gradient steps.
    """
    # TODO: implement your chosen algorithm (projected gradient, simulated annealing, etc.)
    return centers
