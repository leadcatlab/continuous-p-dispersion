import numpy as np
from pdispersion.geometry import Sphere

def test_sphere_distance():
    s1 = Sphere([0, 0, 0], 1)
    s2 = Sphere([3, 0, 0], 1)
    assert abs(s1.distance_to(s2) - 1.0) < 1e-6
