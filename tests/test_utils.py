import numpy as np
from src.utils import local_optimization, local_optimization_phi
from src.sample_polygons import square
import src.energy_functions as ef
import src.utils as utils

def test_local_optimization():
    X0 = np.array([[0.1, 0.3]]).reshape(-1, 2)
    D0 = 0.5
    container = square(side_length=3.0)
    optimized_points = local_optimization(X0, D0, container)
    assert optimized_points.shape == (1, 2)  # Check if the shape is (1, 2)
    assert ef.E(optimized_points, D0, container) == 0  # Check if the energy is minimized
    
def test_local_optimization_phi():
    X0 = np.array([[0.1, 0.3]]).reshape(-1, 2)
    D0 = 0.5
    container = square(side_length=3.0)
    optimized_points, optimized_d = local_optimization_phi(X0, D0, container, mu=10000.0)
    
    print(optimized_points, optimized_d)
    
    assert optimized_points.shape == (1, 2)  # Check if the shape is (1, 2)
    assert ef.E(optimized_points, optimized_d, container) < ef.E(X0, D0, container)  # Check if the energy is reduced
    assert ef.E(optimized_points, optimized_d, container) < 1e-3  # Energy should be close to 0
    assert optimized_d >= 0 # Check if the optimized distance is non-negative
    
def test_monotonic_basin_hopping():
    X0 = np.array([[0.1, 0.3]]).reshape(-1, 2)
    D0 = 1.5
    container = square(side_length=3.0)
    optimized_points = utils.monotonic_basin_hopping(X0, D0, container)
    assert optimized_points.shape == (1, 2)  # Check if the shape is (1, 2)
    assert ef.E(optimized_points, D0, container) < ef.E(X0, D0, container)  # Check if the energy is reduced
    print(f"energy after mbh:{ef.E(optimized_points, D0, container)}")
    print(f"optimized points: {optimized_points}, distance: {D0}")