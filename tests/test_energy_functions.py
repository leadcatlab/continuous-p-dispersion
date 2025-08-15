from src.container import Container
from src.sample_polygons import square, h_shaped_box, v_box
import src.energy_functions as ef
from random import random
import numpy as np

ALPHA, GAMMA, LAMBDA = ef.ALPHA, ef.GAMMA, ef.LAMBDA
square_polygon = square(side_length=3.0)

def test_l_ij():
    D = 1.0
    x_i = np.array([0.0, 0.0])
    x_j = np.array([1.0, 1.0])
    assert ef.l_ij(x_i, x_j, D) == 2 - np.sqrt(2)

    x_j = np.array([2.0, 2.0])
    assert ef.l_ij(x_i, x_j, D) == 0.0
    
    x_j = np.array([0.5, 0.5])
    assert ef.l_ij(x_i, x_j, D) == 2 - np.sqrt(0.5)
    
def test_Oi0_square():
    container = square_polygon
    actual_Oi0 = ef.Oi0(np.array([1.5, 1.5]), 1.0, container)
    expected_Oi0 = 0.0
    assert actual_Oi0 == expected_Oi0

    actual_Oi0 = ef.Oi0(np.array([4.0, 4.0]), 1.0, container)
    expected_Oi0 = GAMMA * (1 + np.sqrt(2)) ** 2
    assert actual_Oi0 == expected_Oi0
    
    actual_Oi0 = ef.Oi0(np.array([1.5, 1.5]), 1e100, container)
    assert actual_Oi0 >= 1e200
    # Very large penalty for huge D
    
def test_E_square():
    container = square_polygon
    actual_E = ef.E(np.array([[1.5, 1.5]]), 1.0, container)
    expected_E = 0.0
    assert actual_E == expected_E

    actual_E = ef.E(np.array([[4.0, 4.0]]), 1.0, container)
    expected_E = GAMMA * (1 + np.sqrt(2)) ** 2
    assert actual_E == expected_E
    
    actual_E = ef.E(np.array([[1.5, 1.5], [2.0, 2.0]]), 1e100, container)
    assert actual_E >= 1e200
    
def test_point_energy_square():
    container = square_polygon
    points = [[1.5, 1.5]]
    actual_point_energy = ef.point_energy(points, 0, 1.0, container)
    expected_point_energy = 0.0
    assert actual_point_energy == expected_point_energy
