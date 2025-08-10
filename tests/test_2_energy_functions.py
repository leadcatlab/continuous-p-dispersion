import random
import numpy as np
from src.container import Container
from src.energy_functions import candidate_quality, E, phi, lij, Oi0

def test_lij():
    xi = [0.0, 0.0]
    xj = [1.0, 1.0]
    D = 2.0
    assert lij(xi, xj, D) == 4 - (2 ** 0.5), "lij should return the correct distance."

    xi = [0.0, 0.0]
    xj = [1.0, 1.0]
    D = 0.5
    assert lij(xi, xj, D) == 0, "lij should return the correct distance."
    
def test_Oi0_internal_cube():
    xi = [0.0, 0.0, 0.0]
    D = 1.0
    container = cube
    result = Oi0(xi, D, container)
    assert result >= 0, "Oi0 should return a non-negative value."
    assert result == 0, "Oi0 should return zero for p = (0, 0, 0) and D = 1.0."
    
def test_Oi0_external_cube_1():
    xi = [2.0, 0.0, 0.0]
    D = 1.0
    container = cube
    result = Oi0(xi, D, container)
    assert result == 2.0 * (1 + 1) ** 2, "Oi0 should return the correct value for points outside the container."
    
def test_Oi0_external_cube_2():
    xi = [1.5, 0.0, 0.0]
    D = 1.0
    container = cube
    result = Oi0(xi, D, container)
    assert result == 2.0 * (1 + 0.5) ** 2, "Oi0 should return the correct value for points outside the container."
    
def test_candidate_quality():
    X = np.array([[0.0, 0.0, 0.0]])
    D = 1.0
    container = cube
    result = candidate_quality(X, D, container)
    assert result == 0, "candidate_quality should return zero for the origin in the box."

def test_candidate_quality_five_points():
    X = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5]])
    D = 0.2
    container = cube
    result = candidate_quality(X, D, container)
    assert result == 0, "candidate_quality should return zero for dispersed points inside the box with D = 0.2."

def test_random_quality():
    for _ in range(5):
        X = np.array([[random.uniform(-1, 1) for _ in range(3)] for _ in range(10)])
        D = random.uniform(0.1, 1.0)
        container = cube
        result = candidate_quality(X, D, container)
        assert isinstance(result, (int, float)), "candidate_quality should return a numeric value."
        assert result >= 0, "candidate_quality should return a non-negative value."