import numpy as np
import time
from scipy.optimize import minimize
from src.tsgo_helpers import local_optimization
from src.container import Container
from src.energy_functions import candidate_quality
from src.point import Point
from src.face import Face
from src.energy_functions import lij, Oi0
from src.sample_polyhedron import cube, h_box

def test_minimize_convex_1d():
    # Simple scalar function in one variable
    def f(x):
        return (x - 2) ** 2
    
    # Initial guess
    x0 = np.array([0.0])
    # Perform minimization
    result = minimize(method='BFGS', fun=f, x0=x0)
    
    print("Optimization result:", result)
    
    # Check if the result is as expected
    assert result.success, "Minimization failed"
    assert np.isclose(result.x, 2.0), "Minimum point is not as expected"
    assert np.isclose(result.fun, 0.0), "Function value at minimum is not as expected"
    
    
def test_minimize_convex_31d():
    # Define a convex quadratic function in 31 variables
    def h(x):
        # Minimum at x = [0, 1, 2, ..., 30]
        return np.sum((x - np.arange(31)) ** 2)
    
    # Initial guess: all zeros
    x0 = np.zeros(31)
    # Perform minimization
    result = minimize(method='BFGS', fun=h, x0=x0)

    print("Optimization result:", result)

    # Check if the result is as expected
    assert result.success, "Minimization failed"
    assert np.allclose(result.x, np.arange(31)), "Minimum point is not as expected"
    assert np.isclose(result.fun, 0.0), "Function value at minimum is not as expected"
    
    
def test_minimize_convex_100d():
    # Define a convex quadratic function in 100 variables
    def h(x):
        # Minimum at x = [0, 1, 2, ..., 99]
        return np.sum((x - np.arange(100)) ** 2)
    
    # Initial guess: all zeros
    x0 = np.zeros(100)
    # Perform minimization
    result = minimize(method='BFGS', fun=h, x0=x0)

    print("Optimization result:", result)

    # Check if the result is as expected
    assert result.success, "Minimization failed"
    assert np.allclose(result.x, np.arange(100)), "Minimum point is not as expected"
    assert np.isclose(result.fun, 0.0), "Function value at minimum is not as expected"

def test_tsgo_helpers():
    X0 = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    D0 = 1.0
    container = cube()
    result = local_optimization(X0, D0, container)
    assert isinstance(result, np.ndarray), "Result should be a numpy array"
    initial_quality = candidate_quality(X0, D0, container)
    optimized_quality = candidate_quality(result, D0, container)
    assert optimized_quality <= initial_quality, "Optimized quality should be less than or equal to initial quality"
    assert optimized_quality >= 0, "Optimized quality should be non-negative"
    
    