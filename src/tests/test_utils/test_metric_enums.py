import numpy as np
import pytest

from tsgo.utils import metric_enums

def test_compute_euclidean_norm():
    a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
    
    assert np.isclose(metric_enums.euclidean(a, b), 27 ** 0.5)
    
def test_compute_taxicab_norm():
    a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
    
    assert np.isclose(metric_enums.taxicab(a, b), 9)
    
def test_compute_sup_norm():
    a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
    
    assert np.isclose(metric_enums.sup_norm(a, b), 3)

def test_compute_n_norm():
    a, b = np.random.rand(2, 3)
    
    # Test that the 2_norm is equal to the Euclidean norm
    assert metric_enums.n_norm(a, b, 2) == np.linalg.norm(a - b, ord=2)
    assert metric_enums.n_norm(a, b, 2) == metric_enums.euclidean(a, b)
    
    # Test that the 1_norm is equal to the taxicab norm
    assert metric_enums.n_norm(a, b, 1) == np.linalg.norm(a - b, ord=1)
    assert metric_enums.n_norm(a, b, 1) == metric_enums.taxicab(a, b)
    
def test_get_metric_function_basic():
    assert metric_enums.get_metric_function('EUCLIDEAN') == metric_enums.euclidean
    assert metric_enums.get_metric_function('TAXICAB') == metric_enums.taxicab
    
def test_get_metric_function_with_n_norm():
    function = metric_enums.get_metric_function('N_NORM_3')
    a, b = np.array([1, 2, 3]), np.array([4, 5, 6])
    
    assert np.isclose(function(a, b), (27 * 3) ** (1/3))
    
def test_get_metric_function_with_invalid_metric():
    with pytest.raises(ValueError):
        metric_enums.get_metric_function('N_NORM_ABC')
    
    with pytest.raises(ValueError):
        metric_enums.get_metric_function('INVALID_METRIC')