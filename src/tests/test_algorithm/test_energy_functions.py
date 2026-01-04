import numpy as np
import time

import tsgo.algorithm.energy_functions as ef
from tsgo.container.container import Container
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import cube

ALPHA, GAMMA, LAMBDA = ef.ALPHA, ef.GAMMA, ef.LAMBDA

def test_record_time_decorator():
    def sample_function(x):
        start_time = time.time()
        ef._record_time('dispersion_penalty', time.time() - start_time)
        return x * x
    
    for _ in range(ef.TIMES_CALLED):
        sample_function(3)
    
    result = sample_function(3)
    assert result == 9, "Function should return square of input"

def test_contains_cached():
    container = cube()
    point = Point(0, 0, 0)
    
    # Clear cache before test
    ef._CONTAINS_CACHE.clear()
    
    # First call should compute and cache
    result1 = ef._contains_cached(container, point)
    assert isinstance(result1, bool), "Should return a boolean"
    assert len(ef._CONTAINS_CACHE) == 1, "Should cache the result"
    
    # Second call should return cached value
    result2 = ef._contains_cached(container, point)
    assert result1 == result2, "Cached result should match original"
    assert len(ef._CONTAINS_CACHE) == 1, "Cache size should not increase"

def test_contains_cached_different_points():
    container = cube()
    point1 = Point(0, 0, 0)
    point2 = Point(0.25, 0.25, 0.25)
    
    ef._CONTAINS_CACHE.clear()
    
    result1 = ef._contains_cached(container, point1)
    result2 = ef._contains_cached(container, point2)
    assert len(ef._CONTAINS_CACHE) == 2, "Should cache both points separately"

def test_contains_cached_rounding():
    container = cube()
    point1 = Point(0.0000001, 0.0000001, 0.0000001)
    point2 = Point(0.0000002, 0.0000002, 0.0000002)
    
    ef._CONTAINS_CACHE.clear()
    
    result1 = ef._contains_cached(container, point1)
    result2 = ef._contains_cached(container, point2)
    assert len(ef._CONTAINS_CACHE) == 1, "Points should be rounded to same cache key"
    assert result1 == result2, "Rounded points should have same cached result"

def test_contains_cached_outside():
    container = cube()
    point = Point(1.5, 1.5, 1.5)
    
    ef._CONTAINS_CACHE.clear()
    
    result = ef._contains_cached(container, point)
    assert result == False, "Point outside container should return False"
    assert len(ef._CONTAINS_CACHE) == 1, "Should cache outside point"

def test_contains_cached_decimals_parameter():
    container = cube()
    point1 = Point(0.123456789, 0.123456789, 0.123456789)
    point2 = Point(0.123456999, 0.123456999, 0.123456999)
    
    ef._CONTAINS_CACHE.clear()
    
    _ = ef._contains_cached(container, point1, decimals=6)
    _ = ef._contains_cached(container, point2, decimals=6)
    assert len(ef._CONTAINS_CACHE) == 1, "Points should round to same key with decimals=6"
    
    ef._CONTAINS_CACHE.clear()
    _ = ef._contains_cached(container, point1, decimals=8)
    _ = ef._contains_cached(container, point2, decimals=8)
    assert len(ef._CONTAINS_CACHE) == 2, "Points should differ with decimals=8"

def test_get_verts():
    container = cube()
    verts = ef._get_verts(container)
    assert isinstance(verts, np.ndarray), "Verts should be a numpy array"
    assert verts.shape[1] == 3, "Verts should have shape (n, 3)"

def test_get_edges_bbox_and_caching():
    container = cube()
    emin, emax, edges = ef._get_edges_bbox(container)
    assert isinstance(edges, list), "Edges should be a list"
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges), "Each edge should be a tuple of two points"
    assert isinstance((emin, emax), tuple) and len((emin, emax)) == 2, "BBox should be a tuple of (min_point, max_point)"
    
    # Test the caching
    cached_emin, cached_emax, cached_edges = ef._get_edges_bbox(container)
    assert np.array_equal(emin, cached_emin) and np.array_equal(emax, cached_emax) and edges == cached_edges, "Cached values should match original"

def test_lij():
    point1 = Point(0, 0, 0)
    point2 = Point(1, 1, 1)

    assert ef.l_ij(point1, point2, D=1.0) == 2 - np.sqrt(3), "Points are less than 2D apart, penalty should be 2 - sqrt(3)"

def test_lij_far_apart():
    point1 = Point(0, 0, 0)
    point2 = Point(10, 10, 10)
    assert ef.l_ij(point1, point2, D=1.0) == 0, "Points far apart should have zero penalty"

def test_lij_within_threshold():
    point1 = Point(0, 0, 0)
    point2 = Point(0.5, 0, 0)
    assert ef.l_ij(point1, point2, D=1.0) > 0, "Points within 2D should have positive penalty"

def test_Oi0_inside_container():
    container = cube()
    point = Point(0, 0, 0)
    D = 0.1
    result = ef.Oi0(point, D, container)
    assert result == 0, "Penalty should be non-negative"

def test_Oi0_outside_container():
    container = cube()
    point = Point(1.0, 1.0, 1.0)
    D = 0.1
    result = ef.Oi0(point, D, container)
    assert result > 0, "Point outside container should have positive penalty"

def test_dispersion_penalty_single_point():
    points = np.array([[0, 0, 0]])
    D = 0.5
    assert ef.dispersion_penalty(points, D) == 0, "Single point should have zero dispersion penalty"

def test_dispersion_penalty_multiple_points():
    points = np.array([[0, 0, 0], [1, 1, 1]])
    D = 0.5
    result = ef.dispersion_penalty(points, D)
    assert result == 0, "Far points should have zero dispersion penalty"

def test_container_penalty():
    container = cube()
    points = np.array([[0.4, 0, 0], [0, 0.4, 0]])
    D = 0.1
    result = ef.container_penalty(points, D, container)
    assert result >= 0, "Container penalty should be non-negative"

def test_E_cube():
    container = cube()
    points = np.array([[-0.49, 0, 0]])
    D = 0.499

    assert container.contains(Point.from_array(points[0])), "Point should be inside the cube"
    assert ef.dispersion_penalty(points, D) == 0, "Single point should have zero dispersion penalty"
    assert ef.container_penalty(points, D, container) > 0, "Point near boundary should have positive container penalty"
    assert ef.E(points, D, container) > 1e-2

def test_E_multiple_points():
    container = cube()
    points = np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]])
    D = 0.1
    result = ef.E(points, D, container)
    assert result == 0, "Configuration should have zero energy"

def test_point_energy():
    container = cube()
    points = [Point(0.1, 0.1, 0.1), Point(-0.1, -0.1, -0.1)]
    D = 0.2
    result = ef.point_energy(points, 0, D, container)
    assert result >= 0, "Point energy should be non-negative"

def test_vacancy_energy():
    container = cube()
    points = [Point(0.1, 0.1, 0.1)]
    vacancy = Point(0.2, 0.2, 0.2)
    D = 0.1
    result = ef.vacancy_energy(points, vacancy, D, container)
    assert result >= 0, "Vacancy energy should be non-negative"

def test_phi():
    container = cube()
    points = np.array([[0.1, 0.1, 0.1]])
    D = 0.2
    mu = 1.0
    result = ef.phi(points, D, mu, container)
    assert isinstance(result, (int, float, np.number)), "Phi should return a scalar"

def test_select_high_energy_points():
    container = cube()
    points = [Point(0.1, 0.1, 0.1), Point(-0.1, -0.1, -0.1), Point(0, 0, 0)]
    D = 0.1
    Q = 2
    result = ef.select_high_energy_points(points, D, container, Q)
    assert len(result) == Q, "Should return Q points"
    assert all(isinstance(i, (int, np.integer)) for i in result), "Should return indices"

def test_find_vacancy_sites():
    container = cube()
    points = [Point(0.1, 0.1, 0.1)]
    D = 0.1
    Q = 3
    result = ef.find_vacancy_sites(points, D, container, Q)
    assert len(result) == Q, "Should return Q vacancy sites"
    assert all(isinstance(p, np.ndarray) for p in result), "Should return numpy.ndarray objects"

def test_E_reshape():
    container = cube()
    points = np.array([0.1, 0.1, 0.1])
    D = 0.1
    result = ef.E(points, D, container)
    assert isinstance(result, (int, float, np.number)), "Should return a scalar"

def test_container_penalty_cube():
    container = cube()
    points = np.array([[0.6, 0, 0], [0, 0.6, 0]])
    D = 0.1
    result = ef.container_penalty(points, D, container)
    assert result > 0, "Points outside cube should have positive container penalty"
    
def test_container_penalty_tetrahedron():
    from tsgo.container.sample_polyhedron import triangular_pyramid
    container = triangular_pyramid()
    points = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])
    D = 1.0
    result = ef.container_penalty(points, D, container)
    assert result > 0, "Points outside tetrahedron should have positive container penalty"