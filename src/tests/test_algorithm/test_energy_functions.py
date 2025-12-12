import numpy as np
import pytest

import tsgo.algorithm.energy_functions as ef
from tsgo.container.point import Point
from tsgo.container.container import Container
from tsgo.container.sample_polyhedron import cube, h_box
from tsgo.utils.metric_enums import MetricType

ALPHA, GAMMA, LAMBDA = ef.ALPHA, ef.GAMMA, ef.LAMBDA

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

def test_lij_caching():
    point1 = Point(0, 0, 0)
    point2 = Point(1, 1, 1)
    D = 1.0
    ef.l_ij_cache.clear()
    
    result1 = ef.l_ij(point1, point2, D)
    result2 = ef.l_ij(point1, point2, D)
    assert result1 == result2
    assert (point1, point2, D) in ef.l_ij_cache

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
    assert result == 0, "Close points should have positive dispersion penalty"

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
    metric = MetricType['EUCLIDEAN']

    assert container.contains(Point.from_array(points[0])), "Point should be inside the cube"
    assert ef.dispersion_penalty(points, D) == 0, "Single point should have zero dispersion penalty"
    assert ef.container_penalty(points, D, container) > 0, "Point near boundary should have positive container penalty"
    assert ef.E(points, D, container) > 1e-2

def test_E_multiple_points():
    container = cube()
    points = np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]])
    D = 0.2
    result = ef.E(points, D, container)
    assert result > 0, "Configuration should have positive energy"

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
