from src.point import Point
from src.container import Container
from src.sample_polyhedron import cube, h_box
from src.metric_enums import MetricType
import src.energy_functions as ef
from random import random
import numpy as np

ALPHA, GAMMA, LAMBDA = ef.ALPHA, ef.GAMMA, ef.LAMBDA

def test_lij():
    point1 = Point(0, 0, 0)
    point2 = Point(1, 1, 1)

    assert ef.l_ij(point1, point2, D=1.0) == 2 - np.sqrt(3), "Points are less than 2D apart, penalty should be 2 - sqrt(3)"

def test_E_cube():
    container = cube(radius=0.5)
    points = np.array([[-0.49, 0, 0]])
    D = 0.499
    metric = MetricType['EUCLIDEAN']

    assert container.contains(Point.from_array(points[0])), "Point should be inside the cube"
    assert ef.dispersion_penalty(points, D) == 0, "Single point should have zero dispersion penalty"
    assert ef.container_penalty(points, D, container) > 0, "Point near boundary should have positive container penalty"
    assert ef.E(points, D, container) > 1e-2
