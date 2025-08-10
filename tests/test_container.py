from src.container import Container
from src.sample_polygons import square, h_shaped_box, v_box
from random import random
import numpy as np

square = square(side_length=3.0)
h_box = h_shaped_box()
v_box = v_box(diag_len=2.0)

def test_get_edges_square():
    square_edges = square.edges

    # Check if the number of edges is correct
    assert len(square_edges) == 4

    # Check if square edges are correct
    expected_square_edges = [
        ([0.0, 0.0], [3.0, 0.0]),
        ([3.0, 0.0], [3.0, 3.0]),
        ([3.0, 3.0], [0.0, 3.0]),
        ([0.0, 3.0], [0.0, 0.0])
    ]

    assert expected_square_edges == square_edges
    
def test_get_edges_h_box():
    h_box_edges = h_box.edges

    # Check if the number of edges is correct
    assert len(h_box_edges) == 12

    # Check if H-box edges are correct
    expected_h_box_edges = [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [1.0, 1.0]),
        ([1.0, 1.0], [2.0, 1.0]),
        ([2.0, 1.0], [2.0, 0.0]),
        ([2.0, 0.0], [3.0, 0.0]),
        ([3.0, 0.0], [3.0, 3.0]),
        ([3.0, 3.0], [2.0, 3.0]),
        ([2.0, 3.0], [2.0, 2.0]),
        ([2.0, 2.0], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, 3.0]),
        ([1.0, 3.0], [0.0, 3.0]),
        ([0.0, 3.0], [0.0, 0.0])
    ]

    assert expected_h_box_edges == h_box_edges

def test_get_edges_v_box():
    v_box_edges = v_box.edges

    # Check if the number of edges is correct
    assert len(v_box_edges) == 4

    # Check if V-box edges are correct
    expected_v_box_edges = [
        ([0.0, 0.0], [2.0, 2.0]),
        ([2.0, 2.0], [0.0, 1.0]),
        ([0.0, 1.0], [-2.0, 2.0]),
        ([-2.0, 2.0], [0.0, 0.0])
    ]

    assert expected_v_box_edges == v_box_edges

def test_point_on_segment():
    point = [0.0, 0.0]
    edge = ([-10.0, 0.0], [10.0, 0.0])
    assert square._point_on_segment(point, edge) == True
    
    point = [1.0, 1.0]
    edge = ([-10.0, 0.0], [10.0, 0.0])
    assert square._point_on_segment(point, edge) == False

    point = [2.0, 0.0]
    edge = ([0.0, 10.0], [10.0, 0.0])
    assert square._point_on_segment(point, edge) == False
    
def test_contains_square():
    for _ in range(100):
        x, y = random() * 4, random() * 4
        assert square.contains([x, y]) == (0 <= x <= 3 and 0 <= y <= 3)

def test_contains_h_box():
    
    def valid_x_y(x, y):
        if 0 <= x <= 3 and 0 <= y <= 3:
            if not (1 <= x <= 2 and (0 <= y <= 1 or 2 <= y <= 3)):
                return True
        return False

    for _ in range(100):
        x, y = random() * 4, random() * 4
        assert h_box.contains([x, y]) == valid_x_y(x, y)
        
def test_contains_v_box():
    eps = 1e-8
    points_to_test = [
        [0.0, 0.0], [0.0, 0.5], 
        [2.0 - eps, 2.0 - eps], [-2.0 + eps, 2.0 - eps], 
        [0.0, 1.0], [2.0 - eps, 2.0 - eps / 2]
    ]
    
    for point in points_to_test:
        assert v_box.contains(point) == True
        
    assert v_box.contains([0.0, 1.1]) == False
    assert v_box.contains([1.1, 0.0]) == False
