from src.container import Container
from src.sample_polygons import square, h_shaped_box, v_box

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
