from src.container import Container, is_counter_clockwise
from src.sample_polyhedron import square, h_container

square_container = square()
h_container = h_container()

def test_is_counter_clockwise():
    assert is_counter_clockwise([0, 0], [1, 1], [1, 0]) == False, "Counter-clockwise check failed."
    assert is_counter_clockwise([0, 0], [1, 0], [1, 1]) == True, "Counter-clockwise check failed."
    assert is_counter_clockwise([-15.3, 100.3], [-1.2, -5.5], [137.8, 0.44]) == True, "Counter-clockwise check failed."
    

def test_get_edges_square_box():
    edges = square_container.get_edges()
    assert len(edges) == 4, "Square box should have 4 edges."
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges), "Each edge should be a tuple of two vertices."
    print(edges)
    expected_edges = [([-1, -1], [1, -1]), ([1, -1], [1, 1]), ([1, 1], [-1, 1]), ([-1, 1], [-1, -1])]
    for edge in edges:
        expected_edges.remove(edge)
    assert len(expected_edges) == 0, "All expected edges should be present in the square box edges."

def test_get_edges_h_box():
    edges = h_container.get_edges()
    assert len(edges) == 12, "H-shaped box should have 12 edges."
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges), "Each edge should be a tuple of two vertices."
    expected_edges = [
        ([0, 0], [1, 0]), ([1, 0], [1, 1]), ([1, 1], [2, 1]), ([2, 1], [2, 0]),
        ([2, 0], [3, 0]), ([3, 0], [3, 3]), ([3, 3], [2, 3]), ([2, 3], [2, 2]),
        ([2, 2], [1, 2]), ([1, 2], [1, 3]), ([1, 3], [0, 3]), ([0, 3], [0, 0])
    ]
    for edge in edges:
        expected_edges.remove(edge)
    assert len(expected_edges) == 0, "All expected edges should be present in the H-shaped box edges."

def test_contains_square_box():
    assert square_container.contains([0, 0]) == True, "Point (0, 0) should be inside the square box."
    assert square_container.contains([0.5, 0.5]) == True, "Point (0.5, 0.5) should be inside the square box."
    assert square_container.contains([1, 1]) == True, "Point (1, 1) should be a corner of the square box."
    assert square_container.contains([1, 0.5]) == True, "Point (1, 0.5) should be on the boundary of the square box."
    assert square_container.contains([1.5, 1.5]) == False, "Point (1.5, 1.5) should be outside the square box."
    
def test_contains_h_box():
    assert h_container.contains([0, 0]) == True, "Point (0, 0) should be a corner of the horizontal box."
    assert h_container.contains([0.5, 0.5]) == True, "Point (0.5, 0.5) should be inside the horizontal box."
    assert h_container.contains([1, 1]) == True, "Point (1, 1) should be a corner of the horizontal box."
    assert h_container.contains([1, 0.5]) == True, "Point (1, 0.5) should be on the boundary of the horizontal box."
    assert h_container.contains([2, 2]) == True, "Point (2, 2) should be a corner of the horizontal box."
    assert h_container.contains([1.5, 1.5]) == True, "Point (1.5, 1.5) should be inside the horizontal box."
    assert h_container.contains([10, 10]) == False, "Point (10, 10) should be outside the horizontal box."
    assert h_container.contains([0, 4]) == False, "Point (0, 4) should be outside the horizontal box."

# @TODO: correct test for the active edges
def test_active_edges_square():
    point = [0.5, 0.5]
    
    # every edge should be active for the point inside the square
    for edge in square_container.edges:
        assert square_container.is_active_edge(point, edge) == True, f"Edge {edge} should be active for point {point}."
        
    point = [0, 0]

    # every edge should be active for the point on the corner of the square
    for edge in square_container.edges:
        assert square_container.is_active_edge(point, edge) == True, f"Edge {edge} should be active for point {point}."

    point = [1.5, 0]
    for edge in square_container.edges:
        a, b = edge
        if a == [1, -1] and b == [1, 1]:
            assert square_container.is_active_edge(point, edge) == True, f"Edge {edge} should be active for point {point}."
        else:
            assert square_container.is_active_edge(point, edge) == False, f"Edge {edge} should not be active for point {point}."
    
    
# @TODO: write a test for active footpoints    

def test_area_square_box():
    assert square_container.area() == 4.0, "Area of the square box should be 4.0."

def test_area_h_box():
    assert h_container.area() == 7.0, "Area of the horizontal box should be 7.0."