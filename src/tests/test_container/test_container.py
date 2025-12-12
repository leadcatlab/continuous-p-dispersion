import copy
import numpy as np
import pytest

from tsgo.container.container import Container
from tsgo.container.face import Face
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import cube as cube_sample, h_box as h_box_sample

cube = cube_sample(radius=1)
h_box = h_box_sample()

def test_point_on_segment():
    assert Container._point_on_segment(Point(1, -0.5, 1), (Point(1, -1, 1), Point(1, 1, 1))) == True
    assert Container._point_on_segment(Point(1, -0.5, 1), (Point(1, 1, 1), Point(1, -1, 1))) == True
    assert Container._point_on_segment(Point(1, -0.5, 1), (Point(1, -1, 1), Point(1, 0, 0))) == False

def test_contains_square_box():
    assert cube.contains((0, 0, 0)) == True, "Point (0, 0, 0) should be inside the cube."
    assert cube.contains(np.array([0.5, 0.5, 0.5])) == True, "Point (0.5, 0.5, 0.5) should be inside the cube."
    assert cube.contains(Point(-0.5, -0.5, -0.5)) == True, "Point (-0.5, -0.5, -0.5) should be inside the cube."
    assert cube.contains(Point(-0.255, -1.12355, 0.53204)) == False, "Point (-0.255, -1.12355, 0.53204) should be outside the cube."
    assert cube.contains(Point(-0.255, -0.12355, 0.53204)) == True, "Point (-0.255, -0.12355, 0.53204) should be inside the cube."
    assert cube.contains(Point(1, -1, 1)) == True, "Point (1, -1, 1) should be on the surface of the cube."
    assert cube.contains(Point(1, -0.5, 0.5)) == True, "Point (1, -0.5, 0.5) should be on the surface of the cube."
    assert cube.contains(Point(1, -0.5, 1)) == True, "Point (1, -0.5, 1) should be on the surface of the cube."
    
def test_contains_h_box():
    assert h_box.contains(Point(0.5, 0.5, 1.5)) == True, "Point (0.5, 0.5, 1.5) should be inside the horizontal box."
    assert h_box.contains(Point(0.5, 0.5, 3)) == True, "Point (0.5, 0.5, 3) should be on the surface of the horizontal box."
    assert h_box.contains(Point(1.5, 0.5, 1.5)) == True, "Point (1.5, 0.5, 1.5) should be inside the central box."
    assert h_box.contains(Point(2.5, 0.5, 1.5)) == True, "Point (2.5, 0.5, 1.5) should be inside the horizontal box."
    assert h_box.contains(Point(2.5, 0.5, 3)) == True, "Point (2.5, 0.5, 3) should be on the surface of the horizontal box."
    assert h_box.contains(Point(0, 0, 0)) == True, "Point (0, 0, 0) should be on the surface of the horizontal box."
    
def test_active_plane_square_box():
    point = Point(0.5, 0.5, 0.5)
    for face in cube.faces:
        assert cube._is_active_plane(face, point) == True, f"Face {face.description} should be an active plane for point {point}."
        
    point = Point(0, 0, -10)
    face = cube.faces[1]
    assert cube._is_active_plane(face, point) == False, f"Face {face.description} should not be an active plane for point {point}."

def test_point_on_segment():
    assert Container._point_on_segment(Point(1, -0.5, 1), (Point(1, -1, 1), Point(1, 1, 1))) == True
    assert Container._point_on_segment(Point(1, -0.5, 1), (Point(1, 1, 1), Point(1, -1, 1))) == True
    assert Container._point_on_segment(Point(1, -0.5, 1), (Point(1, -1, 1), Point(1, 0, 0))) == False

def test_point_on_segment_edge_cases():
    assert Container._point_on_segment(Point(1, -1, 1), (Point(1, -1, 1), Point(1, 1, 1))) == False
    assert Container._point_on_segment(Point(1, 1, 1), (Point(1, -1, 1), Point(1, 1, 1))) == True

def test_contains_square_box():
    assert cube.contains((0, 0, 0)) == True, "Point (0, 0, 0) should be inside the cube."
    assert cube.contains(np.array([0.5, 0.5, 0.5])) == True, "Point (0.5, 0.5, 0.5) should be inside the cube."
    assert cube.contains(Point(-0.5, -0.5, -0.5)) == True, "Point (-0.5, -0.5, -0.5) should be inside the cube."
    assert cube.contains(Point(-0.255, -1.12355, 0.53204)) == False, "Point (-0.255, -1.12355, 0.53204) should be outside the cube."
    assert cube.contains(Point(-0.255, -0.12355, 0.53204)) == True, "Point (-0.255, -0.12355, 0.53204) should be inside the cube."
    assert cube.contains(Point(1, -1, 1)) == True, "Point (1, -1, 1) should be on the surface of the cube."
    assert cube.contains(Point(1, -0.5, 0.5)) == True, "Point (1, -0.5, 0.5) should be on the surface of the cube."
    assert cube.contains(Point(1, -0.5, 1)) == True, "Point (1, -0.5, 1) should be on the surface of the cube."
    
def test_contains_h_box():
    assert h_box.contains(Point(0.5, 0.5, 1.5)) == True, "Point (0.5, 0.5, 1.5) should be inside the horizontal box."
    assert h_box.contains(Point(0.5, 0.5, 3)) == True, "Point (0.5, 0.5, 3) should be on the surface of the horizontal box."
    assert h_box.contains(Point(1.5, 0.5, 1.5)) == True, "Point (1.5, 0.5, 1.5) should be inside the central box."
    assert h_box.contains(Point(2.5, 0.5, 1.5)) == True, "Point (2.5, 0.5, 1.5) should be inside the horizontal box."
    assert h_box.contains(Point(2.5, 0.5, 3)) == True, "Point (2.5, 0.5, 3) should be on the surface of the horizontal box."
    assert h_box.contains(Point(0, 0, 0)) == True, "Point (0, 0, 0) should be on the surface of the horizontal box."

def test_contains_outside_points():
    assert cube.contains(Point(2, 0, 0)) == False, "Point (2, 0, 0) should be outside the cube."
    assert cube.contains(Point(-2, 0, 0)) == False, "Point (-2, 0, 0) should be outside the cube."
    assert cube.contains(Point(0, 2, 0)) == False, "Point (0, 2, 0) should be outside the cube."

def test_active_plane_square_box():
    point = Point(0.5, 0.5, 0.5)
    for face in cube.faces:
        assert cube._is_active_plane(face, point) == True, f"Face {face.description} should be an active plane for point {point}."
        
    point = Point(0, 0, -10)
    face = cube.faces[1]
    assert cube._is_active_plane(face, point) == False, f"Face {face.description} should not be an active plane for point {point}."

def test_active_plane_outside():
    point = Point(10, 10, 10)
    for face in cube.faces:
        is_active = cube._is_active_plane(face, point)
        assert isinstance(is_active, bool)

def test_active_footpoints():
    point = Point(0, 0, 0)
    footpoints = cube.get_active_footpoints(point)
    assert len(footpoints) == 6, "Center point should have 6 active footpoints."
    assert all(isinstance(fp, Point) for fp in footpoints)

def test_active_footpoints_caching():
    point = Point(0.5, 0.5, 0.5)
    footpoints1 = cube.get_active_footpoints(point)
    footpoints2 = cube.get_active_footpoints(point)
    assert footpoints1 == footpoints2, "Cached results should match."

def test_active_footpoints_outside_point():
    point = Point(5, 5, 5)
    footpoints = cube.get_active_footpoints(point)
    assert len(footpoints) <= 6, "Outside point should have fewer active footpoints."

def test_bounding_box():
    bbox = cube._bounding_box()
    assert bbox["min_x"] == -1 and bbox["max_x"] == 1
    assert bbox["min_y"] == -1 and bbox["max_y"] == 1
    assert bbox["min_z"] == -1 and bbox["max_z"] == 1

def test_volume_square_box():
    assert cube.volume() == 8.0, "Volume of the cube should be 8.0."
    
def test_volume_h_box():
    assert h_box.volume() == 7.0, "Volume of the horizontal box should be 7.0."

def test_generate_initial_solution():
    solution = cube.generate_initial_solution(5)
    assert solution.shape == (5, 3), "Generated solution should have shape (5, 3)."
    for point in solution:
        assert cube.contains(point), f"Point {point} should be inside the cube."

def test_generate_initial_solution_different_p():
    for p in [1, 3, 10, 20]:
        solution = cube.generate_initial_solution(p)
        assert solution.shape == (p, 3)
        assert all(cube.contains(point) for point in solution)

def test_get_edges():
    edges = cube._get_edges()
    assert len(edges) > 0, "Cube should have edges."
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges)

def test_get_vertices():
    vertices = cube._get_vertices()
    assert len(vertices) == 8, "Cube should have 8 vertices."
    assert all(isinstance(v, Point) for v in vertices)

def test_contains_with_different_input_types():
    point_list = [0, 0, 0]
    point_array = np.array([0, 0, 0])
    point_obj = Point(0, 0, 0)
    assert cube.contains(point_list) == cube.contains(point_array) == cube.contains(point_obj)
