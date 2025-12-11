from src.point import Point
from src.face import Face
from src.container import Container
from src.sample_polyhedron import cube, h_box

cube = cube()
h_box = h_box()

def test_point_on_segment():
    assert Container._point_on_segment(Point(1, -0.5, 1), (Point(1, -1, 1), Point(1, 1, 1))) == True
    assert Container._point_on_segment(Point(1, -0.5, 1), (Point(1, 1, 1), Point(1, -1, 1))) == True
    assert Container._point_on_segment(Point(1, -0.5, 1), (Point(1, -1, 1), Point(1, 0, 0))) == False

def test_contains_square_box():
    assert cube.contains(Point(0, 0, 0)) == True, "Point (0, 0, 0) should be inside the cube."
    assert cube.contains(Point(0.5, 0.5, 0.5)) == True, "Point (0.5, 0.5, 0.5) should be inside the cube."
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

def test_active_footpoints():
    point = Point(0, 0, 0)
    expected_active_footpoints = [Point(1, 0, 0), Point(-1, 0, 0), Point(0, 0, 1), Point(0, 0, -1), Point(0, 1, 0), Point(0, -1, 0)]
    assert cube.get_active_footpoints(point).sort() == expected_active_footpoints.sort(), f"Active footpoints for {point} should be {expected_active_footpoints}."

def test_area_square_box():
    assert cube.volume() == 8.0, "Volume of the cube should be 8.0."
    
def test_area_h_box():
    assert h_box.volume() == 7.0, "Volume of the horizontal box should be 7.0."
    
def point_in_polygon():
    face = Face(vertices=[Point(0, 0, 0), Point(5, 5, 0), Point(0, 2, 0), Point(-5, 5, 0)], normal=[0, 0, 1], description="v-shaped face")
    point1 = Point(0, 1, 0)
    point2 = Point(0, 3, 0)
    assert cube._point_in_polygon(point1, face) == True, f"Point {point1} should be inside face {face.description}."
    assert cube._point_in_polygon(point2, face) == False, f"Point {point2} should be outside face {face.description}."