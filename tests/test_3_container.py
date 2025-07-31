from src.point import Point
from src.face import Face
from src.container import Container
from src.sample_polyhedron import cube, h_box

cube = cube()
h_box = h_box()

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
    assert h_box.contains(Point(0.5, 0.5, 1.5), num_rays=1000) == True, "Point (0.5, 0.5, 1.5) should be inside the horizontal box."
    assert h_box.contains(Point(0.5, 0.5, 3), num_rays=1000) == True, "Point (0.5, 0.5, 3) should be on the surface of the horizontal box."
    assert h_box.contains(Point(1.5, 0.5, 1.5), num_rays=1000) == True, "Point (1.5, 0.5, 1.5) should be inside the central box."
    assert h_box.contains(Point(2.5, 0.5, 1.5), num_rays=1000) == True, "Point (2.5, 0.5, 1.5) should be inside the horizontal box."
    assert h_box.contains(Point(2.5, 0.5, 3), num_rays=1000) == True, "Point (2.5, 0.5, 3) should be on the surface of the horizontal box."
    assert h_box.contains(Point(0, 0, 0), num_rays=1000) == True, "Point (0, 0, 0) should be on the surface of the horizontal box."
    
def test_active_plane_square_box():
    point = Point(0.5, 0.5, 0.5)
    for face in cube.faces:
        assert cube.is_active_plane(face, point) == True, f"Face {face.description} should be an active plane for point {point}."
        
    point = Point(0, 0, -10)
    face = cube.faces[1]
    assert cube.is_active_plane(face, point) == False, f"Face {face.description} should not be an active plane for point {point}."
    
def test_active_footpoint_square_box():
    point = Point(0.5, 0.5, 0.5)
    for face in cube.faces:
        assert cube.has_active_footpoint(point, face) == True, f"Face {face.description} should have an active footpoint for point {point}."
        
def test_area_square_box():
    assert cube.volume() == 8.0, "Volume of the cube should be 8.0."
    
def test_area_h_box():
    assert h_box.volume() == 7.0, "Volume of the horizontal box should be 7.0."