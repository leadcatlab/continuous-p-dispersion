from tsgo.container.point import Point
from tsgo.container.face import Face
from tsgo.container.sample_polyhedron import cube, h_box

cube = cube(radius=1)
h_box = h_box() 

def test_half_space_1():
    point = Point(0, 0, 0)
    for face in cube.faces:
        assert face.half_space(point) > 0, f"Face {face.description} should have a positive half-space value."
        
def test_half_space_2():
    point = Point(1, 1, 1)
    right_face = cube.faces[0]
    assert right_face.half_space(point) == 0, "Right Face should have a half-space value of 0 for point (1,1,1)."
    
    top_face = cube.faces[2]
    assert top_face.half_space(point) == 0, "Top Face should have a half-space value of 0 for point (1,1,1)."

    bottom_face = cube.faces[3]
    assert bottom_face.half_space(point) > 0, "Bottom Face should have a half-space value of 0 for point (1,1,1)."
    
    
def test_face_contains_1():
    face = Face(vertices=[Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0)], normal=[0, 0, 1], description="Test Face")
    assert face.contains(Point(0.1, 0.1, 0)), "Face should contain the point (0.1, 0.1, 0)."
    assert not face.contains(Point(0.1, 0.1, 1)), "Face should not contain the point (0.1, 0.1, 1)."
    assert not face.contains(Point(1, 1, 1)), "Face should not contain the point (1, 1, 1)."
    
    
def test_face_contains_2():
    face = Face(vertices=[Point(0, 0, 0), Point(1, 1, 0), Point(5, 5, 5)], normal=[1, -1, 0])
    assert face.contains(Point(0.5, 0.5, 0)), "Face should contain the point (0.5, 0.5, 0)."
    assert not face.contains(Point(0.5, 0.5, 1)), "Face should not contain the point (0.5, 0.5, 1)."
    assert face.contains(Point(5, 5, 5)), "Face should contain the point (5, 5, 5)."
    assert not face.contains(Point(5, 5, 6)), "Face should not contain the point (5, 5, 6)."
    assert not face.contains(Point(0, 0, 10)), "Face should not contain the point (0, 0, 10)."

def test_face_contains_non_planar_point():
    """Test contains with point not on the plane."""
    face = Face(vertices=[Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0)], normal=[0, 0, 1])
    assert not face.contains(Point(0.5, 0.5, 0.1)), "Point above plane should not be contained."
    assert not face.contains(Point(0.5, 0.5, -0.1)), "Point below plane should not be contained."

def test_face_contains_outside_polygon():
    """Test contains with point on plane but outside polygon boundary."""
    face = Face(vertices=[Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0)], normal=[0, 0, 1])
    assert not face.contains(Point(2, 2, 0)), "Point outside polygon should not be contained."
    assert not face.contains(Point(-1, -1, 0)), "Point outside polygon should not be contained."


def test_face_contains_on_vertex():
    """Test contains with point exactly on a vertex."""
    face = Face(vertices=[Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0)], normal=[0, 0, 1])
    assert face.contains(Point(0, 0, 0)), "Point on vertex should be contained."
    assert face.contains(Point(1, 0, 0)), "Point on vertex should be contained."
    assert face.contains(Point(1, 1, 0)), "Point on vertex should be contained."


def test_face_contains_on_edge():
    """Test contains with point exactly on an edge."""
    face = Face(vertices=[Point(0, 0, 0), Point(2, 0, 0), Point(0, 2, 0)], normal=[0, 0, 1])
    assert face.contains(Point(1, 0, 0)), "Point on edge should be contained."
    assert face.contains(Point(0, 1, 0)), "Point on edge should be contained."


def test_face_contains_interior():
    """Test contains with point in interior of polygon."""
    face = Face(vertices=[Point(0, 0, 0), Point(4, 0, 0), Point(4, 4, 0), Point(0, 4, 0)], normal=[0, 0, 1])
    assert face.contains(Point(2, 2, 0)), "Point in interior should be contained."
    assert face.contains(Point(1, 1, 0)), "Point in interior should be contained."
