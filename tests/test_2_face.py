from src.point import Point
from src.face import Face
from src.container import Container
from src.sample_polyhedron import cube, h_box

cube = cube()
h_box = h_box()

def test_face_box():
    for face in cube.faces:
        assert face.area() == 4.0, f"Face {face.description} should have an area of 4.0."
        
def test_area_h_box():
    assert h_box.faces[0].area() == 3.0, "Left Front Face should have an area of 3.0."
    assert h_box.faces[1].area() == 3.0, "Left Back Face should have an area of 3.0."
    assert h_box.faces[2].area() == 3.0, "Right Front Face should have an area of 3.0."
    assert h_box.faces[3].area() == 3.0, "Right Back Face should have an area of 3.0."
    assert h_box.faces[4].area() == 1.0, "Middle Front Face should have an area of 1.0."
    assert h_box.faces[5].area() == 1.0, "Middle Back Face should have an area of 1.0."
    assert h_box.faces[6].area() == 3.0, "Left Wall should have an area of 3.0."
    assert h_box.faces[7].area() == 3.0, "Right Wall should have an area of 3.0."
    assert h_box.faces[8].area() == 1.0, "Top Left Face should have an area of 1.0."
    assert h_box.faces[9].area() == 1.0, "Bottom Left Face should have an area of 1.0."
    assert h_box.faces[10].area() == 1.0, "Top Right Face should have an area of 1.0."
    assert h_box.faces[11].area() == 1.0, "Bottom Right Face should have an area of 1.0."
    assert h_box.faces[12].area() == 1.0, "Inner Left Top Face should have an area of 1.0."
    assert h_box.faces[13].area() == 1.0, "Inner Left Bottom Face should have an area of 1.0."
    assert h_box.faces[14].area() == 1.0, "Inner Right Top Face should have an area of 1.0."
    assert h_box.faces[15].area() == 1.0, "Inner Right Bottom Face should have an area of 1.0."
    assert h_box.faces[16].area() == 1.0, "Inner Middle Bottom Face should have an area of 1.0."
    assert h_box.faces[17].area() == 1.0, "Inner Middle Top Face should have an area of 1.0."
