from src.point import Point
from src.face import Face
from src.container import Container

def cube():
    return Container([
        Face([Point(1, -1, 1), Point(1, 1, 1), Point(1, 1, -1), Point(1, -1, -1)], [1, 0, 0], "Right Face of Cube"),
        Face([Point(-1, -1, -1), Point(-1, -1, 1), Point(-1, 1, 1), Point(-1, 1, -1)], [-1, 0, 0], "Left Face of Cube"),
        Face([Point(-1, 1, 1), Point(1, 1, 1), Point(1, -1, 1), Point(-1, -1, 1)], [0, 0, 1], "Top Face of Cube"),
        Face([Point(-1, -1, -1), Point(1, -1, -1), Point(1, 1, -1), Point(-1, 1, -1)], [0, 0, -1], "Bottom Face of Cube"),
        Face([Point(-1, -1, -1), Point(-1, -1, 1), Point(1, -1, 1), Point(1, -1, -1)], [0, -1, 0], "Front Face of Cube"),
        Face([Point(-1, 1, -1), Point(-1, 1, 1), Point(1, 1, 1), Point(1, 1, -1)], [0, 1, 0], "Back Face of Cube")
    ])
    
def h_box():
    return Container([
    Face([Point(0, 0, 0), Point(0, 0, 3), Point(1, 0, 3), Point(1, 0, 0)], [0, -1, 0], "Left Front Face"),
    Face([Point(0, 1, 0), Point(0, 1, 3), Point(1, 1, 3), Point(1, 1, 0)], [0, 1, 0], "Left Back Face"),
    Face([Point(2, 0, 0), Point(2, 0, 3), Point(3, 0, 3), Point(3, 0, 0)], [0, -1, 0], "Right Front Face"),
    Face([Point(2, 1, 0), Point(2, 1, 3), Point(3, 1, 3), Point(3, 1, 0)], [0, 1, 0], "Right Back Face"),
    Face([Point(1, 0, 1), Point(1, 0, 2), Point(2, 0, 2), Point(2, 0, 1)], [0, -1, 0], "Middle Front Face"),
    Face([Point(1, 1, 1), Point(1, 1, 2), Point(2, 1, 2), Point(2, 1, 1)], [0, 1, 0], "Middle Back Face"),
    Face([Point(0, 0, 0), Point(0, 0, 3), Point(0, 1, 3), Point(0, 1, 0)], [-1, 0, 0], "Left Wall"),
    Face([Point(3, 0, 0), Point(3, 0, 3), Point(3, 1, 3), Point(3, 1, 0)], [1, 0, 0], "Right Wall"),
    Face([Point(0, 0, 3), Point(1, 0, 3), Point(1, 1, 3), Point(0, 1, 3)], [0, 0, 1], "Top Left Face"),
    Face([Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0)], [0, 0, -1], "Bottom Left Face"),
    Face([Point(2, 0, 3), Point(3, 0, 3), Point(3, 1, 3), Point(2, 1, 3)], [0, 0, 1], "Top Right Face"),
    Face([Point(2, 0, 0), Point(3, 0, 0), Point(3, 1, 0), Point(2, 1, 0)], [0, 0, -1], "Bottom Right Face"),
    Face([Point(1, 0, 2), Point(1, 0, 3), Point(1, 1, 3), Point(1, 1, 2)], [1, 0, 0], "Inner Left Top Face"),
    Face([Point(1, 0, 0), Point(1, 0, 1), Point(1, 1, 1), Point(1, 1, 0)], [1, 0, 0], "Inner Left Bottom Face"),
    Face([Point(2, 0, 2), Point(2, 0, 3), Point(2, 1, 3), Point(2, 1, 2)], [-1, 0, 0], "Inner Right Top Face"),
    Face([Point(2, 0, 0), Point(2, 0, 1), Point(2, 1, 1), Point(2, 1, 0)], [-1, 0, 0], "Inner Right Bottom Face"),
    Face([Point(1, 0, 1), Point(1, 1, 1), Point(2, 1, 1), Point(2, 0, 1)], [0, 0, -1], "Inner Middle Bottom Face"),
    Face([Point(1, 0, 2), Point(1, 1, 2), Point(2, 1, 2), Point(2, 0, 2)], [0, 0, 1], "Inner Middle Top Face")
])