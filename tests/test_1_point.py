from src.point import Point

point1 = Point(1, 2, 3)
point2 = Point(4, 5, 6)
point3 = Point(2, 2, 3)
point4 = Point(2, 3, -1)

def test_dist_to():
    assert point1.dist_to(point2) == (9 + 9 + 9) ** 0.5
    assert point1.dist_to(point3) == 1.0
    assert point1.dist_to(point4) == (1 + 1 + 16) ** 0.5
    assert point2.dist_to(point3) == (4 + 9 + 9) ** 0.5
    assert point2.dist_to(point4) == (4 + 4 + 49) ** 0.5
    assert point3.dist_to(point4) == (0 + 1 + 16) ** 0.5