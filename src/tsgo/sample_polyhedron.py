from tsgo.point import Point
from tsgo.face import Face
from tsgo.container import Container

import numpy as np

# INWARD NORMALS
# X POSITIVE is RIGHT
# Y POSITIVE is BACK
# Z POSITIVE is UP

def cube(radius: float = 0.5):
    return Container([
        Face([Point(radius, -radius, radius), Point(radius, radius, radius), Point(radius, radius, -radius), Point(radius, -radius, -radius)], [-1, 0, 0], "Right Face of Cube"),
        Face([Point(-radius, -radius, -radius), Point(-radius, -radius, radius), Point(-radius, radius, radius), Point(-radius, radius, -radius)], [1, 0, 0], "Left Face of Cube"),
        Face([Point(-radius, radius, radius), Point(radius, radius, radius), Point(radius, -radius, radius), Point(-radius, -radius, radius)], [0, 0, -1], "Top Face of Cube"),
        Face([Point(-radius, -radius, -radius), Point(radius, -radius, -radius), Point(radius, radius, -radius), Point(-radius, radius, -radius)], [0, 0, 1], "Bottom Face of Cube"),
        Face([Point(-radius, -radius, radius), Point(radius, -radius, radius), Point(radius, -radius, -radius), Point(-radius, -radius, -radius)], [0, 1, 0], "Front Face of Cube"),
        Face([Point(-radius, radius, -radius), Point(radius, radius, -radius), Point(radius, radius, radius), Point(-radius, radius, radius)], [0, -1, 0], "Back Face of Cube")
    ], description=f"cube with radius {radius}")

def h_box():
    return Container([
        # Combined Front Face (Left, Middle, Right)
        Face([
            Point(0, 0, 0), Point(1, 0, 0), Point(1, 0, 1), Point(2, 0, 1),
            Point(2, 0, 0), Point(3, 0, 0), Point(3, 0, 3), Point(2, 0, 3),
            Point(2, 0, 2), Point(1, 0, 2), Point(1, 0, 3), Point(0, 0, 3),
        ], [0, 1, 0], "Combined Front Face"),
        # Combined Back Face (Left, Middle, Right)
        Face([
            Point(0, 1, 0), Point(1, 1, 0), Point(1, 1, 1), Point(2, 1, 1),
            Point(2, 1, 0), Point(3, 1, 0), Point(3, 1, 3), Point(2, 1, 3),
            Point(2, 1, 2), Point(1, 1, 2), Point(1, 1, 3), Point(0, 1, 3),
        ], [0, -1, 0], "Combined Back Face"),
        Face([Point(0, 0, 0), Point(0, 1, 0), Point(0, 1, 3), Point(0, 0, 3)], [1, 0, 0], "Left Wall"),
        Face([Point(3, 0, 0), Point(3, 1, 0), Point(3, 1, 3), Point(3, 0, 3)], [-1, 0, 0], "Right Wall"),
        Face([Point(0, 0, 3), Point(0, 1, 3), Point(1, 1, 3), Point(1, 0, 3)], [0, 0, -1], "Top Left Face"),
        Face([Point(0, 0, 0), Point(0, 1, 0), Point(1, 1, 0), Point(1, 0, 0)], [0, 0, 1], "Bottom Left Face"),
        Face([Point(2, 0, 3), Point(2, 1, 3), Point(3, 1, 3), Point(3, 0, 3)], [0, 0, -1], "Top Right Face"),
        Face([Point(2, 0, 0), Point(2, 1, 0), Point(3, 1, 0), Point(3, 0, 0)], [0, 0, 1], "Bottom Right Face"),
        Face([Point(1, 0, 2), Point(1, 1, 2), Point(1, 1, 3), Point(1, 0, 3)], [-1, 0, 0], "Inner Left Top Face"),
        Face([Point(1, 0, 0), Point(1, 1, 0), Point(1, 1, 1), Point(1, 0, 1)], [-1, 0, 0], "Inner Left Bottom Face"),
        Face([Point(2, 0, 2), Point(2, 1, 2), Point(2, 1, 3), Point(2, 0, 3)], [1, 0, 0], "Inner Right Top Face"),
        Face([Point(2, 0, 0), Point(2, 1, 0), Point(2, 1, 1), Point(2, 0, 1)], [1, 0, 0], "Inner Right Bottom Face"),
        Face([Point(1, 0, 1), Point(2, 0, 1), Point(2, 1, 1), Point(1, 1, 1)], [0, 0, 1], "Inner Middle Bottom Face"),
        Face([Point(1, 0, 2), Point(2, 0, 2), Point(2, 1, 2), Point(1, 1, 2)], [0, 0, -1], "Inner Middle Top Face")
    ], description="h-box")
    
def triangular_pyramid():
    return Container([
        Face([Point(0, 0, 0), Point(1, 0, 0), Point(0.5, np.sqrt(3)/2, 0)], [0, 0, 1], "Base Face"),
        Face([Point(0, 0, 0), Point(1, 0, 0), Point(0.5, np.sqrt(3)/6, np.sqrt(6)/3)], [0, np.sqrt(3)/3, -np.sqrt(6)/3], "Side Face 1"),
        Face([Point(1, 0, 0), Point(0.5, np.sqrt(3)/2, 0), Point(0.5, np.sqrt(3)/6, np.sqrt(6)/3)], [-np.sqrt(3)/2, -np.sqrt(3)/6, -np.sqrt(6)/3], "Side Face 2"),
        Face([Point(0.5, np.sqrt(3)/2, 0), Point(0, 0, 0), Point(0.5, np.sqrt(3)/6, np.sqrt(6)/3)], [np.sqrt(3)/2, -np.sqrt(3)/6, -np.sqrt(6)/3], "Side Face 3")
    ], description="tetrahedron")
    
def pointy_star():
    return Container([
        Face([Point(0, 0, 0), Point(1, -1, 4), Point(-1, -1, 4)], [0, 1, 0], "Bottom Spike Front"),
        Face([Point(0, 0, 0), Point(1, -1, 4), Point(1, 1, 4)], [-1, 0, 0], "Bottom Spike Right"),
        Face([Point(0, 0, 0), Point(1, 1, 4), Point(-1, 1, 4)], [0, -1, 0], "Bottom Spike Back"),
        Face([Point(0, 0, 0), Point(-1, 1, 4), Point(-1, -1, 4)], [1, 0, 0], "Bottom Spike Left"),
        
        Face([Point(1, -1, 4), Point(5, 0, 5), Point(1, -1, 6)], [0, 1, 0], "Right Spike Front"),
        Face([Point(1, -1, 6), Point(5, 0, 5), Point(1, 1, 6)], [0, 0, -1], "Right Spike Top"),
        Face([Point(1, 1, 6), Point(5, 0, 5), Point(1, 1, 4)], [0, -1, 0], "Right Spike Back"),
        Face([Point(1, 1, 4), Point(5, 0, 5), Point(1, -1, 4)], [0, 0, 1], "Right Spike Bottom"),
        
        Face([Point(-1, -1, 4), Point(-5, 0, 5), Point(-1, -1, 6)], [0, 1, 0], "Left Spike Front"),
        Face([Point(-1, -1, 6), Point(-5, 0, 5), Point(-1, 1, 6)], [0, 0, -1], "Left Spike Top"),
        Face([Point(-1, 1, 6), Point(-5, 0, 5), Point(-1, 1, 4)], [0, -1, 0], "Left Spike Back"),
        Face([Point(-1, 1, 4), Point(-5, 0, 5), Point(-1, -1, 4)], [0, 0, 1], "Left Spike Bottom"),
        
        Face([Point(1, 1, 6), Point(0, 0, 10), Point(-1, 1, 6)], [0, -1, 0], "Top Spike Back"),
        Face([Point(-1, 1, 6), Point(0, 0, 10), Point(-1, -1, 6)], [1, 0, 0], "Top Spike Left"),
        Face([Point(-1, -1, 6), Point(0, 0, 10), Point(1, -1, 6)], [0, 1, 0], "Top Spike Front"),
        Face([Point(1, -1, 6), Point(0, 0, 10), Point(1, 1, 6)], [-1, 0, 0], "Top Spike Right"),
        
        Face([Point(1, -1, 4), Point(0, -5, 5), Point(-1, -1, 4)], [0, 0, 1], "Front Spike Bottom"),
        Face([Point(-1, -1, 4), Point(0, -5, 5), Point(-1, -1, 6)], [1, 0, 0], "Front Spike Left"),
        Face([Point(-1, -1, 6), Point(0, -5, 5), Point(1, -1, 6)], [0, 0, -1], "Front Spike Top"),
        Face([Point(1, -1, 6), Point(0, -5, 5), Point(1, -1, 4)], [-1, 0, 0], "Front Spike Right"),
        
        Face([Point(-1, 1, 4), Point(0, 5, 5), Point(1, 1, 4)], [0, 0, 1], "Back Spike Bottom"),
        Face([Point(1, 1, 4), Point(0, 5, 5), Point(1, 1, 6)], [-1, 0, 0], "Back Spike Right"),
        Face([Point(1, 1, 6), Point(0, 5, 5), Point(-1, 1, 6)], [0, 0, -1], "Back Spike Top"),
        Face([Point(-1, 1, 6), Point(0, 5, 5), Point(-1, 1, 4)], [1, 0, 0], "Back Spike Left")
    ], description="pointy-star")