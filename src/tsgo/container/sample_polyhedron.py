from tsgo.container.point import Point
from tsgo.container.face import Face
from tsgo.container.container import Container

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
    ], description=f"cube")

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

def ribbed_cage():
    return Container([
        # Base Face
        Face([Point(0, 0, 0), Point(10, 0, 0), Point(10, 5, 0), Point(6, 5, 0), Point(6, 6, 0), Point(10, 6, 0), Point(10, 11, 0), Point(0, 11, 0), Point(0, 6, 0), Point(4, 6, 0), Point(4, 5, 0), Point(0, 5, 0)], [0, 0, 1], "Bottom Face", 
             holes=[
                [Point(1, 1, 0), Point(9, 1, 0), Point(9, 2, 0), Point(1, 2, 0)],  # Bottom Rib Hole
                [Point(1, 3, 0), Point(9, 3, 0), Point(9, 4, 0), Point(1, 4, 0)],  # Bottom Middle Rib Hole
                [Point(1, 7, 0), Point(9, 7, 0), Point(9, 8, 0), Point(1, 8, 0)],  # Top Middle Rib Hole
                [Point(1, 9, 0), Point(9, 9, 0), Point(9, 10, 0), Point(1, 10, 0)] # Top Rib Hole
            ]
        ),
        
        # Bottom Rib
        Face([Point(1, 1, 0), Point(1, 1, 1), Point(9, 1, 1), Point(9, 1, 0)], [0, -1, 0], "Bottom Rib Front Face"),
        Face([Point(1, 1, 0), Point(1, 1, 1), Point(1, 2, 1), Point(1, 2, 0)], [-1, 0, 0], "Bottom Rib Left Face"),
        Face([Point(1, 2, 0), Point(1, 2, 1), Point(9, 2, 1), Point(9, 2, 0)], [0, 1, 0], "Bottom Rib Back Face"),
        Face([Point(9, 1, 0), Point(9, 1, 1), Point(9, 2, 1), Point(9, 2, 0)], [1, 0, 0], "Bottom Rib Right Face"),
        Face([Point(1, 1, 1), Point(9, 1, 1), Point(9, 2, 1), Point(1, 2, 1)], [0, 0, 1], "Bottom Rib Top Face"),
        
        # Bottom Middle Rib
        Face([Point(1, 3, 0), Point(1, 3, 1), Point(9, 3, 1), Point(9, 3, 0)], [0, -1, 0], "Bottom Middle Rib Front Face"),
        Face([Point(1, 3, 0), Point(1, 3, 1), Point(1, 4, 1), Point(1, 4, 0)], [-1, 0, 0], "Bottom Middle Rib Left Face"),
        Face([Point(1, 4, 0), Point(1, 4, 1), Point(9, 4, 1), Point(9, 4, 0)], [0, 1, 0], "Bottom Middle Rib Back Face"),
        Face([Point(9, 3, 0), Point(9, 3, 1), Point(9, 4, 1), Point(9, 4, 0)], [1, 0, 0], "Bottom Middle Rib Right Face"),
        Face([Point(1, 3, 1), Point(9, 3, 1), Point(9, 4, 1), Point(1, 4, 1)], [0, 0, 1], "Bottom Middle Rib Top Face"),
        
        # Top Middle Rib
        Face([Point(1, 7, 0), Point(1, 7, 1), Point(9, 7, 1), Point(9, 7, 0)], [0, -1, 0], "Top Middle Rib Front Face"),
        Face([Point(1, 7, 0), Point(1, 7, 1), Point(1, 8, 1), Point(1, 8, 0)], [-1, 0, 0], "Top Middle Rib Left Face"),
        Face([Point(1, 8, 0), Point(1, 8, 1), Point(9, 8, 1), Point(9, 8, 0)], [0, 1, 0], "Top Middle Rib Back Face"),
        Face([Point(9, 7, 0), Point(9, 7, 1), Point(9, 8, 1), Point(9, 8, 0)], [1, 0, 0], "Top Middle Rib Right Face"),
        Face([Point(1, 7, 1), Point(9, 7, 1), Point(9, 8, 1), Point(1, 8, 1)], [0, 0, 1], "Top Middle Rib Top Face"),

        # Top Rib
        Face([Point(1, 9, 0), Point(1, 9, 1), Point(9, 9, 1), Point(9, 9, 0)], [0, -1, 0], "Top Rib Front Face"),
        Face([Point(1, 9, 0), Point(1, 9, 1), Point(1, 10, 1), Point(1, 10, 0)], [-1, 0, 0], "Top Rib Left Face"),
        Face([Point(1, 10, 0), Point(1, 10, 1), Point(9, 10, 1), Point(9, 10, 0)], [0, 1, 0], "Top Rib Back Face"),
        Face([Point(9, 9, 0), Point(9, 9, 1), Point(9, 10, 1), Point(9, 10, 0)], [1, 0, 0], "Top Rib Right Face"),
        Face([Point(1, 9, 1), Point(9, 9, 1), Point(9, 10, 1), Point(1, 10, 1)], [0, 0, 1], "Top Rib Top Face"),
        
        # Left Partition
        Face([Point(0, 5, 0), Point(4, 5, 0), Point(4, 5, 4), Point(0, 5, 4)], [0, -1, 0], "Left Partition Front Face"),
        Face([Point(4, 5, 0), Point(4, 5, 4), Point(4, 6, 4), Point(4, 6, 0)], [1, 0, 0], "Left Partition Right Face"),
        Face([Point(0, 6, 0), Point(0, 6, 4), Point(4, 6, 4), Point(4, 6, 0)], [0, 1, 0], "Left Partition Back Face"),
        Face([Point(0, 5, 4), Point(4, 5, 4), Point(4, 6, 4), Point(0, 6, 4)], [0, 0, -1], "Left Partition Top Face"),

        
        # Right Partition
        Face([Point(6, 5, 0), Point(10, 5, 0), Point(10, 5, 4), Point(6, 5, 4)], [0, -1, 0], "Right Partition Front Face"),
        Face([Point(6, 5, 0), Point(6, 5, 4), Point(6, 6, 4), Point(6, 6, 0)], [-1, 0, 0], "Right Partition Left Face"),
        Face([Point(6, 6, 0), Point(6, 6, 4), Point(10, 6, 4), Point(10, 6, 0)], [0, 1, 0], "Right Partition Back Face"),
        Face([Point(10, 5, 4), Point(10, 6, 4), Point(6, 6, 4), Point(6, 5, 4)], [0, 0, -1], "Right Partition Top Face"),
        
        # Left Boundary Face
        Face([Point(0, 0, 0), Point(0, 0, 5), Point(0, 11, 5), Point(0, 11, 0), Point(0, 6, 0), Point(0, 6, 4), Point(0, 5, 4), Point(0, 5, 0)], [1, 0, 0], "Left Boundary Face"),
        
        # Right Boundary Face
        Face([Point(10, 0, 0), Point(10, 0, 5), Point(10, 11, 5), Point(10, 11, 0), Point(10, 6, 0), Point(10, 6, 4), Point(10, 5, 4), Point(10, 5, 0)], [-1, 0, 0], "Right Boundary Face"),
        
        # Front Boundary Face
        Face([Point(0, 0, 0), Point(10, 0, 0), Point(10, 0, 5), Point(0, 0, 5)], [0, 1, 0], "Front Boundary Face"),
        
        # Back Boundary Face
        Face([Point(0, 11, 0), Point(10, 11, 0), Point(10, 11, 5), Point(0, 11, 5)], [0, -1, 0], "Back Boundary Face"),
        
        # Top Face
        Face([Point(0, 0, 5), Point(10, 0, 5), Point(10, 11, 5), Point(0, 11, 5)], [0, 0, -1], "Top Face", 
             holes=[
                [Point(1.5, 1, 5), Point(2.5, 1, 5), Point(2.5, 10, 5), Point(1.5, 10, 5)],  # Top Vertical Rib 1 Hole
                [Point(3.5, 1, 5), Point(4.5, 1, 5), Point(4.5, 10, 5), Point(3.5, 10, 5)],  # Top Vertical Rib 2 Hole
                [Point(5.5, 1, 5), Point(6.5, 1, 5), Point(6.5, 10, 5), Point(5.5, 10, 5)],  # Top Vertical Rib 3 Hole
                [Point(7.5, 1, 5), Point(8.5, 1, 5), Point(8.5, 10, 5), Point(7.5, 10, 5)]   # Top Vertical Rib 4 Hole
            ]
        ),

        # ============================
        # Top Vertical Ribs
        # ============================

        # Top Vertical Rib 1 (x = 1.5–2.5)
        Face([Point(1.5, 1, 4), Point(2.5, 1, 4), Point(2.5, 10, 4), Point(1.5, 10, 4)], [0, 0, -1], "Top Vertical Rib 1 Bottom Face"),
        Face([Point(1.5, 1, 4), Point(1.5, 1, 5), Point(2.5, 1, 5), Point(2.5, 1, 4)], [0, -1, 0], "Top Vertical Rib 1 Front Face"),
        Face([Point(1.5, 10, 4), Point(2.5, 10, 4), Point(2.5, 10, 5), Point(1.5, 10, 5)], [0, 1, 0], "Top Vertical Rib 1 Back Face"),
        Face([Point(1.5, 1, 4), Point(1.5, 10, 4), Point(1.5, 10, 5), Point(1.5, 1, 5)], [-1, 0, 0], "Top Vertical Rib 1 Left Face"),
        Face([Point(2.5, 1, 4), Point(2.5, 1, 5), Point(2.5, 10, 5), Point(2.5, 10, 4)], [1, 0, 0], "Top Vertical Rib 1 Right Face"),

        # Top Vertical Rib 2 (x = 3.5–4.5)
        Face([Point(3.5, 1, 4), Point(4.5, 1, 4), Point(4.5, 10, 4), Point(3.5, 10, 4)], [0, 0, -1], "Top Vertical Rib 2 Bottom Face"),
        Face([Point(3.5, 1, 4), Point(3.5, 1, 5), Point(4.5, 1, 5), Point(4.5, 1, 4)], [0, -1, 0], "Top Vertical Rib 2 Front Face"),
        Face([Point(3.5, 10, 4), Point(4.5, 10, 4), Point(4.5, 10, 5), Point(3.5, 10, 5)], [0, 1, 0], "Top Vertical Rib 2 Back Face"),
        Face([Point(3.5, 1, 4), Point(3.5, 10, 4), Point(3.5, 10, 5), Point(3.5, 1, 5)], [-1, 0, 0], "Top Vertical Rib 2 Left Face"),
        Face([Point(4.5, 1, 4), Point(4.5, 1, 5), Point(4.5, 10, 5), Point(4.5, 10, 4)], [1, 0, 0], "Top Vertical Rib 2 Right Face"),

        # Top Vertical Rib 3 (x = 5.5–6.5)
        Face([Point(5.5, 1, 4), Point(6.5, 1, 4), Point(6.5, 10, 4), Point(5.5, 10, 4)], [0, 0, -1], "Top Vertical Rib 3 Bottom Face"),
        Face([Point(5.5, 1, 4), Point(5.5, 1, 5), Point(6.5, 1, 5), Point(6.5, 1, 4)], [0, -1, 0], "Top Vertical Rib 3 Front Face"),
        Face([Point(5.5, 10, 4), Point(6.5, 10, 4), Point(6.5, 10, 5), Point(5.5, 10, 5)], [0, 1, 0], "Top Vertical Rib 3 Back Face"),
        Face([Point(5.5, 1, 4), Point(5.5, 10, 4), Point(5.5, 10, 5), Point(5.5, 1, 5)], [-1, 0, 0], "Top Vertical Rib 3 Left Face"),
        Face([Point(6.5, 1, 4), Point(6.5, 1, 5), Point(6.5, 10, 5), Point(6.5, 10, 4)], [1, 0, 0], "Top Vertical Rib 3 Right Face"),

        # Top Vertical Rib 4 (x = 7.5–8.5)
        Face([Point(7.5, 1, 4), Point(8.5, 1, 4), Point(8.5, 10, 4), Point(7.5, 10, 4)], [0, 0, -1], "Top Vertical Rib 4 Bottom Face"),
        Face([Point(7.5, 1, 4), Point(7.5, 1, 5), Point(8.5, 1, 5), Point(8.5, 1, 4)], [0, -1, 0], "Top Vertical Rib 4 Front Face"),
        Face([Point(7.5, 10, 4), Point(8.5, 10, 4), Point(8.5, 10, 5), Point(7.5, 10, 5)], [0, 1, 0], "Top Vertical Rib 4 Back Face"),
        Face([Point(7.5, 1, 4), Point(7.5, 10, 4), Point(7.5, 10, 5), Point(7.5, 1, 5)], [-1, 0, 0], "Top Vertical Rib 4 Left Face"),
        Face([Point(8.5, 1, 4), Point(8.5, 1, 5), Point(8.5, 10, 5), Point(8.5, 10, 4)], [1, 0, 0], "Top Vertical Rib 4 Right Face"),
    
    ], description="ribbed-cage")