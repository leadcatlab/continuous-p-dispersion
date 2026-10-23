import copy
import numpy as np
import pytest

from tsgo.container.container import Container
from tsgo.container.face import Face
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import cube as cube_sample, h_box as h_box_sample, ribbed_cage as cage_sample

cube = cube_sample(radius=1)
h_box = h_box_sample()
cage = cage_sample()


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
        assert cube._is_active_plane(face, point, True) == True, f"Face {face.description} should be an active plane for point {point}."
        
    point = Point(0, 0, -10)
    face = cube.faces[1]
    assert cube._is_active_plane(face, point, False) == False, f"Face {face.description} should not be an active plane for point {point}."

def test_active_plane_outside():
    point = Point(10, 10, 10)
    for face in cube.faces:
        is_active = cube._is_active_plane(face, point, False)
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
    
def test_contains_ribbed_cage():
    assert cage.contains(Point(0, 0, 0)) == True, "Point (0, 0, 0) is on the boundary of the ribbed cage."
    
    # Bottom rib
    assert cage.contains(Point(1.5, 1.5, 0.1)) == False, "Point (1.5, 1.5, 0.1) should be outside the ribbed cage."
    assert cage.contains(Point(1.5, 1.5, -0.1)) == False, "Point (1.5, 1.5, -0.1) should be outside the ribbed cage."
    assert cage.contains(Point(1.5, 1.5, 0)) == False, "Point (1.5, 1.5, 0) should be outside the ribbed cage."
    assert cage.contains(Point(1.5, 1.5, 0.5)) == False, "Point (1.5, 1.5, 0.5) should be outside the ribbed cage."
    assert cage.contains(Point(1.5, 1.5, 0.75)) == False, "Point (1.5, 1.5, 0.75) should be outside the ribbed cage."
    
    # Second rib
    assert cage.contains(Point(2.5, 3.5, 0.5)) == False, "Point (2.5, 3.5, 0.5) should be outside of the ribbed cage."
    assert cage.contains(Point(2.5, 3.5, 1)) == True, "Point (2.5, 3.5, 1) should be on the boundary of the ribbed cage."
    
    # Left partiton
    assert cage.contains(Point(0, 5.5, 1)) == False, "Point (0, 5.5, 1) should be outside of the ribbed cage."
    assert cage.contains(Point(0, 5.5, 0)) == False, "Point (0, 5.5, 0) should be outside of the ribbed cage."
    assert cage.contains(Point(0, 5.5, 3)) == False, "Point (0, 5.5, 3) should be outside of the ribbed cage."
    assert cage.contains(Point(4, 5.5, 0)) == True, "Point (4, 5.5, 0) should be on the boundary of the ribbed cage."
    
    # Right partition
    assert cage.contains(Point(9.5, 5.5, 1)) == False, "Point (4, 5.5, 1) should be outside of the ribbed cage."
    assert cage.contains(Point(9.5, 5.5, 0)) == False, "Point (4, 5.5, 0) should be outside of the ribbed cage."
    assert cage.contains(Point(9.3, 5.5, 3)) == False, "Point (4, 5.5, 3) should be outside of the ribbed cage."
    assert cage.contains(Point(6, 5.5, 0)) == True, "Point (6, 5.5, 0) should be on the boundary of the ribbed cage."
    
    # Top Ribs
    assert cage.contains(Point(2, 5, 5)) == False, "Point (2, 5, 5) should be outside of the ribbed cage."
    assert cage.contains(Point(2, 5, 4.5)) == False, "Point (2, 5, 4.5) should be outside of the ribbed cage."
    assert cage.contains(Point(4, 5, 5)) == False, "Point (4, 5, 5) should be outside of the ribbed cage."
    assert cage.contains(Point(6, 5, 4.5)) == False, "Point (6, 5, 4.5) should be outside of the ribbed cage."
    assert cage.contains(Point(8, 5, 4.37)) == False, "Point (8, 5, 4.37) should be outside of the ribbed cage."
    assert cage.contains(Point(8, 5, 4)) == True, "Point (8, 5, 4) should be on the boundary of the ribbed cage."
    
def test_contains_ribbed_cage_outside_points():
    point = Point.from_array([-0.08396896, 3.23984771, -1.15235849])
    assert cage.contains(point) == False, "Point (-0.08396896,3.23984771,-1.15235849) should be outside the ribbed cage."
    

    

def test_contains_does_not_share_a_verdict_across_a_wall():
    # Regression: the cache key rounded to 6 decimals, so a point 4e-7 outside
    # the top face shared a key with one 2e-7 inside it and inherited "inside".
    # The optimiser found that pocket; 201 of 250 infeasible sweep runs ended
    # with one point parked in it.
    from tsgo.container.sample_polyhedron import CONTAINERS
    c = CONTAINERS["cube"]()
    assert c.contains(np.array([-0.0476, -0.0269, 0.4999998]))
    assert not c.contains(np.array([-0.0476, -0.0269, 0.5000004]))


def test_energy_does_not_reuse_a_penalty_across_a_wall():
    # Same pocket, seen through the energy: after a neighbour just inside the
    # wall has been scored, a point just outside it must still be penalised,
    # and score the same as it does on a cold cache.
    import tsgo.algorithm.energy_functions as ef
    from tsgo.container.sample_polyhedron import CONTAINERS
    D = 0.1
    inside = np.array([[-0.0476, -0.0269, 0.4999998]])
    outside = np.array([[-0.0476, -0.0269, 0.5000004]])
    ef.clear_caches()
    cold = ef.E(outside, D, CONTAINERS["cube"](), metric="TAXICAB")
    c = CONTAINERS["cube"]()
    ef.E(inside, D, c, metric="TAXICAB")
    warm = ef.E(outside, D, c, metric="TAXICAB")
    assert cold > 1e-3
    assert warm == cold


def _oriented_rings(name):
    """Every ring of a container file, oriented counter-clockwise about the
    outward normal its face's inward-normal hint implies (holes clockwise)."""
    import json, os
    from tsgo.container import sample_polyhedron as sp

    def vec_area(r):
        return 0.5 * np.sum(np.cross(r, np.roll(r, -1, axis=0)), axis=0)

    spec = json.load(open(os.path.join(sp.DATA_DIR, name)))
    rings = []
    for i, f in enumerate(spec["faces"]):
        out = -np.asarray(f["normal"], dtype=float)
        R = np.asarray(f["vertices"], dtype=float)
        rings.append((i, R if vec_area(R) @ out > 0 else R[::-1]))
        for h in f.get("holes", []):
            H = np.asarray(h, dtype=float)
            rings.append((i, H if vec_area(H) @ out < 0 else H[::-1]))
    return rings, vec_area


@pytest.mark.parametrize("name", ["cube.json", "h_box.json", "tetrahedron.json",
                                  "pointy_star.json", "ribbed_cage.json"])
def test_normal_hints_orient_a_closed_surface_consistently(name):
    # Regression: two faces of the cage carried outward "inward" normals. The
    # active-face rule then hid those walls from points inside, and clearance
    # to them read as larger than it was -- three stored cage solutions put a
    # ball through a partition top. With consistent hints every edge is
    # walked once in each direction by the two faces that share it.
    from collections import Counter
    rings, _ = _oriented_rings(name)
    walked = Counter()
    for _, R in rings:
        for a, b in zip(R, np.roll(R, -1, axis=0)):
            walked[(tuple(np.round(a, 9)), tuple(np.round(b, 9)))] += 1
    for (a, b), n in walked.items():
        assert n == 1 and walked[(b, a)] == 1, f"edge {a}->{b} is not shared by oppositely walked faces"


@pytest.mark.parametrize("key,name", [("cube", "cube.json"), ("h-box", "h_box.json"),
                                      ("tetrahedron", "tetrahedron.json"),
                                      ("pointy-star", "pointy_star.json"),
                                      ("cage", "ribbed_cage.json")])
def test_volume_matches_the_divergence_theorem_on_the_raw_rings(key, name):
    # Independent of Face's 2-D projection and of its area property, which
    # used to ignore holes.
    from tsgo.container.sample_polyhedron import CONTAINERS
    rings, vec_area = _oriented_rings(name)
    exact = sum(vec_area(R) @ R[0] for _, R in rings) / 3.0
    assert exact > 0
    assert CONTAINERS[key]().volume == pytest.approx(exact, rel=1e-12)
