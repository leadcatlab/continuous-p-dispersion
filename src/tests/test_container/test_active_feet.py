"""Regressions for the active-face rule and the metric-nearest footpoint.

Three defects are pinned here.

A5  A point a few 1e-9 *outside* a wall is classified inside the container by
    contains(), which admits anything within EPS_PLANE of a face. The active
    test then demanded h >= 0 strictly, so that point's own wall dropped out of
    its active set: clearance fell back to a distant edge and a ball resting on
    the floor scored E ~ 1e-16, i.e. perfectly feasible.

tie The nearest point of a plane is not unique under a polyhedral norm. Picking
    one minimiser by argmax made face membership -- hence the energy -- depend
    on the order the unit ball's vertices are listed in.

E/G  The scalar energy and the analytic-gradient path implemented different
    versions of that rule, so the optimiser descended a function other than the
    one reported.
"""

import numpy as np
import pytest

import tsgo.algorithm.energy_functions as ef
from tsgo.container.face import EPS_PLANE
from tsgo.container.face_batch import active_feet, EPS_PLANE as BATCH_EPS
from tsgo.container.point import Point
from tsgo.container.sample_polyhedron import CONTAINERS

CONTAINER_NAMES = ["cube", "h-box", "tetrahedron", "pointy-star"]
METRICS = ["EUCLIDEAN", "TAXICAB", "SUP_NORM"]


def test_plane_tolerance_is_shared():
    """The two rules that disagreed must read one constant."""
    assert BATCH_EPS is EPS_PLANE


@pytest.mark.parametrize("metric", METRICS)
def test_point_just_outside_a_wall_keeps_that_wall_active(metric):
    """A5: inside-by-tolerance must imply the wall is still active."""
    C = CONTAINERS["cube"]()
    # 6.6e-9 below the floor z = -0.5, the offset actually observed in
    # output/cube/p3/sup_norm/seed27.npz.
    x = np.array([0.0, 0.0, -0.5 - 6.6e-9])
    pt = Point(*map(float, x))
    assert C.contains(pt), "the tolerance should still call this inside"

    feet = C.get_active_footpoints(pt, metric)
    assert feet, "the point's own wall must stay in its active set"
    assert min(abs(f.z + 0.5) for f in feet) < 1e-8, "the floor must be among them"

    D = 0.25
    assert ef.E(x[None, :], D, C, metric) > 0.9 * D * D, \
        "a ball sitting on the floor must not score as feasible"


def test_stored_false_feasible_run_is_now_caught():
    """The exact configuration that reported E ~ 1e-16 while resting on a wall."""
    C = CONTAINERS["cube"]()
    X = np.array([[-0.25, 0.22583308, 0.22424319],
                  [-0.22899049, 0.22246941, -0.50000001],
                  [0.25000001, 0.22234215, 0.17148304]])
    D = 0.2500000035933015
    assert ef.E(X, D, C, "SUP_NORM") == pytest.approx(D * D, rel=1e-6)


def test_tied_minimisers_prefer_a_foot_on_the_face():
    """Four ball vertices tie on an axis-aligned wall under the sup norm.

    The centroid of the tied set is the orthogonal direction, so the foot must
    come back orthogonal rather than dragged diagonally off the face.
    """
    C = CONTAINERS["cube"]()
    x = np.array([-0.48347236, 0.31327024, 0.41275558])
    feet, keep, hs = active_feet(C, x, np.ones(len(C.faces), dtype=bool), "SUP_NORM")
    wall = int(np.argmin(np.abs(hs - 0.01652764)))
    assert keep[wall], "the wall nearest the point must contribute a footpoint"
    assert feet[wall] == pytest.approx([-0.5, x[1], x[2]], abs=1e-12)


@pytest.mark.parametrize("name", CONTAINER_NAMES)
@pytest.mark.parametrize("metric", METRICS)
def test_energy_and_gradient_paths_agree(name, metric):
    """E and E_and_grad must be the same function, not two readings of one."""
    C = CONTAINERS[name]()
    rng = np.random.default_rng(0)
    for _ in range(8):
        X = C.generate_initial_solution(4, rng)
        for D in (0.05, 0.15):
            assert ef.E(X, D, C, metric) == pytest.approx(
                ef.E_and_grad(X, D, C, metric)[0], abs=1e-9)


@pytest.mark.parametrize("name", CONTAINER_NAMES)
@pytest.mark.parametrize("metric", METRICS)
def test_clearance_never_over_estimates_the_boundary_distance(name, metric):
    """The model's min over V u E u H^i must not exceed the true clearance.

    Over-estimating is the unsafe direction: it lets a ball cross a wall while
    the energy still reads zero. Ground truth is a coarse boundary sample, which
    is itself an upper bound on the true distance, so the test only asserts the
    safe side.
    """
    from tsgo.container.geometry import distance
    from tsgo.container.norms import segment_distance

    C = CONTAINERS[name]()
    S = np.vstack([np.array([v.as_array() for v in f.vertices], dtype=float)
                   for f in C.faces])
    # sample each edge as well, so the reference is not vertices alone
    for a, b in C.edges:
        a3, b3 = a.as_array().astype(float), b.as_array().astype(float)
        t = np.linspace(0, 1, 40)[:, None]
        S = np.vstack([S, a3[None, :] + t * (b3 - a3)[None, :]])

    from tsgo.container.norms import get_norm
    norm = get_norm(metric)
    rng = np.random.default_rng(3)
    Sedge = np.array([e[0].as_array() for e in C.edges], dtype=float)
    Eedge = np.array([e[1].as_array() for e in C.edges], dtype=float)

    for _ in range(15):
        x = C._sample_random_point(rng)
        pt = Point(*map(float, x))
        refs = [v.as_array() for v in C.vertices]
        refs += [f.as_array() for f in C.get_active_footpoints(pt, metric)]
        model = min(distance(x, r, metric) for r in refs)
        d, _ = segment_distance(np.broadcast_to(x, Sedge.shape), Sedge, Eedge, metric)
        model = min(model, float(d.min()))

        upper = float(norm.value(S - x[None, :]).min())
        assert model <= upper * (1 + 1e-9), \
            f"clearance {model} over-estimates the boundary distance {upper}"
