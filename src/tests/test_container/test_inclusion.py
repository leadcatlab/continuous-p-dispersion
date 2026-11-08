import numpy as np
import pytest

from tsgo.container import inclusion
from tsgo.container.sample_polyhedron import CONTAINERS


def _points(C, rng, n=600, n_wall=150):
    """Random points in the padded bounding box, and points 1e-6 either side
    of random spots on random faces."""
    V = np.asarray([v.as_array() for f in C.faces for v in f.vertices], dtype=float)
    lo, hi = V.min(axis=0), V.max(axis=0)
    pad = 0.05 * (hi - lo)
    X = rng.uniform(lo - pad, hi + pad, (n, 3))
    W = []
    for i in rng.integers(0, len(C.faces), n_wall):
        f = C.faces[i]
        R = np.asarray([v.as_array() for v in f.vertices], dtype=float)
        q = rng.dirichlet(np.ones(len(R))) @ R
        nrm = inclusion._normal(f)
        nrm = nrm / np.linalg.norm(nrm)
        W += [q + 1e-6 * nrm, q - 1e-6 * nrm]
    return X, np.asarray(W)


@pytest.mark.parametrize('name', sorted(CONTAINERS))
def test_tetra_agrees_with_the_ray_casts(name):
    C = CONTAINERS[name]()
    X, W = _points(C, np.random.default_rng(0))
    for P in (X, W):
        assert np.array_equal(C._contains_many_tetra(P), C._contains_many_raycast(P))


@pytest.mark.parametrize('name', ['cube', 'h-box', 'cage'])
def test_a_point_on_a_face_is_inside(name):
    C = CONTAINERS[name]()
    rng = np.random.default_rng(1)
    for f in C.faces[:10]:
        R = np.asarray([v.as_array() for v in f.vertices], dtype=float)
        q = rng.dirichlet(np.ones(len(R))) @ R
        if f.contains(q):
            assert C._contains_many_tetra(q[None, :])[0]


@pytest.mark.parametrize('name', ['cube', 'h-box', 'pointy-star', 'cage'])
def test_a_point_on_a_plane_through_the_origin_is_resolved(name):
    """A point on the plane through the first origin and a triangle edge is
    ambiguous from that origin; the second must decide it, correctly."""
    C = CONTAINERS[name]()
    prep = inclusion.prepare(C)
    O = prep['origins'][0]['O']
    T = inclusion.signed_triangles(C)
    V = T.reshape(-1, 3)
    lo, hi = V.min(axis=0), V.max(axis=0)
    found = 0
    for a, b, _ in T:
        m = 0.5 * (a + b)
        x = O + 0.999 * (m - O)  # on the line from O through m, so on the plane
        if not np.all((x > lo) & (x < hi)):
            continue
        _, amb = inclusion._count(prep['origins'][0], x[None, :])
        if not amb[0]:
            continue
        found += 1
        assert C._contains_many_tetra(x[None, :])[0] == C._contains_many_raycast(x[None, :])[0]
    assert found > 0


def test_the_count_is_deterministic():
    C = CONTAINERS['cage']()
    X, _ = _points(C, np.random.default_rng(2), n=300, n_wall=0)
    assert np.array_equal(C._contains_many_tetra(X), CONTAINERS['cage']()._contains_many_tetra(X))


def test_contains_and_contains_many_agree():
    C = CONTAINERS['h-box']()
    X, W = _points(C, np.random.default_rng(3), n=200, n_wall=50)
    P = np.vstack([X, W])
    assert np.array_equal(C.contains_many(P), np.array([C.contains(p) for p in P]))
