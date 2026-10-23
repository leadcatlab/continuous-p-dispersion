"""Q defaults to max(3, round(ln p))."""
import pytest

from tsgo.algorithm.tsgo_algorithm import default_Q


@pytest.mark.parametrize("p, Q", [(1, 3), (2, 3), (20, 3), (33, 3), (34, 4), (60, 4), (90, 4), (91, 5), (100, 5), (244, 5), (245, 6)])
def test_default_q(p, Q):
    assert default_Q(p) == Q
