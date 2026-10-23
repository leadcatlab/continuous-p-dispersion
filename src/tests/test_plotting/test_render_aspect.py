"""strip_chrome must not change the 3-D box aspect.

Axes3D defaults to a 4:4:3 box, which compresses z by 25% and renders spheres
as visibly oblate ellipsoids. strip_chrome ends with set_box_aspect(..., zoom=)
to trim padding, and passing None there silently restored that default -- so
every clean paper figure came out vertically squashed. It is only visible by
eye, hence this test.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tsgo.experiments.render import strip_chrome


def _axes3d(box_aspect):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect(box_aspect)
    return fig, ax


@pytest.mark.parametrize("aspect", [(1, 1, 1), (1.1, 1.1, 1.1), (2, 1, 1)])
def test_strip_chrome_preserves_box_aspect(aspect):
    fig, ax = _axes3d(aspect)
    before = np.asarray(ax.get_box_aspect(), dtype=float)
    strip_chrome(ax)
    after = np.asarray(ax.get_box_aspect(), dtype=float)
    plt.close(fig)
    # zoom may rescale uniformly; the ratios between axes must not change
    assert np.allclose(after / after[0], before / before[0], atol=1e-9), (
        f"strip_chrome changed the box aspect ratios: {before} -> {after}")


def test_strip_chrome_keeps_equal_axes_equal():
    """The case that matters: an equal box must stay equal, or spheres skew."""
    fig, ax = _axes3d((1, 1, 1))
    strip_chrome(ax)
    after = np.asarray(ax.get_box_aspect(), dtype=float)
    plt.close(fig)
    assert np.allclose(after, after[0], atol=1e-9), (
        f"equal box aspect became unequal: {after}")
