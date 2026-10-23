import pytest

import tsgo.algorithm.energy_functions as ef


@pytest.fixture(autouse=True)
def _fresh_geometry_caches():
    """Tests build containers of different geometry (a radius-1 cube beside
    the unit one), and the energy caches are keyed by id(container). CPython
    reuses ids, so without this a test can score its container with another
    test's cached geometry -- which failed test_energy_and_gradient_paths_agree
    about one run in six."""
    ef.clear_caches()
    yield
