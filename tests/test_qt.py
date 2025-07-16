import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

from multicam.qt import qt, qt_gauss, qt_uniform


def test_simple():
    x = np.array([1, 3, 2, 5, 3, 3, 3, 4, 9, 11, 14, 17])  # works with repeated values
    z = qt_gauss(x)
    u = qt_uniform(x)

    assert_allclose(x, qt(z, x))
    assert_allclose(x, qt(u, x))


def test_exponential():
    x = stats.expon.rvs(size=(100_000))
    z = qt_gauss(x)
    u = qt_uniform(x)

    assert_allclose(x, qt(z, x))
    assert_allclose(x, qt(u, x))
