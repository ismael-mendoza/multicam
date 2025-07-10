import numpy as np
from numpy.testing import assert_allclose

from multicam.qt import qt, qt_gauss, qt_uniform


def test_qts():
    x = np.array([1, 3, 2, 5, 4, 9, 11, 14, 17])
    z = qt_gauss(x)
    u = qt_uniform(x)

    assert_allclose(x, qt(z, x))
    assert_allclose(x, qt(u, x))
