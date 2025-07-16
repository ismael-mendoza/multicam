import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import rankdata


def _phil_rank(x):
    n = len(x)
    rank = np.zeros(n)
    rank[np.argsort(x)] = np.arange(n)
    return rank


def test_ranking_function():
    x = np.random.randn(1000)
    assert_allclose(rankdata(x, method="ordinal") - 1, _phil_rank(x))

    y = np.array([0, 1, 1, 2])
    assert_allclose(rankdata(y, method="ordinal") - 1, _phil_rank(y))

    # default handles repetiations differently
    assert not np.allclose(rankdata(y, method="average") - 1, _phil_rank(y))
