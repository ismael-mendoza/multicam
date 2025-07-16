"""Our own implementation of quantile transformers."""

import numpy as np
from scipy.stats import norm, rankdata


def qt(x, y):
    """Performs a quantile transformation from x -> y marginally over first axis.

    Returns what values would x have if they had the same ranks within the y
    distribution.

    CREDIT: Phil Mansfield
    """
    assert x.ndim == 1 and y.ndim == 1
    ranks = rankdata(x, method="ordinal") - 1  # want 0 indexed ranks
    ranks = ranks.astype(float)

    # rescale to size of y and convert to indices/remainders
    ranks *= (len(y) - 1) / (len(x) - 1)
    idx = np.array(ranks, dtype=int)
    remainder = ranks - idx

    # reduce the indices that are too large and make up for it with the remainder
    too_big = idx == len(y) - 1
    idx[too_big], remainder[too_big] = len(y) - 2, 1.0

    # linearly interpolate
    _y = np.sort(y)
    dy = _y[1:] - _y[:-1]
    return _y[idx] + dy[idx] * remainder


def qt_uniform(x, axis: int = 0, method="ordinal"):  # default in rankdata = 'average'
    """Transform array to a uniform distribution based on ranks."""
    ranks = rankdata(x, axis=axis, method=method)
    return ranks / (ranks.shape[axis] + 1)  # excludes 0 and 1


def qt_gauss(x, axis: int = 0, method="ordinal"):
    """Transform array to a Gaussian distribution based on ranks."""
    u = qt_uniform(x, axis=axis, method=method)
    return norm.ppf(u)
