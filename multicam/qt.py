"""Our own implementation of quantile transformers."""

import numpy as np
from scipy.stats import norm, rankdata


def qt(x, y):
    """qt performs a quantile transformation from x -> y. (i.e. what values
    would the values in x have if they had the same ranks within the y
    distribution).

    CREDIT: Phil Mansfield
    """
    assert x.ndim == 1 and y.ndim == 1
    ranks = rankdata(x, axis=0) - 1  # want 0 indexed ranks

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


def qt_uniform(x):
    """Transform array to a uniform distribution based on ranks."""
    assert x.ndim == 1
    ranks = rankdata(x, axis=0) - 1
    return ranks / (len(ranks) - 1)  # includes 0 and 1


def qt_gauss(x):
    """Transform array to a Gaussian distribution based on ranks."""
    assert x.ndim == 1
    eps = 1 / len(x) / 2  # ppf at 0 and 1 returns infinity, does not change rank
    u = qt_uniform(x)
    u_clipped = np.clip(u, eps, 1 - eps)
    return norm.ppf(u_clipped)
