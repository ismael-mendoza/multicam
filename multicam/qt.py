"""Our own implementation of quantile transformers."""

import numpy as np
from scipy.stats import norm


def _get_ranks(x):
    assert x.ndim == 1
    # get ranks for each element of x
    n = len(x)
    ranks = np.zeros(n)
    ranks[np.argsort(x)] = np.arange(n)
    return ranks


def qt(x, y):
    """qt performs a quantile transformation from x -> y. (i.e. what values
    would the values in x have if they had the same ranks within the y
    distribution).

    CREDIT: Phil Mansfield
    """
    assert x.ndim == 1 and y.ndim == 1
    ranks = _get_ranks(x)

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
    ranks = _get_ranks(x)
    return ranks / (len(ranks) - 1)  # includes 0 and 1


def qt_gauss(x):
    """Transform array to a Gaussian distribution based on ranks."""
    eps = 1 / len(x) / 2
    u = qt_uniform(x)
    u_clipped = np.clip(u, eps, 1 - eps)
    return norm.ppf(u_clipped)
