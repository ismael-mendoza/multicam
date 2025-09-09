"""Our own implementation of quantile transformers."""

import numpy as np
from numpy import ndarray
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


# TODO: account for edges?
# TODO: consider pre-sorted input.
def qt_gauss_base(x: ndarray, x_base: ndarray):
    """Gaussinize input based on another dataset."""
    # always assume second dimension is n_features.
    assert x.ndim == 2 and x_base.ndim == 2
    assert x.shape[1] == x_base.shape[1]
    n_features = x.shape[1]
    x_gauss = np.zeros_like(x) * np.nan
    for jj in range(n_features):
        x_jj = x[:, jj]
        xb_jj = np.sort(x_base[:, jj])  # required for np.interp
        xb_gauss_jj = qt_gauss(xb_jj, method="ordinal")
        x_gauss[:, jj] = np.interp(x_jj, xb_jj, xb_gauss_jj)
    return x_gauss


def qt_inverse_gauss_base(x_gauss: ndarray, x_base: ndarray):
    assert x_gauss.ndim == 2 and x_base.ndim == 2
    assert x_gauss.shape[1] == x_base.shape[1]
    n_features = x_gauss.shape[1]
    x_out = np.zeros_like(x_gauss) * np.nan
    for jj in range(n_features):
        xg_jj = x_gauss[:, jj]
        xb_jj = np.sort(x_base[:, jj])  # required for np.interp
        xb_gauss_jj = qt_gauss(xb_jj, method="ordinal")
        x_out[:, jj] = np.interp(xg_jj, xb_gauss_jj, xb_jj)
    return x_out
