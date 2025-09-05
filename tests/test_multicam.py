# unit tests: check it works for a very large number of data points, small number of datasets (2, 3) size of datasets. large training, small test, ...
# ks test statistic for same distribution
# discrete, repeated values

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import ks_2samp, spearmanr

from multicam.multicam import MultiCAM, multicam_prediction
from multicam.train import get_tt_indices


def test_class_and_functional_multicam():
    x = np.random.randn(10_000) * 5 + 4
    y = np.random.randn(10_000) + 2 * x + 10

    # train and test split
    train_idx, test_idx = get_tt_indices(10_000, np.random.default_rng(42), 0.2)

    x_train = x[train_idx].reshape(-1, 1)
    x_test = x[test_idx].reshape(-1, 1)
    y_train = y[train_idx].reshape(-1, 1)

    # train
    model = MultiCAM(1, 1)
    model.fit(x_train, y_train)
    y_pred1 = model.predict(x_train)
    y_pred2 = multicam_prediction(x_train, x_train, y_train)
    assert_allclose(y_pred1, y_pred2)

    # now test
    y_pred1 = model.predict(x_test)
    y_pred2 = multicam_prediction(x_test, x_train, y_train)
    assert_allclose(y_pred1, y_pred2)


def test_1d_simple():
    model = MultiCAM(1, 1)

    # perfectly monotonic 1D
    x = np.array([0.1, 0.2, 0.3]).reshape(-1, 1)
    y = np.array([1, 2, 3]).reshape(-1, 1)
    model.fit(x, y)
    y_pred = model.predict(x)
    assert_allclose(y_pred, y)


def test_1d_interpol():
    # does not interpolate correctly yet
    model = MultiCAM(1, 1)

    # perfectly monotonic 1D
    x = np.array([0.1, 0.2, 0.3]).reshape(-1, 1)
    y = np.array([1, 2, 3]).reshape(-1, 1)
    model.fit(x, y)

    # check interpol
    x_test = np.array([0.11]).reshape(-1, 1)
    y_pred = model.predict(x_test)
    assert_allclose(y_pred.item(), 2.0)


def test_1d_complex():
    x = np.random.randn(10_000) * 5 + 4
    y = np.random.randn(10_000) * 0.5 + 2 * x + 10  # small noise compared to x scatter
    train_idx, test_idx = get_tt_indices(10_000, np.random.default_rng(42), 0.25)

    x_train = x[train_idx].reshape(-1, 1)
    x_test = x[test_idx].reshape(-1, 1)
    y_train = y[train_idx].reshape(-1, 1)
    y_test = y[test_idx].reshape(-1, 1)

    model = MultiCAM(1, 1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    assert spearmanr(y_pred, y_test).statistic > 0.8

    # distribution is the same (KS 2 sample test)
    result = ks_2samp(y_pred, y_test)
    pval = result.pvalue
    ks = result.statistic
    assert pval > 0.99
    assert ks < 0.02


def test_2d():
    x = np.random.randn(10_000, 2) * 2 + 1
    y = np.random.randn(10_000) * 0.5 + 2 * x[:, 0] + 3 * x[:, 1] + 10  # small noise
    y = y.reshape(-1, 1)

    train_idx, test_idx = get_tt_indices(10_000, np.random.default_rng(42), 0.25)

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = MultiCAM(2, 1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    assert spearmanr(y_pred, y_test).statistic > 0.8

    # distribution is the same (KS 2 sample test)
    result = ks_2samp(y_pred, y_test)
    pval = result.pvalue
    ks = result.statistic
    assert pval > 0.99
    assert ks < 0.02


def test_1d_repeats():
    pass


def test_2d_repeats():
    pass
