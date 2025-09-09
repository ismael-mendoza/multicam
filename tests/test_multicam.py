import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import ks_2samp, spearmanr

from multicam.multicam import MultiCAM, multicam_prediction
from multicam.train import get_tt_indices


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
    rng = np.random.default_rng(42)
    x = rng.standard_normal(10_000) * 5 + 4
    # small noise compared to x scatter
    y = rng.standard_normal(10_000) * 0.5 + 2 * x + 10
    train_idx, test_idx = get_tt_indices(10_000, rng, 0.25)

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
    rng = np.random.default_rng(42)
    x = rng.standard_normal(size=(10_000, 2)) * 2 + 1
    y = rng.standard_normal(10_000) * 0.5 + 2 * x[:, 0] + 3 * x[:, 1] + 10
    y = y.reshape(-1, 1)

    train_idx, test_idx = get_tt_indices(10_000, rng, 0.25)

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = MultiCAM(2, 1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    assert spearmanr(y_pred, y_test).statistic > 0.9

    # distribution is the same (KS 2 sample test)
    result = ks_2samp(y_pred, y_test)
    pval = result.pvalue
    ks = result.statistic
    assert pval > 0.99
    assert ks < 0.02


def test_2d_repeats():
    n_points = 10_000
    rng = np.random.default_rng(42)
    random_indices = rng.choice(np.arange(n_points), replace=False, size=(1000,))

    x = rng.standard_normal(size=(n_points, 2)) * 2 + 1
    x[random_indices, 1] = 1.0
    y = rng.standard_normal(n_points) * 0.5 + 2 * x[:, 0] + 3 * x[:, 1] + 10
    y = y.reshape(-1, 1)

    train_idx, test_idx = get_tt_indices(n_points, rng, 0.25)

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    model = MultiCAM(2, 1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    assert spearmanr(y_pred, y_test).statistic > 0.9

    result = ks_2samp(y_pred, y)
    pval = result.pvalue
    ks = result.statistic

    # degraded
    assert ks < 0.02
    assert pval > 0.5  # large p-value => no evidence for alternative (different dists)


def test_ks_numerical_precision():
    rng = np.random.default_rng(42)

    # ks test should be perfect to numerical accuracy if same train and test sets are the same
    n_points = 10_000
    x = rng.standard_normal(size=(n_points, 1)) * 5 + 4
    y = rng.standard_normal(size=(n_points, 1)) + 2 * x + 10

    model = MultiCAM(1, 1)
    model.fit(x, y)
    y_pred = model.predict(x)

    assert spearmanr(y_pred, y).statistic > 0.9
    result = ks_2samp(y_pred, y)
    pval = result.pvalue
    ks = result.statistic
    assert pval > 1 - 1e-5
    assert ks < 1e-5


def test_class_and_functional_multicam():
    rng = np.random.default_rng(42)
    n_points = 10_000
    train_idx, test_idx = get_tt_indices(n_points, rng, 0.25)

    x = rng.standard_normal(size=(n_points, 1)) * 5 + 4
    y = rng.standard_normal(size=(n_points, 1)) + 2 * x + 10

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
