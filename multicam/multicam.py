"""Implementation of statistical models."""

import numpy as np
from numpy import linalg, ndarray
from scipy.stats import rankdata
from sklearn import linear_model

from multicam.base import PredictionModel
from multicam.qt import (
    qt_gauss,
    qt_gauss_base,
    qt_inverse_gauss_base,
)


def multicam_prediction(x: ndarray, x_train: ndarray, y_train: ndarray):
    """MultiCAM algorithm without the class scaffolding. Still very fast."""
    assert x.ndim == x_train.ndim == y_train.ndim == 2

    # ranks are unnecessary to start as qt_gauss already uses 'ordinal'
    # the one that we have to be careful with is `qt_gauss_base` below
    xgt = qt_gauss(x_train, axis=0)
    ygt = qt_gauss(y_train, axis=0)

    reg = linear_model.LinearRegression()
    reg.fit(xgt, ygt)

    rank_lookup = _create_rank_lookup(x_train)

    xr = _get_ranks_based(x, x_train, rank_lookup, mode="middle")
    xrt = rankdata(x_train, axis=0, method="ordinal")
    xg = qt_gauss_base(xr, xrt)

    yng = reg.predict(xg)

    yngt = reg.predict(xgt)
    yg = qt_gauss_base(yng, yngt)

    # invert y_gauss to data space based on gaussianized y_train.
    yp = qt_inverse_gauss_base(yg, y_train)
    return yp, yg, yng, xg


class MultiCAM(PredictionModel):
    """MultiCAM model described in our first paper."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__(n_features, n_targets)
        self.x_train = None
        self.y_train = None
        self.y_not_gauss_train = None
        self.rank_lookup = None
        self.reg = linear_model.LinearRegression()

    def _fit(self, x: ndarray, y: ndarray):
        """Fit model using training data"""
        assert not self.trained
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        assert y.shape == (x.shape[0], self.n_targets)

        # ranks need to be based on training set!
        self.x_train = x.copy()
        self.y_train = y.copy()

        # transform variables to be (marginally) gaussian and break ties.
        xg = qt_gauss(x, axis=0)
        yg = qt_gauss(y, axis=0)

        # then fit a linear regression model to the transformed data.
        self.reg.fit(xg, yg)

        # ranks based on training x data for prediction.
        self.rank_lookup = _create_rank_lookup(self.x_train)

        # prediction on training data needed for gaussianization of prediction
        self.y_not_gauss_train = self.reg.predict(xg)

        return xg, yg

    def _predict(self, x):
        # assume continuous data for now
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        # gaussianize x based on x_train
        xr = _get_ranks_based(x, self.x_train, self.rank_lookup, mode="middle")
        xrt = rankdata(self.x_train, axis=0, method="ordinal")
        xg = qt_gauss_base(xr, xrt)

        # predict with linear regression
        yng = self.reg.predict(xg)

        # gaussianize the y_not_gauss using the predictions on train data.
        yg = qt_gauss_base(yng, self.y_not_gauss_train)

        # invert y_gauss to data space based on gaussianized y_train.
        y_pred = qt_inverse_gauss_base(yg, self.y_train)

        return y_pred


class MultiCamSampling(MultiCAM):
    """Multi-Variate Gaussian w/ full covariance matrix (returns conditional mean)."""

    def __init__(self, n_features: int, n_targets: int, rng=None) -> None:
        super().__init__(n_features, n_targets)

        self.rng = np.random.default_rng(42) if rng is None else rng
        self.mu1 = None
        self.mu2 = None
        self.rho = None
        self.sigma_cond = None
        self.Sigma11 = None
        self.Sigma12 = None
        self.Sigma22 = None
        self.sigma_bar = None
        self.Sigma = None
        self.rho = None

    def _fit(self, x, y):
        """
        Fit the Gaussian model.

        We assume a multivariate-gaussian distribution P(X, Y) with conditional distribution
        P(Y | X) = uses the rules here:
        https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
        """
        xg, yg = super()._fit(x, y)  # gaussianized x and y
        fit_params = fit_multi_gauss(xg, yg)

        # update prediction attributes
        self.mu1 = fit_params["mu1"]
        self.mu2 = fit_params["mu2"]
        self.Sigma11 = fit_params["Sigma11"]
        self.Sigma12 = fit_params["Sigma12"]
        self.Sigma22 = fit_params["Sigma22"]
        self.sigma_bar = fit_params["sigma_bar"]
        self.Sigma = fit_params["Sigma"]
        self.rho = fit_params["rho"]

    def sample(self, x):
        """Sample (once) from the conditional distribution P(y | x)"""
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        n_points = x.shape[0]

        # gaussianize input data based on training one
        xr = _get_ranks_based(x, self.x_train, self.rank_lookup, mode="random")
        xg = qt_gauss(xr, axis=0)

        # sample on gaussianized ranks.
        _zero = np.zeros((self.n_targets,))
        mu_cond = get_mu_cond(
            xg,
            mu1=self.mu1,
            mu2=self.mu2,
            Sigma12=self.Sigma12,
            Sigma22=self.Sigma22,
        )
        y_gauss = self.rng.multivariate_normal(
            mean=_zero, cov=self.sigma_bar, size=(n_points,)
        )
        assert y_gauss.shape == (n_points, self.n_targets)
        y_gauss += mu_cond

        # interpolate, by definition y_gauss follows a Gaussian distribution (in each feature)
        # no need to do another transformation
        y_samples = qt_inverse_gauss_base(y_gauss, self.y_train)

        return y_samples


def _get_ranks_based(
    x: ndarray, x_base: ndarray, rank_lookup: dict, mode: str = "middle"
):
    assert mode in {"middle", "random"}
    assert x.ndim == 2
    assert x_base.ndim == 2
    n_features = x.shape[1]

    # get ranks of test data (based on training data)
    xr = np.zeros_like(x) * np.nan
    for jj in range(n_features):
        x_jj = x[:, jj]
        xb_jj = np.sort(x_base[:, jj])
        uniq, lranks, hranks = rank_lookup[jj]
        xr[:, jj] = np.searchsorted(xb_jj, x_jj) + 1  # indices to ranks

        # if value is in training data, get middle or random rank
        in_train = np.isin(x_jj, uniq)
        u_indices = np.searchsorted(uniq, x_jj[in_train])
        lr, hr = lranks[u_indices], hranks[u_indices]  # repeat appropriately
        xr[in_train, jj] = (
            np.random.randint(lr, hr + 1) if mode == "random" else (lr + hr) / 2
        )

    assert np.sum(np.isnan(xr)) == 0

    return xr


def _create_rank_lookup(x):
    assert x.ndim == 2
    n_features = x.shape[1]
    rank_lookup = {}

    # lookup table of ranks
    for jj in range(n_features):
        xjj = np.sort(x[:, jj])
        u, c = np.unique(xjj, return_counts=True)
        lranks = np.cumsum(c) - c + 1
        hranks = np.cumsum(c)
        rank_lookup[jj] = (u, lranks, hranks)

    return rank_lookup


def get_mu_cond(
    x: ndarray,
    *,
    mu1: ndarray,
    mu2: ndarray,
    Sigma12: ndarray,
    Sigma22: ndarray,
):
    """Mean of distribution P(Y|X)."""
    assert np.sum(np.isnan(x)) == 0
    n_points = x.shape[0]
    x = x.reshape(n_points, -1).T
    mu_cond = mu1 + Sigma12.dot(linalg.inv(Sigma22)).dot(x - mu2)
    return mu_cond.T.reshape(n_points, -1)


def fit_multi_gauss(x: ndarray, y: ndarray):
    """Return parameters of a multivariate Gaussian fit on input."""
    n_features = x.shape[1]
    n_targets = y.shape[1]
    assert y.shape[0] == x.shape[0]

    z = np.hstack([y.reshape(-1, n_targets), x])

    # some sanity checks
    assert z.shape == (y.shape[0], n_targets + n_features)
    np.testing.assert_equal(y, z[:, :n_targets])
    np.testing.assert_equal(x[:, 0], z[:, n_targets])  # ignore mutual nan's
    np.testing.assert_equal(x[:, -1], z[:, -1])

    # calculate covariances
    total_features = n_targets + n_features
    Sigma = np.zeros((total_features, total_features))
    rho = np.zeros((total_features, total_features))
    for i in range(total_features):
        for j in range(total_features):
            if i <= j:
                # calculate correlation coefficient keeping only non-nan values
                z1, z2 = z[:, i], z[:, j]
                keep = ~np.isnan(z1) & ~np.isnan(z2)
                cov = np.cov(z1[keep], z2[keep])
                assert cov.shape == (2, 2)
                Sigma[i, j] = cov[0, 1]
                rho[i, j] = np.corrcoef(z1[keep], z2[keep])[0, 1]
            else:
                rho[i, j] = rho[j, i]
                Sigma[i, j] = Sigma[j, i]

    # more sanity checks.
    assert np.all(~np.isnan(Sigma))
    assert np.all(~np.isnan(rho))

    mu1 = np.nanmean(y, axis=0).reshape(n_targets, 1)
    mu2 = np.nanmean(x, axis=0).reshape(n_features, 1)
    Sigma11 = Sigma[:n_targets, :n_targets].reshape(n_targets, n_targets)
    Sigma12 = Sigma[:n_targets, n_targets:].reshape(n_targets, n_features)
    Sigma22 = Sigma[n_targets:, n_targets:].reshape(n_features, n_features)
    sigma_bar = Sigma11 - Sigma12.dot(np.linalg.solve(Sigma22, Sigma12.T))
    sigma_bar = sigma_bar.reshape(n_targets, n_targets)

    return {
        "mu1": mu1,
        "mu2": mu2,
        "Sigma11": Sigma11,
        "Sigma12": Sigma12,
        "Sigma22": Sigma22,
        "sigma_bar": sigma_bar,
        "Sigma": Sigma,
        "rho": rho,
    }
