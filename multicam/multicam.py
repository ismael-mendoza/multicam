"""Implementation of statistical models."""

import numpy as np
from numpy import linalg
from numpy.typing import NDArray
from sklearn import linear_model

from multicam.base import PredictionModel
from multicam.qt import qt, qt_gauss


class MultiCAM(PredictionModel):
    """MultiCAM model described in our first paper."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__(n_features, n_targets)
        self.reg = linear_model.LinearRegression()

    def _fit(self, x, y):
        """Fit model using training data"""
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        assert y.shape == (x.shape[0], self.n_targets)

        # transform variables to be (marginally) gaussian.
        x_gauss = qt_gauss(x, axis=0)
        y_gauss = qt_gauss(y, axis=0)

        # then fit a linear regression model to the transformed data.
        self.reg.fit(x_gauss, y_gauss)

        return x_gauss, y_gauss

    def _predict(self, x, method="ordinal"):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        # transform ranks to be (marginally) gaussian.
        x_gauss = qt_gauss(x, axis=0, method=method)

        # predict on transformed ranks.
        y_not_gauss = self.reg.predict(x_gauss)

        # now we just need to collect the elements of `y_train` that have the same
        # corresponding rank as `y_not_gauss` (per feature)
        y_pred = np.zeros((x.shape[0], self.n_features))
        for jj in range(self.n_features):
            y_pred[:, jj] = qt(y_not_gauss[:, jj], self.y_train[:, jj])

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

        # gaussianize input data
        x_gauss = qt_gauss(x, axis=0)

        # sample on gaussianized ranks.
        _zero = np.zeros((self.n_targets,))
        mu_cond = get_mu_cond(
            x_gauss,
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

        # get samples
        y_samples = np.zeros((x.shape[0], self.n_features))
        for jj in range(self.n_features):
            y_samples[:, jj] = qt(y_gauss[:, jj], self.y_train[:, jj])

        return y_samples


def get_mu_cond(
    x: NDArray,
    *,
    mu1: NDArray,
    mu2: NDArray,
    Sigma12: NDArray,
    Sigma22: NDArray,
):
    """Mean of distribution P(Y|X)."""
    assert np.sum(np.isnan(x)) == 0
    n_points = x.shape[0]
    x = x.reshape(n_points, -1).T
    mu_cond = mu1 + Sigma12.dot(linalg.inv(Sigma22)).dot(x - mu2)
    return mu_cond.T.reshape(n_points, -1)


def fit_multi_gauss(x: NDArray, y: NDArray):
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
