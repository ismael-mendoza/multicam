"""Implementation of statistical models."""

import numpy as np
from numpy import linalg, ndarray
from numpy.random import default_rng
from scipy.stats import rankdata
from sklearn import linear_model

from multicam.qt import qt, qt_gauss, qt_gauss_base, qt_ranks_base


class MultiCAM:
    """MultiCAM model described in our first paper."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        assert isinstance(n_features, int) and n_features > 0
        assert isinstance(n_targets, int) and n_targets > 0
        self.n_features = n_features
        self.n_targets = n_targets
        self.x_train = None
        self.rank_lookup = None
        self.trained = False
        self.reg = None

    def fit(self, x: ndarray, y: ndarray) -> None:
        """Fit model using training data"""
        assert not self.trained
        assert x.ndim == 2 and y.ndim == 2
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        assert y.shape == (x.shape[0], self.n_targets)

        self.x_train = x.copy()

        # create lookup table for ranks in training features
        # useful esp. for features with repetitions.
        self.rank_lookup = _create_rank_lookup(self.x_train)

        # transform variables to be (marginally) gaussian and break ties.
        xg = qt_gauss(x, axis=0, method="ordinal")
        yg = qt_gauss(y, axis=0, method="ordinal")

        # then fit a linear regression model to the transformed data.
        self.reg = linear_model.LinearRegression()
        self.reg.fit(xg, yg)

        self.trained = True

    def predict(self, x: ndarray, *, y_target: ndarray) -> ndarray:
        # assume continuous data for now
        assert len(x) > 1, "MultiCAM works with distributions, not single data points."
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert y_target.shape[1] == self.n_targets
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        xg = _gaussianize_test_features(x, x_base=self.x_train, mode="middle")

        # predict gaussianized target with linear regression
        yg = self.reg.predict(xg)

        # KEY: finally we want to reproduce some final 'true' distribution
        # so we abundance match each corresponding target variable outputed from the LR prediction
        # Also, avoid repeats 'bunching up' to reproduce correct output distribution in ALL cases.
        # qt handles this internally by using "ordinal"
        return _abundance_match_all(yg, y_target)


class MultiCamSampling(MultiCAM):
    """Multi-Variate Gaussian w/ full covariance matrix (returns conditional mean)."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__(n_features, n_targets)

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

    def fit(self, x: ndarray, y: ndarray) -> None:
        """
        Fit the Gaussian model.

        We assume a multivariate-gaussian distribution P(X, Y) with conditional distribution
        P(Y | X) = uses the rules here:
        https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
        """

        # see parent class procedure
        self.x_train = x.copy()
        self.rank_lookup = _create_rank_lookup(self.x_train)
        xg = qt_gauss(x, axis=0, method="ordinal")
        yg = qt_gauss(y, axis=0, method="ordinal")

        fit_params = _fit_multi_gauss(xg, yg)

        # update prediction attributes
        for k in fit_params:
            assert getattr(self, k) is None
            setattr(self, k, fit_params[k])

        self.trained = True

    def sample(
        self, x: ndarray, *, y_target: ndarray, seed: int | None = None
    ) -> ndarray:
        """Sample (once) from the conditional distribution P(y | x)"""
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        rng = default_rng(42) if seed is None else default_rng(seed)
        n_points = x.shape[0]

        xg = self._gaussianize_test_features(x, mode="random")

        # sample on gaussianized ranks.
        _zero = np.zeros((self.n_targets,))
        mu_cond = _get_mu_cond(
            xg, mu1=self.mu1, mu2=self.mu2, Sigma12=self.Sigma12, Sigma22=self.Sigma22
        )
        y_gauss = rng.multivariate_normal(
            mean=_zero, cov=self.sigma_bar, size=(n_points,)
        )
        assert y_gauss.shape == (n_points, self.n_targets)
        y_gauss += mu_cond

        # abundance match on sampled y_gauss
        # ALWAYS follow target distribution marginally
        y_samples = _abundance_match_all(y_gauss, y_target)

        return y_samples


def _create_rank_lookup(x) -> dict:
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


def _get_ranks_based(
    x: ndarray, x_base: ndarray, rank_lookup: dict, mode: str = "middle"
) -> ndarray:
    assert mode in {"middle", "random"}
    assert x.ndim == 2
    assert x_base.ndim == 2
    n_features = x.shape[1]

    # start by interpolating ranks naively
    xr = qt_ranks_base(x, x_base)

    # if value is in training data, get middle or random rank
    for jj in range(n_features):
        x_jj = x[:, jj]
        uniq, lranks, hranks = rank_lookup[jj]

        in_train = np.isin(x_jj, uniq)
        u_indices = np.searchsorted(uniq, x_jj[in_train])
        lr, hr = lranks[u_indices], hranks[u_indices]  # repeat appropriately
        xr[in_train, jj] = (
            np.random.randint(lr, hr + 1) if mode == "random" else (lr + hr) / 2
        )

    assert np.sum(np.isnan(xr)) == 0
    return xr


def _gaussianize_test_features(x: ndarray, *, x_base: ndarray, mode: str) -> ndarray:
    _rank_lookup = _create_rank_lookup(x_base)
    xr = _get_ranks_based(x, x_base, _rank_lookup, mode=mode)
    xrt = rankdata(x_base, axis=0, method="ordinal")
    return qt_gauss_base(xr, xrt)


def _abundance_match_all(x: ndarray, y: ndarray) -> ndarray:
    """Abundance match each dimension separately."""
    assert x.shape[1] == y.shape[1]
    n_targets = x.shape[1]
    xp = np.full_like(x, fill_value=np.nan)
    for ii in range(n_targets):
        xp[:, ii] = qt(x[:, ii], y[:, ii])
    return xp


def _get_mu_cond(
    x: ndarray,
    *,
    mu1: ndarray,
    mu2: ndarray,
    Sigma12: ndarray,
    Sigma22: ndarray,
) -> ndarray:
    """Mean of distribution P(Y|X)."""
    assert np.sum(np.isnan(x)) == 0
    n_points = x.shape[0]
    x = x.reshape(n_points, -1).T
    mu_cond = mu1 + Sigma12.dot(linalg.inv(Sigma22)).dot(x - mu2)
    return mu_cond.T.reshape(n_points, -1)


def _fit_multi_gauss(x: ndarray, y: ndarray) -> dict[str, ndarray]:
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
