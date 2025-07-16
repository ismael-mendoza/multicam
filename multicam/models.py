"""Implementation of statistical models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy import random
from scipy.interpolate import interp1d
from sklearn import linear_model

from multicam.qt import qt, qt_gauss


def _get_an_from_am(am, mass_bins, mbin=0.498):
    """Return scale corresponding to first mass bin bigger than `mbin`."""
    idx = np.where(mass_bins > mbin)[0][0]
    return am[:, idx]


class PredictionModel(ABC):
    """Abstract base class for prediction models."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        assert isinstance(n_features, int) and n_features > 0
        assert isinstance(n_targets, int) and n_targets > 0

        self.n_features = n_features
        self.n_targets = n_targets
        self.trained = False  # whether model has been trained yet.

    def fit(self, x, y):
        """Fit model using training data."""
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        assert y.shape == (x.shape[0], self.n_targets)
        self._fit(x, y)
        self.trained = True

    def predict(self, x):
        """Predict y given x."""
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained
        return self._predict(x).reshape(x.shape[0], self.n_targets)

    @abstractmethod
    def _fit(self, x, y):
        pass

    @abstractmethod
    def _predict(self, x):
        pass


class MultiCAM(PredictionModel):
    """MultiCAM model described in our first paper."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        super().__init__(n_features, n_targets)

        # additional metadata that needs to be saved for prediction.
        self.qt_xr = None
        self.qt_yr = None
        self.qt_pred = None
        self.x_train = None
        self.y_train = None
        self.rank_lookup = {}

        # setup linear regression model
        self.reg = linear_model.LinearRegression()

    def _fit(self, x, y):
        """Fit model using training data"""
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        assert y.shape == (x.shape[0], self.n_targets)

        # need to save training data to predict from ranks later.
        self.x_train = x.copy()
        self.y_train = y.copy()

        # transform variables to be (marginally) gaussian.
        # default version accounts for repeated values
        x_gauss = qt_gauss(x, axis=0)
        y_gauss = qt_gauss(y, axis=0)

        # then fit a linear regression model to the transformed data.
        self.reg.fit(x_gauss, y_gauss)

        # finally, create lookup table for low and high ranks of each feature.
        for jj in range(self.n_features):
            x_train_jj = np.sort(self.x_train[:, jj])
            u, c = np.unique(x_train_jj, return_counts=True)
            lranks = np.cumsum(c) - c + 1
            hranks = np.cumsum(c)
            self.rank_lookup[jj] = (u, lranks, hranks)

        return x_gauss, y_gauss

    def _get_ranks(self, x, rank_mode="middle"):
        """Get ranks fo test data based on training data ranks."""
        assert rank_mode in {"middle", "random"}

        xr = np.zeros_like(x) * np.nan
        for jj in range(self.n_features):
            x_jj = x[:, jj]
            x_train_jj = np.sort(self.x_train[:, jj])
            uniq, lranks, hranks = self.rank_lookup[jj]
            xr[:, jj] = np.searchsorted(x_train_jj, x_jj) + 1  # indices to ranks

            # if value is in training data, get middle or random rank
            in_train = np.isin(x_jj, uniq)
            u_indices = np.searchsorted(uniq, x_jj[in_train])
            lr, hr = lranks[u_indices], hranks[u_indices]  # repeat appropriately
            _xr = random.randint(lr, hr + 1) if rank_mode == "random" else (lr + hr) / 2
            xr[in_train, jj] = _xr

        assert np.sum(np.isnan(xr)) == 0
        return xr

    def _predict(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        # get ranks of testing data corresponding to
        xr = self._get_ranks(x, rank_mode="middle")

        # transform ranks to be (marginally) gaussian.
        # for 'middle' we actually want to output exactly the same y if x repeats
        x_gauss = qt_gauss(xr, axis=0, method="average")

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
        self.Sigma = None
        self.rho = None
        self.sigma_cond = None
        self.Sigma11 = None
        self.Sigma12 = None
        self.Sigma22 = None
        self.sigma_bar = None

    def _fit(self, x, y):
        """
        We assume a multivariate-gaussian distribution P(X, Y) with conditional distribution
        P(Y | X) = uses the rules here:
        https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution
        """
        xg, yg = super()._fit(x, y)  # gaussianized x and y

        n_features = self.n_features
        n_targets = self.n_targets

        z = np.hstack([yg.reshape(-1, n_targets), xg])

        # some sanity checks
        assert z.shape == (yg.shape[0], n_targets + n_features)
        np.testing.assert_equal(yg, z[:, :n_targets])
        np.testing.assert_equal(xg[:, 0], z[:, n_targets])  # ignore mutual nan's
        np.testing.assert_equal(xg[:, -1], z[:, -1])

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

        mu1 = np.nanmean(yg, axis=0).reshape(n_targets, 1)
        mu2 = np.nanmean(xg, axis=0).reshape(n_features, 1)
        Sigma11 = Sigma[:n_targets, :n_targets].reshape(n_targets, n_targets)
        Sigma12 = Sigma[:n_targets, n_targets:].reshape(n_targets, n_features)
        Sigma22 = Sigma[n_targets:, n_targets:].reshape(n_features, n_features)
        sigma_bar = Sigma11 - Sigma12.dot(np.linalg.solve(Sigma22, Sigma12.T))

        # update prediction attributes
        self.mu1 = mu1
        self.mu2 = mu2
        self.Sigma = Sigma
        self.Sigma11 = Sigma11
        self.Sigma12 = Sigma12
        self.Sigma22 = Sigma22
        self.rho = rho
        self.sigma_bar = sigma_bar.reshape(n_targets, n_targets)

    def _get_mu_cond(self, x):
        # returns mu_cond evaluated at given x (gaussianized).
        assert self.trained
        assert np.sum(np.isnan(x)) == 0
        n_points = x.shape[0]
        x = x.reshape(n_points, self.n_features).T
        mu_cond = self.mu1 + self.Sigma12.dot(np.linalg.inv(self.Sigma22)).dot(
            x - self.mu2
        )
        return mu_cond.T.reshape(n_points, self.n_targets)

    def sample(self, x):
        """Sample (once) from the conditional distribution P(y | x)"""
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_features
        assert np.sum(np.isnan(x)) == 0
        assert self.trained

        n_points = x.shape[0]

        # get ranks of test data (based on training data)
        # use 'random' mode to account for uncertainty in ranks for each test point's features.
        xr = self._get_ranks(x, mode="random")
        x_gauss = qt_gauss(xr, axis=0, method="ordinal")  # TODO: method?

        # sample on gaussianized ranks.
        _zero = np.zeros((self.n_targets,))
        mu_cond = self._get_mu_cond(x_gauss)
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


class CAM(PredictionModel):
    """Conditional Abundance Matching"""

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        mass_bins: np.ndarray,
        opt_mbin: float,
        cam_order: int = -1,
    ) -> None:
        # cam_order: +1 or -1 depending on correlation of a_{n} with y
        assert n_features == len(mass_bins)
        assert n_targets == 1
        super().__init__(n_features, n_targets)

        assert cam_order in {-1, 1}
        assert isinstance(mass_bins, np.ndarray)
        self.opt_mbin = opt_mbin
        self.cam_order = cam_order
        self.mass_bins = mass_bins

        # fit attributes
        self.an_to_mark = None
        self.mark_to_Y = None

    def _fit(self, x, y):
        am = x
        y = y.reshape(-1)
        an_train = _get_an_from_am(am, self.mass_bins, mbin=self.opt_mbin).reshape(-1)
        assert an_train.shape[0] == am.shape[0]

        y_sort, an_sort = (
            self.cam_order * np.sort(self.cam_order * y),
            np.sort(an_train),
        )
        marks = np.arange(len(y_sort)) / len(y_sort)
        marks += (marks[1] - marks[0]) / 2
        self.an_to_mark = interp1d(
            an_sort, marks, fill_value=(0, 1), bounds_error=False
        )
        self.mark_to_Y = interp1d(
            marks, y_sort, fill_value=(y_sort[0], y_sort[-1]), bounds_error=False
        )

    def _predict(self, x):
        am = x
        an = _get_an_from_am(am, self.mass_bins, mbin=self.opt_mbin)
        return self.mark_to_Y(self.an_to_mark(an))


class MixedCAM(PredictionModel):
    """CAM but w/ multiple indepent CAMs inside to allow multiple predictors."""

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        mass_bins: np.ndarray,
        opt_mbins: tuple,
        cam_orders: tuple = (-1,),
    ) -> None:
        assert n_features == len(mass_bins)
        assert n_targets > 1, "Use 'CAM' instead"
        super().__init__(n_features, n_targets)

        # create `n_targets` independent CAMs.
        self.cams = [
            CAM(n_features, 1, mass_bins, opt_mbins[ii], cam_orders[ii])
            for ii in range(n_targets)
        ]

    def _fit(self, x, y):
        for jj in range(self.n_targets):
            self.cams[jj].fit(x, y[:, jj].reshape(-1, 1))

    def _predict(self, x):
        y_pred = []
        for jj in range(self.n_targets):
            y = self.cams[jj].predict(x)
            y_pred.append(y)
        return np.hstack(y_pred)


def training_suite(info: dict):
    """Returned models specified in the data dictionary.

    Args:
        info:  Dictionary containing all the information required to train models. Using the format
            `name:value` where `name` is an identifier for the model (can be anything)
            and `value` is a dictionary with keys:
                - 'xy': (x,y) tuple containing data to train model with.
                - 'model': Which model from `available_models` to use.
                - 'n_features': Number of features for this model.
                - 'kwargs': Keyword argument dict to initialize the model.
    """
    # check data dict is in the right format.
    assert isinstance(info, dict)
    for name in info:
        assert isinstance(info[name]["xy"], tuple)
        assert info[name]["model"] in available_models
        assert isinstance(info[name]["n_features"], int)
        assert isinstance(info[name]["n_targets"], int)
        assert isinstance(info[name]["kwargs"], dict)

    trained_models = {}
    for name in info:
        m = info[name]["model"]
        kwargs = info[name]["kwargs"]
        n_features = info[name]["n_features"]
        n_targets = info[name]["n_targets"]
        x, y = info[name]["xy"]
        model = available_models[m](n_features, n_targets, **kwargs)
        model.fit(x, y)
        trained_models[name] = model

    return trained_models


def _tbl_to_arr(table, names=None):
    if not names:
        names = table.colnames

    return np.hstack([table[param].reshape(-1, 1) for param in names])


def _get_tt_indices(n_points, rng=None, test_ratio=0.2):
    if rng is None:
        rng = np.random.default_rng(0)
    test_size = int(np.ceil(test_ratio * n_points))
    test_idx = rng.choice(range(n_points), replace=False, size=test_size)
    assert len(test_idx) == len(set(test_idx))
    train_idx = np.array(list(set(range(n_points)) - set(test_idx)))
    assert set(train_idx).intersection(set(test_idx)) == set()
    assert max(test_idx.max(), train_idx.max()) == n_points - 1
    assert min(test_idx.min(), train_idx.min()) == 0
    return train_idx, test_idx


def prepare_datasets(cat, datasets: dict, rng, test_ratio=0.3):
    """Prepare datasets for training and testing."""

    # train/test split
    train_idx, test_idx = _get_tt_indices(len(cat), rng, test_ratio=test_ratio)
    cat_train, cat_test = cat[train_idx], cat[test_idx]
    output = {}
    for name in datasets:
        x_params, y_params = datasets[name]["x"], datasets[name]["y"]
        x_train, x_test = (
            _tbl_to_arr(cat_train, x_params),
            _tbl_to_arr(cat_test, x_params),
        )
        y_train, y_test = (
            _tbl_to_arr(cat_train, y_params),
            _tbl_to_arr(cat_test, y_params),
        )
        output[name] = {"train": (x_train, y_train), "test": (x_test, y_test)}
    return output, train_idx, test_idx


available_models = {
    "linear": MultiCAM,
    "gaussian": MultiCamSampling,
    "cam": CAM,
    "mixed_cam": MixedCAM,
}
