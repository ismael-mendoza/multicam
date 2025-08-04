import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from multicam.base import PredictionModel


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


def _get_an_from_am(am: NDArray, mass_bins: NDArray, mbin=0.498):
    """Return scale corresponding to first mass bin bigger than `mbin`."""
    idx = np.where(mass_bins > mbin)[0][0]
    return am[:, idx]


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
