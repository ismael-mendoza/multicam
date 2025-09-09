from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class PredictionModel(ABC):
    """Abstract base class for prediction models."""

    def __init__(self, n_features: int, n_targets: int) -> None:
        assert isinstance(n_features, int) and n_features > 0
        assert isinstance(n_targets, int) and n_targets > 0

        self.n_features = n_features
        self.n_targets = n_targets
        self.trained = False  # whether model has been trained yet.

    def fit(self, x: NDArray, y: NDArray):
        """Fit model using training data."""
        assert x.ndim == 2 and y.ndim == 2
        assert np.sum(np.isnan(x)) == np.sum(np.isnan(y)) == 0
        assert x.shape == (y.shape[0], self.n_features)
        assert y.shape == (x.shape[0], self.n_targets)
        self._fit(x, y)
        self.trained = True

    def predict(self, x: NDArray):
        """Predict y given x."""
        assert x.ndim == 2
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
