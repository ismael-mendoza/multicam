"""Functions to ease training of models."""

import numpy as np

from multicam.cam import CAM, MixedCAM
from multicam.multicam import MultiCAM, MultiCamSampling

available_models = {
    "linear": MultiCAM,
    "gaussian": MultiCamSampling,
    "cam": CAM,
    "mixed_cam": MixedCAM,
}


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


def get_tt_indices(n_points: int, rng=None, test_ratio=0.2):
    if rng is None:
        rng = np.random.default_rng()
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
    train_idx, test_idx = get_tt_indices(len(cat), rng, test_ratio=test_ratio)
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


def _tbl_to_arr(table, names=None):
    if not names:
        names = table.colnames

    return np.hstack([table[param].reshape(-1, 1) for param in names])
