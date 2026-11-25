import numpy as np
import pytest

from ajeppa_m4.data import WindowArrays


def make_arrays(n=4, context=10, horizon_max=20, state_dim=14, action_dim=14):
    horizons = np.array([5, 10, 15, 20][:n], dtype=np.float32)
    mask = np.zeros((n, horizon_max), dtype=np.float32)
    for i, h in enumerate(horizons.astype(int)):
        mask[i, :h] = 1.0
    return WindowArrays(
        state_window=np.zeros((n, context, state_dim), dtype=np.float32),
        action_sequence=np.zeros((n, horizon_max, action_dim), dtype=np.float32),
        action_mask=mask,
        target_state=np.zeros((n, state_dim), dtype=np.float32),
        horizon=horizons,
    )


def test_valid_window_arrays_pass():
    make_arrays().validate()


def test_bad_horizon_mask_is_rejected():
    arrays = make_arrays()
    arrays.action_mask[0, 0] = 0.0
    with pytest.raises(ValueError, match="valid actions"):
        arrays.validate()
