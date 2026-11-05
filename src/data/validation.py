from __future__ import annotations

import numpy as np


REQUIRED_KEYS = {
    "state_window",
    "action_sequence",
    "action_mask",
    "target_state",
    "horizon",
}


def validate_sample(sample: dict, context_length: int, max_horizon: int,
                     state_dim: int, action_dim: int) -> None:
    missing = REQUIRED_KEYS.difference(sample)
    if missing:
        raise KeyError(f"Sample is missing keys: {sorted(missing)}")
    expected = {
        "state_window": (context_length, state_dim),
        "action_sequence": (max_horizon, action_dim),
        "action_mask": (max_horizon,),
        "target_state": (state_dim,),
    }
    for key, shape in expected.items():
        actual = np.asarray(sample[key]).shape
        if actual != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {actual}.")
    horizon = int(np.asarray(sample["horizon"]).item())
    if not 1 <= horizon <= max_horizon:
        raise ValueError(f"horizon must be in [1, {max_horizon}], got {horizon}.")
    mask = np.asarray(sample["action_mask"], dtype=bool)
    if int(mask.sum()) != horizon or not np.array_equal(mask, np.arange(max_horizon) < horizon):
        raise ValueError("action_mask must contain exactly `horizon` leading True values.")
    for key in ("state_window", "action_sequence", "target_state"):
        if not np.isfinite(np.asarray(sample[key])).all():
            raise ValueError(f"{key} contains NaN or infinite values.")


def validate_batch(batch: dict, context_length: int, max_horizon: int,
                   state_dim: int, action_dim: int) -> None:
    missing = REQUIRED_KEYS.difference(batch)
    if missing:
        raise KeyError(f"Batch is missing keys: {sorted(missing)}")
    state_window = np.asarray(batch["state_window"])
    if state_window.ndim != 3:
        raise ValueError(f"state_window must be [B,C,Ds], got {state_window.shape}.")
    batch_size = state_window.shape[0]
    expected = {
        "state_window": (batch_size, context_length, state_dim),
        "action_sequence": (batch_size, max_horizon, action_dim),
        "action_mask": (batch_size, max_horizon),
        "target_state": (batch_size, state_dim),
        "horizon": (batch_size,),
    }
    for key, shape in expected.items():
        actual = np.asarray(batch[key]).shape
        if actual != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {actual}.")
    horizons = np.asarray(batch["horizon"]).reshape(-1)
    masks = np.asarray(batch["action_mask"], dtype=bool)
    if np.any(horizons < 1) or np.any(horizons > max_horizon):
        raise ValueError(f"Every horizon must be in [1, {max_horizon}].")
    expected_masks = np.arange(max_horizon)[None, :] < horizons[:, None]
    if not np.array_equal(masks, expected_masks):
        raise ValueError("action_mask does not match the batch horizons.")
    for key in ("state_window", "action_sequence", "target_state"):
        if not np.isfinite(np.asarray(batch[key])).all():
            raise ValueError(f"{key} contains NaN or infinite values.")


def to_jax_batch(batch: dict) -> dict:
    """Convert numeric batch values to JAX arrays, preserving integer/bool types."""
    import jax.numpy as jnp

    return {key: jnp.asarray(value) for key, value in batch.items()}
