from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

REQUIRED_KEYS = (
    "state_window",
    "action_sequence",
    "action_mask",
    "target_state",
    "horizon",
)


@dataclass
class WindowArrays:
    state_window: np.ndarray
    action_sequence: np.ndarray
    action_mask: np.ndarray
    target_state: np.ndarray
    horizon: np.ndarray

    def __len__(self) -> int:
        return int(self.state_window.shape[0])

    def validate(self) -> None:
        n = len(self)
        if self.state_window.ndim != 3:
            raise ValueError(f"state_window must be [N, C, S], got {self.state_window.shape}")
        if self.action_sequence.ndim != 3:
            raise ValueError(f"action_sequence must be [N, H, A], got {self.action_sequence.shape}")
        if self.action_mask.ndim != 2:
            raise ValueError(f"action_mask must be [N, H], got {self.action_mask.shape}")
        if self.target_state.ndim != 2:
            raise ValueError(f"target_state must be [N, S], got {self.target_state.shape}")
        if self.horizon.ndim != 1:
            raise ValueError(f"horizon must be [N], got {self.horizon.shape}")
        if any(x.shape[0] != n for x in self.as_dict().values()):
            raise ValueError("All arrays must have the same number of samples.")
        if self.action_sequence.shape[:2] != self.action_mask.shape:
            raise ValueError("action_sequence and action_mask must agree on [N, H].")
        if self.state_window.shape[-1] != self.target_state.shape[-1]:
            raise ValueError("State dimension differs between context and target.")
        if not np.isfinite(self.state_window).all():
            raise ValueError("state_window contains NaN or infinite values.")
        if not np.isfinite(self.action_sequence).all():
            raise ValueError("action_sequence contains NaN or infinite values.")
        if not np.isfinite(self.target_state).all():
            raise ValueError("target_state contains NaN or infinite values.")
        if not np.isfinite(self.horizon).all():
            raise ValueError("horizon contains NaN or infinite values.")
        if np.any(self.horizon < 1) or np.any(self.horizon > self.action_sequence.shape[1]):
            raise ValueError("Each horizon must be between 1 and padded max horizon.")
        if np.any((self.action_mask != 0) & (self.action_mask != 1)):
            raise ValueError("action_mask must contain only zeros and ones.")
        if np.any(self.action_mask.sum(axis=1) != self.horizon):
            raise ValueError("The number of valid actions must equal horizon for each sample.")

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "state_window": self.state_window,
            "action_sequence": self.action_sequence,
            "action_mask": self.action_mask,
            "target_state": self.target_state,
            "horizon": self.horizon,
        }

    @classmethod
    def from_npz(cls, path: str | Path) -> "WindowArrays":
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in REQUIRED_KEYS if key not in data]
            if missing:
                raise KeyError(f"Missing keys in {path}: {missing}")
            result = cls(**{key: np.asarray(data[key]) for key in REQUIRED_KEYS})
        result.validate()
        return result


def save_window_samples(samples: Sequence[Mapping[str, object]], path: str | Path) -> None:
    """Save M2 window dictionaries to the NPZ format consumed by M4.

    Each item must contain the five REQUIRED_KEYS. Arrays are stacked on axis 0.
    """
    if not samples:
        raise ValueError("Cannot save an empty sample collection.")
    arrays = {
        key: np.stack([np.asarray(sample[key]) for sample in samples], axis=0)
        for key in REQUIRED_KEYS
    }
    WindowArrays(**arrays).validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def iter_batches(
    dataset: WindowArrays,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    drop_last: bool = True,
) -> Iterable[dict[str, np.ndarray]]:
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 for VICReg batch statistics.")
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    arrays = dataset.as_dict()
    stop = len(indices) if not drop_last else (len(indices) // batch_size) * batch_size
    for start in range(0, stop, batch_size):
        selected = indices[start : min(start + batch_size, stop)]
        if len(selected) < 2:
            continue
        yield {key: value[selected] for key, value in arrays.items()}
