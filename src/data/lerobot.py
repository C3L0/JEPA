from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeData:
    """One episode of aligned observations and actions.

    Convention used by the window dataset:
        action[t] is the command applied from observation[t] to observation[t + 1].
    """

    episode_id: int
    states: np.ndarray
    actions: np.ndarray
    frame_indices: np.ndarray
    timestamps: np.ndarray | None = None

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError("states and actions must both have shape [frames, features].")
        if len(states) != len(actions) or len(states) != len(frame_indices):
            raise ValueError("states, actions and frame_indices must have the same length.")
        if len(states) == 0:
            raise ValueError(f"Episode {self.episode_id} is empty.")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Episode {self.episode_id} contains NaN or infinite values.")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "frame_indices", frame_indices)
        if self.timestamps is not None:
            timestamps = np.asarray(self.timestamps, dtype=np.float32)
            if len(timestamps) != len(states):
                raise ValueError("timestamps must have the same length as states.")
            object.__setattr__(self, "timestamps", timestamps)


def _stack_feature(values: Iterable[object], feature_name: str) -> np.ndarray:
    rows = []
    for row_idx, value in enumerate(values):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"Empty value in {feature_name} at row {row_idx}.")
        rows.append(arr)
    try:
        result = np.stack(rows, axis=0).astype(np.float32, copy=False)
    except ValueError as exc:
        raise ValueError(f"Inconsistent vector sizes in {feature_name}.") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{feature_name} contains NaN or infinite values.")
    return result


def _find_parquet_files(root: Path) -> list[Path]:
    candidates = sorted(root.rglob("*.parquet"))
    # Keep only data shards; ignore unrelated evaluation/cache outputs if present.
    candidates = [p for p in candidates if "data" in p.parts or p.parent == root]
    if not candidates:
        raise FileNotFoundError(
            f"No parquet files found under {root}. Expected LeRobot data/**/*.parquet."
        )
    return candidates


def load_lerobot_episodes(
    root: str | Path,
    state_key: str = "observation.state",
    action_key: str = "action",
    episode_key: str = "episode_index",
    frame_key: str = "frame_index",
    timestamp_key: str = "timestamp",
    expected_state_dim: int | None = 14,
    expected_action_dim: int | None = 14,
) -> list[EpisodeData]:
    """Load state/action vectors from LeRobot parquet shards.

    This loader intentionally handles the state/action-only parquet representation.
    Camera videos, when present, are not decoded by this M2 loader.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    frames = []
    for parquet_path in _find_parquet_files(root):
        frame = pd.read_parquet(parquet_path)
        required = {state_key, action_key, episode_key, frame_key}
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(
                f"Missing columns {sorted(missing)} in {parquet_path}. "
                f"Available columns: {list(frame.columns)}"
            )
        keep = [state_key, action_key, episode_key, frame_key]
        if timestamp_key in frame.columns:
            keep.append(timestamp_key)
        frames.append(frame[keep])

    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values([episode_key, frame_key], kind="stable").reset_index(drop=True)
    if table.duplicated([episode_key, frame_key]).any():
        duplicates = table.loc[table.duplicated([episode_key, frame_key], keep=False), [episode_key, frame_key]]
        raise ValueError(f"Duplicate episode/frame rows found:\n{duplicates.head(10)}")

    episodes: list[EpisodeData] = []
    for episode_id, group in table.groupby(episode_key, sort=True):
        group = group.sort_values(frame_key, kind="stable")
        states = _stack_feature(group[state_key].tolist(), state_key)
        actions = _stack_feature(group[action_key].tolist(), action_key)
        frame_indices = group[frame_key].to_numpy(dtype=np.int64)
        if expected_state_dim is not None and states.shape[1] != expected_state_dim:
            raise ValueError(
                f"Episode {episode_id}: expected state dimension {expected_state_dim}, got {states.shape[1]}."
            )
        if expected_action_dim is not None and actions.shape[1] != expected_action_dim:
            raise ValueError(
                f"Episode {episode_id}: expected action dimension {expected_action_dim}, got {actions.shape[1]}."
            )
        if len(frame_indices) > 1 and not np.all(np.diff(frame_indices) == 1):
            raise ValueError(
                f"Episode {episode_id} has non-contiguous frame indices. "
                "Do not construct temporal windows across missing frames."
            )
        timestamps = (
            group[timestamp_key].to_numpy(dtype=np.float32)
            if timestamp_key in group.columns
            else None
        )
        episodes.append(
            EpisodeData(
                episode_id=int(episode_id),
                states=states,
                actions=actions,
                frame_indices=frame_indices,
                timestamps=timestamps,
            )
        )

    if not episodes:
        raise ValueError(f"No episodes were loaded from {root}.")
    state_dims = {ep.states.shape[1] for ep in episodes}
    action_dims = {ep.actions.shape[1] for ep in episodes}
    if len(state_dims) != 1 or len(action_dims) != 1:
        raise ValueError("All episodes must have consistent state and action dimensions.")
    return episodes


def split_episode_ids(
    episodes: list[EpisodeData],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split by episode (never by individual frame) to avoid temporal leakage."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train_fraction and validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")
    ids = np.asarray([ep.episode_id for ep in episodes], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Episode IDs must be unique.")
    if len(ids) < 3:
        raise ValueError("At least 3 episodes are required for train/validation/test splits.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_train = max(1, int(round(len(ids) * train_fraction)))
    n_val = max(1, int(round(len(ids) * validation_fraction)))
    if n_train + n_val >= len(ids):
        n_train = len(ids) - 2
        n_val = 1
    return {
        "train": sorted(shuffled[:n_train].tolist()),
        "validation": sorted(shuffled[n_train : n_train + n_val].tolist()),
        "test": sorted(shuffled[n_train + n_val :].tolist()),
    }
