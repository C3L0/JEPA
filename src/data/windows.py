from __future__ import annotations

import numpy as np

from .lerobot import EpisodeData
from .normalization import NormalizationBundle


class EpisodeAwareWindowDataset:
    """Samples context windows and future action sequences without crossing episodes.

    Returned action_sequence has fixed length max_horizon for batch compatibility.
    The first `horizon` entries are valid; action_mask marks those entries.

    Alignment convention: action[t] drives the transition state[t] -> state[t + 1].
    """

    def __init__(
        self,
        episodes: list[EpisodeData],
        normalizers: NormalizationBundle,
        context_length: int = 10,
        min_horizon: int = 5,
        max_horizon: int = 20,
        random_horizon: bool = True,
        fixed_horizon: int = 10,
        seed: int = 42,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be >= 1.")
        if min_horizon < 1 or max_horizon < min_horizon:
            raise ValueError("Require 1 <= min_horizon <= max_horizon.")
        if not random_horizon and not min_horizon <= fixed_horizon <= max_horizon:
            raise ValueError("fixed_horizon must be between min_horizon and max_horizon when random_horizon=False.")
        if not episodes:
            raise ValueError("episodes cannot be empty.")

        self.episodes = list(episodes)
        self.normalizers = normalizers
        self.context_length = int(context_length)
        self.min_horizon = int(min_horizon)
        self.max_horizon = int(max_horizon)
        self.random_horizon = bool(random_horizon)
        self.fixed_horizon = int(fixed_horizon)
        self._rng = np.random.default_rng(seed)

        state_dims = {ep.states.shape[1] for ep in self.episodes}
        action_dims = {ep.actions.shape[1] for ep in self.episodes}
        if len(state_dims) != 1 or len(action_dims) != 1:
            raise ValueError("All episodes must have equal state/action dimensions.")
        self.state_dim = next(iter(state_dims))
        self.action_dim = next(iter(action_dims))
        if len(normalizers.state.center) != self.state_dim:
            raise ValueError("State normalizer dimension does not match episode states.")
        if len(normalizers.action.center) != self.action_dim:
            raise ValueError("Action normalizer dimension does not match episode actions.")

        # Store valid anchors (episode index, t). Requiring max_horizon here ensures
        # every randomly selected horizon is valid for every anchor.
        self._anchors: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            first_t = self.context_length - 1
            stop_t = len(ep.states) - self.max_horizon
            self._anchors.extend((ep_idx, t) for t in range(first_t, stop_t))
        if not self._anchors:
            raise ValueError(
                "No valid windows. Reduce context_length/max_horizon or provide longer episodes."
            )

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        ep_idx, t = self._anchors[index]
        ep = self.episodes[ep_idx]
        horizon = (
            int(self._rng.integers(self.min_horizon, self.max_horizon + 1))
            if self.random_horizon
            else self.fixed_horizon
        )

        context_start = t - self.context_length + 1
        context = ep.states[context_start : t + 1]
        action_seq = ep.actions[t : t + horizon]
        target = ep.states[t + horizon]

        context = self.normalizers.state.transform(context)
        action_seq = self.normalizers.action.transform(action_seq)
        target = self.normalizers.state.transform(target)

        # Fixed-size sequence for ordinary batching; padded values must be masked.
        padded_actions = np.zeros((self.max_horizon, self.action_dim), dtype=np.float32)
        padded_actions[:horizon] = action_seq
        action_mask = np.zeros((self.max_horizon,), dtype=np.bool_)
        action_mask[:horizon] = True

        sample = {
            "state_window": context.astype(np.float32, copy=False),
            "action_sequence": padded_actions,
            "action_mask": action_mask,
            "target_state": target.astype(np.float32, copy=False),
            "horizon": np.asarray(horizon, dtype=np.int32),
            "episode_id": np.asarray(ep.episode_id, dtype=np.int64),
            "anchor_frame": np.asarray(ep.frame_indices[t], dtype=np.int64),
            "target_frame": np.asarray(ep.frame_indices[t + horizon], dtype=np.int64),
        }
        return sample

    def set_epoch_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)


def seed_worker(worker_id: int) -> None:
    """PyTorch DataLoader worker initializer; safe to use even with num_workers=0."""
    try:
        import torch
        from torch.utils.data import get_worker_info

        info = get_worker_info()
        if info is not None and hasattr(info.dataset, "set_epoch_seed"):
            info.dataset.set_epoch_seed(torch.initial_seed() % (2**32) + worker_id)
    except ImportError:
        return
