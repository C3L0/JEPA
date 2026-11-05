from __future__ import annotations

from dataclasses import dataclass

from .lerobot import EpisodeData, split_episode_ids
from .normalization import NormalizationBundle
from .windows import EpisodeAwareWindowDataset


@dataclass
class DataPipeline:
    train_dataset: EpisodeAwareWindowDataset
    validation_dataset: EpisodeAwareWindowDataset
    test_dataset: EpisodeAwareWindowDataset
    normalizers: NormalizationBundle
    episode_split: dict[str, list[int]]


def build_data_pipeline(
    episodes: list[EpisodeData],
    context_length: int = 10,
    min_horizon: int = 5,
    max_horizon: int = 20,
    eval_horizon: int = 10,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    normalization: str = "standard",
    normalization_clip: float | None = 10.0,
    seed: int = 42,
) -> DataPipeline:
    split = split_episode_ids(
        episodes,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    by_id = {ep.episode_id: ep for ep in episodes}
    train_eps = [by_id[i] for i in split["train"]]
    val_eps = [by_id[i] for i in split["validation"]]
    test_eps = [by_id[i] for i in split["test"]]

    # Fit normalization only on training episodes to avoid validation/test leakage.
    normalizers = NormalizationBundle.fit(
        train_eps, method=normalization, clip=normalization_clip
    )
    common = dict(
        normalizers=normalizers,
        context_length=context_length,
        min_horizon=min_horizon,
        max_horizon=max_horizon,
    )
    train_dataset = EpisodeAwareWindowDataset(
        train_eps, random_horizon=True, seed=seed, **common
    )
    validation_dataset = EpisodeAwareWindowDataset(
        val_eps, random_horizon=False, fixed_horizon=eval_horizon, seed=seed + 1, **common
    )
    test_dataset = EpisodeAwareWindowDataset(
        test_eps, random_horizon=False, fixed_horizon=eval_horizon, seed=seed + 2, **common
    )
    return DataPipeline(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        normalizers=normalizers,
        episode_split=split,
    )
