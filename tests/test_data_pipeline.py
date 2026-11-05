import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.lerobot import EpisodeData
from data.normalization import NormalizationBundle
from data.windows import EpisodeAwareWindowDataset
from data.validation import validate_sample, validate_batch
from data.pipeline import build_data_pipeline


def make_episode(episode_id: int, length: int = 100):
    rng = np.random.default_rng(episode_id)
    states = rng.normal(size=(length, 14)).astype(np.float32)
    actions = rng.normal(size=(length, 14)).astype(np.float32)
    return EpisodeData(
        episode_id=episode_id,
        states=states,
        actions=actions,
        frame_indices=np.arange(length),
    )


def test_window_shapes_and_horizon_mask():
    episodes = [make_episode(0), make_episode(1)]
    normalizers = NormalizationBundle.fit(episodes)
    dataset = EpisodeAwareWindowDataset(
        episodes, normalizers, context_length=10, min_horizon=5,
        max_horizon=20, random_horizon=True, seed=1
    )
    sample = dataset[0]
    validate_sample(sample, 10, 20, 14, 14)
    assert sample["state_window"].shape == (10, 14)
    assert sample["action_sequence"].shape == (20, 14)
    assert int(sample["action_mask"].sum()) == int(sample["horizon"])


def test_no_window_crosses_episode():
    episode = make_episode(7, length=40)
    norm = NormalizationBundle.fit([episode])
    ds = EpisodeAwareWindowDataset(
        [episode], norm, context_length=4, min_horizon=3,
        max_horizon=5, random_horizon=False, fixed_horizon=5
    )
    for i in range(len(ds)):
        sample = ds[i]
        assert int(sample["target_frame"]) - int(sample["anchor_frame"]) == 5
        assert int(sample["episode_id"]) == 7


def test_episode_split_pipeline():
    episodes = [make_episode(i) for i in range(10)]
    pipeline = build_data_pipeline(
        episodes, context_length=5, min_horizon=2, max_horizon=8,
        eval_horizon=4, seed=123
    )
    train_ids = set(pipeline.episode_split["train"])
    val_ids = set(pipeline.episode_split["validation"])
    test_ids = set(pipeline.episode_split["test"])
    assert not train_ids & val_ids
    assert not train_ids & test_ids
    assert not val_ids & test_ids
    validate_sample(pipeline.validation_dataset[0], 5, 8, 14, 14)


def test_batch_validation():
    episodes = [make_episode(i) for i in range(3)]
    norm = NormalizationBundle.fit(episodes)
    ds = EpisodeAwareWindowDataset(episodes, norm, context_length=6,
                                   min_horizon=2, max_horizon=7)
    samples = [ds[i] for i in range(4)]
    batch = {key: np.stack([s[key] for s in samples]) for key in samples[0]}
    validate_batch(batch, 6, 7, 14, 14)
