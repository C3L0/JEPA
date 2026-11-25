from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.lerobot import load_lerobot_episodes
from data.pipeline import build_data_pipeline
from data.validation import validate_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LeRobot data for A-JEPA M2.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m2"
    )
    parser.add_argument("--context-length", type=int, default=10)
    parser.add_argument("--min-horizon", type=int, default=5)
    parser.add_argument("--max-horizon", type=int, default=20)
    parser.add_argument("--eval-horizon", type=int, default=10)
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument(
        "--normalization", choices=["standard", "minmax", "none"], default="standard"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def save_dataset(dataset, output_path: Path):
    """Export an EpisodeAwareWindowDataset to a compressed NPZ file."""
    samples = [dataset[i] for i in range(len(dataset))]

    arrays = {
        key: np.stack([np.asarray(sample[key]) for sample in samples])
        for key in samples[0]
    }

    np.savez_compressed(output_path, **arrays)
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_lerobot_episodes(
        args.dataset_root,
        expected_state_dim=args.state_dim,
        expected_action_dim=args.action_dim,
    )
    pipeline = build_data_pipeline(
        episodes,
        context_length=args.context_length,
        min_horizon=args.min_horizon,
        max_horizon=args.max_horizon,
        eval_horizon=args.eval_horizon,
        normalization=args.normalization,
        seed=args.seed,
    )

    # pipeline = build_data_pipeline(episodes)

    print("Episode split:")
    print("Train:", len(pipeline.episode_split["train"]))
    print("Validation:", len(pipeline.episode_split["validation"]))
    print("Test:", len(pipeline.episode_split["test"]))

    print("\nDataset sizes:")
    print("Train:", len(pipeline.train_dataset))
    print("Validation:", len(pipeline.validation_dataset))
    print("Test:", len(pipeline.test_dataset))

    print("\nNormalization:")
    print("State center:", pipeline.normalizers.state.center)
    print("State scale:", pipeline.normalizers.state.scale)
    print("Action center:", pipeline.normalizers.action.center)
    print("Action scale:", pipeline.normalizers.action.scale)

    val_horizons = np.array(
        [
            pipeline.validation_dataset[i]["horizon"]
            for i in range(len(pipeline.validation_dataset))
        ]
    )

    print("\nValidation horizons:")
    print("Unique:", np.unique(val_horizons, return_counts=True))

    first = pipeline.train_dataset[0]
    validate_sample(
        first,
        context_length=args.context_length,
        max_horizon=args.max_horizon,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )
    pipeline.normalizers.save(args.output_dir / "normalization.json")

    # Export datasets for M4 training
    dataset_output_dir = PROJECT_ROOT / "data" / "m4"

    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    save_dataset(
        pipeline.train_dataset,
        dataset_output_dir / "train_windows.npz",
    )

    save_dataset(
        pipeline.validation_dataset,
        dataset_output_dir / "val_windows.npz",
    )

    save_dataset(
        pipeline.test_dataset,
        dataset_output_dir / "test_windows.npz",
    )

    metadata = {
        "num_episodes": len(episodes),
        "episode_lengths": {str(ep.episode_id): len(ep.states) for ep in episodes},
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "context_length": args.context_length,
        "min_horizon": args.min_horizon,
        "max_horizon": args.max_horizon,
        "eval_horizon": args.eval_horizon,
        "normalization": args.normalization,
        "episode_split": pipeline.episode_split,
        "num_train_windows": len(pipeline.train_dataset),
        "num_validation_windows": len(pipeline.validation_dataset),
        "num_test_windows": len(pipeline.test_dataset),
        "sample_shapes": {k: list(np.asarray(v).shape) for k, v in first.items()},
    }
    (args.output_dir / "pipeline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))

    try:
        import torch
        from torch.utils.data import DataLoader

        from data.validation import validate_batch
        from data.windows import seed_worker

        loader = DataLoader(
            pipeline.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            drop_last=False,
        )
        batch = next(iter(loader))
        validate_batch(
            batch,
            context_length=args.context_length,
            max_horizon=args.max_horizon,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
        )
        print("Batch validation: OK")
    except ImportError:
        print("PyTorch not installed; skipped batched collation check.")


if __name__ == "__main__":
    main()
