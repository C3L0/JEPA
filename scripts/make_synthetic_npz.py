"""Create tiny synthetic M2-shaped NPZ files for a training smoke test."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_split(path: Path, n: int, seed: int, context_len: int = 10, horizon_max: int = 20) -> None:
    rng = np.random.default_rng(seed)
    state_dim = 14
    action_dim = 14
    states = rng.normal(size=(n, context_len, state_dim)).astype(np.float32)
    actions = np.zeros((n, horizon_max, action_dim), dtype=np.float32)
    masks = np.zeros((n, horizon_max), dtype=np.float32)
    horizons = rng.integers(5, horizon_max + 1, size=n, dtype=np.int32)
    targets = np.zeros((n, state_dim), dtype=np.float32)
    for i, h in enumerate(horizons):
        actions[i, :h] = rng.normal(scale=0.1, size=(h, action_dim)).astype(np.float32)
        masks[i, :h] = 1.0
        targets[i] = states[i, -1] + actions[i, :h].mean(axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state_window=states,
        action_sequence=actions,
        action_mask=masks,
        target_state=targets,
        horizon=horizons.astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/m4_smoke")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output_dir)
    make_split(output / "train_windows.npz", args.train_samples, seed=1)
    make_split(output / "val_windows.npz", args.val_samples, seed=2)
    print(f"Wrote synthetic windows to {output}")


if __name__ == "__main__":
    main()
