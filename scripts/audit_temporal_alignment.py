from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.lerobot import load_lerobot_episodes


def summarize(name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    print(f"\n{name}")
    print(f"  shape: {values.shape}")
    print(f"  mean:  {values.mean():.6f}")
    print(f"  std:   {values.std():.6f}")
    print(f"  min:   {values.min():.6f}")
    print(f"  max:   {values.max():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
    )
    args = parser.parse_args()

    episodes = load_lerobot_episodes(args.dataset_root)

    print(f"Episodes: {len(episodes)}")
    print(f"Total frames: {sum(len(ep.states) for ep in episodes)}")

    all_state_deltas = {h: [] for h in args.horizons}
    all_action_values = []
    all_action_deltas = []

    for ep in episodes:
        states = ep.states
        actions = ep.actions
        frames = ep.frame_indices

        assert len(states) == len(actions) == len(frames)
        assert np.all(np.diff(frames) == 1)

        if ep.timestamps is not None:
            dt = np.diff(ep.timestamps)
            print(
                f"Episode {ep.episode_id}: "
                f"median dt={np.median(dt):.6f}s, "
                f"min dt={dt.min():.6f}s, "
                f"max dt={dt.max():.6f}s"
            )

        all_action_values.append(actions)

        if len(actions) > 1:
            all_action_deltas.append(np.diff(actions, axis=0))

        for horizon in args.horizons:
            if horizon >= len(states):
                continue

            # State displacement over the specified number of transitions.
            delta = states[horizon:] - states[:-horizon]
            all_state_deltas[horizon].append(delta)

    summarize("Raw actions", np.concatenate(all_action_values, axis=0))
    summarize(
        "Consecutive action differences",
        np.concatenate(all_action_deltas, axis=0),
    )

    print("\nState displacement by horizon")
    for horizon, chunks in all_state_deltas.items():
        if not chunks:
            continue

        delta = np.concatenate(chunks, axis=0)

        print(f"\nHorizon = {horizon}")
        print(f"  Mean absolute displacement: {np.mean(np.abs(delta)):.6f}")
        print(f"  RMS displacement:            {np.sqrt(np.mean(delta**2)):.6f}")
        print(f"  Per-dimension std:            {np.std(delta, axis=0)}")


if __name__ == "__main__":
    main()
