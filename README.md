# A-JEPA M2 — Data Engineering

Place `src/data/` into the existing project `src/data/` package (or merge its files with an existing package). The loader reads LeRobot parquet shards containing `observation.state`, `action`, `episode_index`, and `frame_index`.

Run from the project root:

```bash
pip install -r requirements-data.txt
python scripts/prepare_data.py --dataset-root /path/to/aloha_static_coffee_new
```

Default sample configuration: 14-D states/actions, context length 10, random horizon 5–20 frames, 70/15/15 episode split. Normalization statistics are fitted on training episodes only and saved to `artifacts/m2/normalization.json`.

Each sample contains:

- `state_window`: `[context_length, state_dim]`
- `action_sequence`: `[max_horizon, action_dim]` (zero-padded after the sampled horizon)
- `action_mask`: `[max_horizon]` (True for valid action tokens)
- `target_state`: `[state_dim]`
- `horizon`: scalar prediction horizon in frames

Alignment convention: `action[t]` drives the transition from `state[t]` to `state[t+1]`. Check the dataset's action semantics before using this convention for control.

The pipeline splits by episode, never by frame, and uses only training episodes to fit normalization statistics. The initial implementation is state/action-only; camera video decoding is intentionally separate.
