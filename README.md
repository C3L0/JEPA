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


# A-JEPA M3 — Architecture

This module implements the first multimodal A-JEPA architecture in JAX/Equinox.

## Important data requirement

The architecture expects DINOv2 patch tokens for each current and future image:
- `visual_tokens`: `[num_patches, 384]`
- `future_visual_tokens`: `[num_patches, 384]`

Your earlier M2 inspection showed no image feature keys. Before real training, confirm that the LeRobot dataset has video files and decode synchronized frames. The architecture tests use dummy visual tokens only.

## Install with uv

Core model:
```bash
uv add jax equinox optax
uv add --dev pytest
```

Frozen DINOv2 extractor (PyTorch/Hugging Face):
```bash
uv add torch transformers pillow
```

## Files

- `src/models/blocks.py`: MLP and Transformer block.
- `src/models/encoders.py`: state/action encoders, context Transformer, predictor, projector.
- `src/models/dinov2.py`: frozen DINOv2 feature extractor. Keeps patch tokens and drops CLS.
- `src/models/ema.py`: exact target copy and EMA update.
- `src/models/aje_pa.py`: assembled A-JEPA model.
- `tests/test_m3_shapes.py`: forward-shape and target-initialization tests.

## Minimal usage

```python
import jax
import jax.numpy as jnp
from src.models import AJEPA

model = AJEPA(key=jax.random.PRNGKey(0))

# One sample:
state = jnp.zeros((14,))
visual_tokens = jnp.zeros((256, 384))  # replace with DINOv2 output
actions = jnp.zeros((20, 14))
action_mask = jnp.array([True] * 10 + [False] * 10)
horizon = jnp.array(10)

z_context = model.encode_context(state, visual_tokens)
z_pred = model.predict(z_context, actions, action_mask, horizon)
```

Target encoder is initialized as an exact copy of context encoder. After an optimizer
step on online parameters, update target parameters with:

```python
model = model.update_target(momentum=0.996)
```

For target-side loss computation, use `jax.lax.stop_gradient(model.encode_target(...))`.

## Notes

- The model methods currently operate on one sample. Batch them with `jax.vmap`.
- `action_mask` has shape `[max_horizon]`, with True for valid action tokens.
- `horizon` is an integer in `[1, max_horizon]`.
- This is an architecture baseline, not yet a training loop or complete VICReg loss.
- DINOv2 is frozen and runs in PyTorch; its output tokens are passed to the JAX model.
- Ensure the installed PyTorch build matches your CPU/CUDA environment.
