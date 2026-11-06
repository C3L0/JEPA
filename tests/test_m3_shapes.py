# import sys
# from pathlib import Path

import jax
import jax.numpy as jnp

from src.models.aje_pa import AJEPA

# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT / "src"))


def test_aje_pa_forward_shapes():
    model = AJEPA(
        state_dim=14,
        action_dim=14,
        visual_dim=384,
        model_dim=384,
        proj_dim=512,
        context_depth=2,
        predictor_depth=2,
        num_heads=6,
        key=jax.random.PRNGKey(0),
    )
    state = jnp.zeros((14,))
    future_state = jnp.ones((14,))
    visual = jnp.zeros((16, 384))
    future_visual = jnp.ones((16, 384))
    actions = jnp.zeros((20, 14))
    mask = jnp.array([True] * 10 + [False] * 10)
    horizon = jnp.array(10)

    z = model.encode_context(state, visual)
    z_target = model.encode_target(future_state, future_visual)
    prediction = model.predict(z, actions, mask, horizon)
    projected = model.project(prediction)

    assert z.shape == (384,)
    assert z_target.shape == (384,)
    assert prediction.shape == (384,)
    assert projected.shape == (512,)


def test_target_encoder_starts_equal():
    model = AJEPA(key=jax.random.PRNGKey(1))
    x = jnp.zeros((14,))
    visual = jnp.zeros((4, 384))
    online = model.context_encoder(x, visual)
    target = model.target_encoder(x, visual)
    assert jnp.allclose(online, target)
