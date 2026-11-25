from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def cosine_ema_momentum(step: jax.Array, total_steps: int, start: float = 0.99, end: float = 0.999) -> jax.Array:
    """Cosine schedule increasing EMA momentum from start to end."""
    progress = jnp.clip(jnp.asarray(step, dtype=jnp.float32) / max(total_steps, 1), 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return end - (end - start) * cosine


def ema_update(target: eqx.Module, online: eqx.Module, momentum: jax.Array) -> eqx.Module:
    """Update inexact-array leaves only; keep static/non-array leaves unchanged."""
    def update_leaf(target_leaf, online_leaf):
        if eqx.is_inexact_array(target_leaf):
            return momentum * target_leaf + (1.0 - momentum) * online_leaf
        return target_leaf

    return jax.tree_util.tree_map(update_leaf, target, online)
