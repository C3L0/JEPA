from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def copy_module(module):
    """Create an independent tree copy, including a distinct set of array leaves."""
    return jax.tree_util.tree_map(
        lambda x: jnp.array(x) if eqx.is_array(x) else x,
        module,
    )


def ema_update(target, online, momentum: float):
    """Update target parameters: target <- m*target + (1-m)*online."""
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    return jax.tree_util.tree_map(
        lambda t, o: momentum * t + (1.0 - momentum) * o
        if eqx.is_array(t) else t,
        target,
        online,
    )
