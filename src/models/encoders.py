from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .blocks import MLP, TransformerBlock


class StateEncoder(eqx.Module):
    network: MLP

    def __init__(self, state_dim: int, model_dim: int, *, key):
        self.network = MLP(state_dim, (model_dim, model_dim), model_dim, key=key)

    def __call__(self, state):
        return self.network(state)


class ActionEncoder(eqx.Module):
    network: MLP

    def __init__(self, action_dim: int, model_dim: int, *, key):
        self.network = MLP(action_dim, (model_dim, model_dim), model_dim, key=key)

    def __call__(self, action_sequence):
        # action_sequence: [horizon, action_dim]
        return jax.vmap(self.network)(action_sequence)


class ContextEncoder(eqx.Module):
    state_encoder: StateEncoder
    visual_projection: eqx.nn.Linear
    blocks: tuple
    model_dim: int = eqx.field(static=True)

    def __init__(
        self, state_dim: int, visual_dim: int, model_dim: int,
        depth: int, num_heads: int, *, key
    ):
        keys = jax.random.split(key, depth + 2)
        self.model_dim = model_dim
        self.state_encoder = StateEncoder(state_dim, model_dim, key=keys[0])
        self.visual_projection = eqx.nn.Linear(visual_dim, model_dim, key=keys[1])
        self.blocks = tuple(
            TransformerBlock(model_dim, num_heads, key=keys[i + 2])
            for i in range(depth)
        )

    def __call__(self, state, visual_tokens):
        # state: [state_dim]; visual_tokens: [num_patches, visual_dim]
        state_token = self.state_encoder(state)[None, :]
        visual = jax.vmap(self.visual_projection)(visual_tokens)
        tokens = jnp.concatenate([state_token, visual], axis=0)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens[0]


class ActionConditionedPredictor(eqx.Module):
    blocks: tuple
    norm: eqx.nn.LayerNorm
    model_dim: int = eqx.field(static=True)

    def __init__(self, model_dim: int, depth: int, num_heads: int, *, key):
        keys = jax.random.split(key, depth)
        self.model_dim = model_dim
        self.blocks = tuple(
            TransformerBlock(model_dim, num_heads, key=keys[i])
            for i in range(depth)
        )
        self.norm = eqx.nn.LayerNorm(model_dim)

    def __call__(self, context_latent, action_tokens, action_mask, horizon):
        # context_latent: [D], action_tokens: [max_horizon, D]
        # horizon is the number of valid future actions (1..max_horizon).
        sequence = jnp.concatenate([context_latent[None, :], action_tokens], axis=0)
        key_mask = jnp.concatenate([jnp.array([True]), action_mask.astype(bool)], axis=0)
        for block in self.blocks:
            sequence = block(sequence, key_mask=key_mask)
        sequence = jax.vmap(self.norm)(sequence)
        # Index 0 is context; index `horizon` is the representation after the
        # final valid action token. This supports variable horizons with padding.
        return sequence[horizon]


class Projector(eqx.Module):
    network: MLP

    def __init__(self, model_dim: int, proj_dim: int, *, key):
        self.network = MLP(model_dim, (model_dim, model_dim), proj_dim, key=key)

    def __call__(self, latent):
        return self.network(latent)
