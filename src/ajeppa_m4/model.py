from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .config import ModelConfig


class ProprioAJEPA(eqx.Module):
    """Small proprioception-only A-JEPA baseline.

    The context encoder embeds each state in the context window and pools the
    temporal embeddings. The action encoder consumes the full padded action
    sequence, its validity mask, and the normalized horizon.

    This baseline intentionally does not use DINOv2. It lets us validate the
    self-supervised training objective with the state/action fields currently
    confirmed in the dataset.
    """

    encoder: eqx.nn.MLP
    context_pool: eqx.nn.MLP
    action_encoder: eqx.nn.MLP
    predictor: eqx.nn.MLP
    projector: eqx.nn.MLP
    config: ModelConfig = eqx.field(static=True)

    @classmethod
    def create(cls, config: ModelConfig, key: jax.Array) -> "ProprioAJEPA":
        keys = jax.random.split(key, 5)
        encoder = eqx.nn.MLP(
            in_size=config.state_dim,
            out_size=config.latent_dim,
            width_size=config.hidden_dim,
            depth=config.mlp_depth,
            activation=jax.nn.gelu,
            key=keys[0],
        )
        context_pool = eqx.nn.MLP(
            in_size=2 * config.latent_dim,
            out_size=config.latent_dim,
            width_size=config.hidden_dim,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[1],
        )
        action_input_dim = (
            config.max_horizon * config.action_dim
            + config.max_horizon
            + 1
        )
        action_encoder = eqx.nn.MLP(
            in_size=action_input_dim,
            out_size=config.latent_dim,
            width_size=config.hidden_dim,
            depth=config.mlp_depth,
            activation=jax.nn.gelu,
            key=keys[2],
        )
        predictor = eqx.nn.MLP(
            in_size=2 * config.latent_dim,
            out_size=config.latent_dim,
            width_size=config.hidden_dim,
            depth=config.mlp_depth,
            activation=jax.nn.gelu,
            key=keys[3],
        )
        projector = eqx.nn.MLP(
            in_size=config.latent_dim,
            out_size=config.projector_dim,
            width_size=config.hidden_dim,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[4],
        )
        return cls(
            encoder=encoder,
            context_pool=context_pool,
            action_encoder=action_encoder,
            predictor=predictor,
            projector=projector,
            config=config,
        )

    def encode_context(self, state_window: jax.Array) -> jax.Array:
        """Encode one [context_length, state_dim] context window."""
        per_frame = jax.vmap(self.encoder)(state_window)
        mean_token = jnp.mean(per_frame, axis=0)
        last_token = per_frame[-1]
        return self.context_pool(jnp.concatenate([mean_token, last_token], axis=-1))

    def encode_actions(
        self,
        action_sequence: jax.Array,
        action_mask: jax.Array,
        horizon: jax.Array,
    ) -> jax.Array:
        """Encode one [max_horizon, action_dim] padded action sequence."""
        mask = action_mask.astype(action_sequence.dtype)
        masked_actions = action_sequence * mask[:, None]
        normalized_horizon = jnp.asarray(horizon, dtype=action_sequence.dtype) / float(
            self.config.max_horizon
        )
        features = jnp.concatenate(
            [masked_actions.reshape(-1), mask, normalized_horizon.reshape(1)],
            axis=0,
        )
        return self.action_encoder(features)

    def predict(
        self,
        context_latent: jax.Array,
        action_sequence: jax.Array,
        action_mask: jax.Array,
        horizon: jax.Array,
    ) -> jax.Array:
        action_latent = self.encode_actions(action_sequence, action_mask, horizon)
        return self.predictor(jnp.concatenate([context_latent, action_latent], axis=-1))

    def project(self, latent: jax.Array) -> jax.Array:
        return self.projector(latent)


def initialize_online_and_target(
    config: ModelConfig, key: jax.Array
) -> tuple[ProprioAJEPA, eqx.nn.MLP]:
    """Initialize online model and an exact target-encoder copy."""
    online = ProprioAJEPA.create(config, key)
    # JAX arrays are immutable; tree_map creates a distinct pytree with equal values.
    target_encoder = jax.tree_util.tree_map(
        lambda x: jnp.array(x) if eqx.is_array(x) else x,
        online.encoder,
    )
    return online, target_encoder
