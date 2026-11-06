from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .ema import copy_module, ema_update
from .encoders import ContextEncoder, ActionEncoder, ActionConditionedPredictor, Projector


class AJEPA(eqx.Module):
    context_encoder: ContextEncoder
    target_encoder: ContextEncoder
    action_encoder: ActionEncoder
    predictor: ActionConditionedPredictor
    projector: Projector

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 14,
        visual_dim: int = 384,
        model_dim: int = 384,
        proj_dim: int = 512,
        context_depth: int = 2,
        predictor_depth: int = 4,
        num_heads: int = 6,
        *,
        key,
    ):
        keys = jax.random.split(key, 4)
        context = ContextEncoder(
            state_dim, visual_dim, model_dim, context_depth, num_heads, key=keys[0]
        )
        # Target starts as an exact parameter copy of the online context encoder.
        self.context_encoder = context
        self.target_encoder = copy_module(context)
        self.action_encoder = ActionEncoder(action_dim, model_dim, key=keys[1])
        self.predictor = ActionConditionedPredictor(
            model_dim, predictor_depth, num_heads, key=keys[2]
        )
        self.projector = Projector(model_dim, proj_dim, key=keys[3])

    def encode_context(self, state, visual_tokens):
        return self.context_encoder(state, visual_tokens)

    def encode_target(self, future_state, future_visual_tokens):
        # Caller should apply stop_gradient to this result in the loss function.
        return self.target_encoder(future_state, future_visual_tokens)

    def predict(self, context_latent, action_sequence, action_mask, horizon):
        action_tokens = self.action_encoder(action_sequence)
        return self.predictor(context_latent, action_tokens, action_mask, horizon)

    def project(self, latent):
        return self.projector(latent)

    def update_target(self, momentum: float = 0.996):
        updated = ema_update(self.target_encoder, self.context_encoder, momentum)
        return eqx.tree_at(lambda model: model.target_encoder, self, updated)
