from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class MLP(eqx.Module):
    layers: tuple
    activation: object = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, *, key):
        dims = (in_dim, *hidden_dims, out_dim)
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = tuple(
            eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i])
            for i in range(len(dims) - 1)
        )
        self.activation = jax.nn.gelu

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    qkv: eqx.nn.Linear
    attention_out: eqx.nn.Linear
    norm2: eqx.nn.LayerNorm
    mlp: MLP
    num_heads: int = eqx.field(static=True)
    model_dim: int = eqx.field(static=True)

    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: int = 4, *, key):
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        k1, k2, k3 = jax.random.split(key, 3)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.norm1 = eqx.nn.LayerNorm(model_dim)
        self.qkv = eqx.nn.Linear(model_dim, 3 * model_dim, key=k1)
        self.attention_out = eqx.nn.Linear(model_dim, model_dim, key=k2)
        self.norm2 = eqx.nn.LayerNorm(model_dim)
        self.mlp = MLP(model_dim, (mlp_ratio * model_dim,), model_dim, key=k3)

    def __call__(self, x, key_mask=None):
        # x: [sequence_length, model_dim]; key_mask: [sequence_length], True=valid
        seq_len = x.shape[0]
        h = jax.vmap(self.norm1)(x)
        qkv = jax.vmap(self.qkv)(h).reshape(seq_len, 3, self.num_heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        logits = jnp.einsum("ihd,jhd->hij", q, k) / (q.shape[-1] ** 0.5)
        if key_mask is not None:
            logits = jnp.where(key_mask[None, None, :], logits, -1e30)
        weights = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("hij,jhd->ihd", weights, v).reshape(seq_len, self.model_dim)
        x = x + jax.vmap(self.attention_out)(attended)
        h = jax.vmap(self.norm2)(x)
        return x + jax.vmap(self.mlp)(h)
