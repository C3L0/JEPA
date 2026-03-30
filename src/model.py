import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom


class MLP(eqx.Module):
    """A standard multi-layer perceptron used as a building block."""

    layers: list

    def __init__(self, in_size, out_size, width_size, depth, key):
        keys = jrandom.split(key, depth + 1)
        layers = []

        # if no hidden layers
        if depth == 0:
            layers.append(eqx.nn.Linear(in_size, out_size, key=keys[0]))
        else:
            layers.append(eqx.nn.Linear(in_size, width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i + 1]))
            layers.append(eqx.nn.Linear(width_size, out_size, key=keys[-1]))
        self.layers = layers

    def __call__(self, x):
        for _, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = jax.nn.gelu(x)
        x = self.layers[-1](x)
        return x


class Encoder(eqx.Module):
    """Encodes a window of states into a latent vector."""

    mlp: MLP

    def __init__(self, state_dim, window_size, latent_dim, key):
        self.mlp = MLP(
            in_size=state_dim * window_size,
            out_size=latent_dim,
            width_size=256,
            depth=3,
            key=key,
        )

    def __call__(self, x):
        # x shape: (window_size, state_dim)
        x_flat = x.reshape(-1)
        return self.mlp(x_flat)


class Predictor(eqx.Module):
    """Predicts the next latent state given a current latent and an action."""

    mlp: MLP

    def __init__(self, latent_dim, action_dim, key):
        self.mlp = MLP(
            in_size=latent_dim + action_dim,
            out_size=latent_dim,
            width_size=256,
            depth=3,
            key=key,
        )

    def __call__(self, z, a):
        # z shape: (latent_dim,), a shape: (action_dim,)
        combined = jnp.concatenate([z, a], axis=-1)
        return self.mlp(combined)


class Projector(eqx.Module):
    """Projects latents into a higher-dimensional space for VICReg loss."""

    mlp: MLP

    def __init__(self, latent_dim, proj_dim, key):
        self.mlp = MLP(
            in_size=latent_dim, out_size=proj_dim, width_size=512, depth=2, key=key
        )

    def __call__(self, x):
        return self.mlp(x)


class AJEPA(eqx.Module):
    """The full A-JEPA model."""

    encoder: Encoder
    target_encoder: Encoder
    predictor: Predictor
    projector: Projector

    state_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    proj_dim: int = eqx.field(static=True)

    def __init__(self, state_dim, action_dim, window_size, latent_dim, proj_dim, key):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.proj_dim = proj_dim

        k1, k2, k3, k4 = jrandom.split(key, 4)

        self.encoder = Encoder(state_dim, window_size, latent_dim, k1)
        self.target_encoder = Encoder(state_dim, window_size, latent_dim, k2)
        self.predictor = Predictor(latent_dim, action_dim, k3)
        self.projector = Projector(latent_dim, proj_dim, k4)

    def __call__(self, s_ctx, a_cond):
        """Forward pass: context states + action -> predicted latent."""
        z_ctx = self.encoder(s_ctx)
        z_pred = self.predictor(z_ctx, a_cond)
        return z_pred

    def encode_target(self, s_target):
        """Encode the target window using the target encoder."""
        return self.target_encoder(s_target)

    def project(self, z):
        """Project a latent vector into the projection space."""
        return self.projector(z)
