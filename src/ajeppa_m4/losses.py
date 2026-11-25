from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import LossConfig


def prediction_loss(
    prediction: jax.Array, target: jax.Array, kind: str = "smooth_l1"
) -> jax.Array:
    target = jax.lax.stop_gradient(target)
    if kind == "smooth_l1":
        error = jnp.abs(prediction - target)
        return jnp.mean(jnp.where(error < 1.0, 0.5 * error**2, error - 0.5))
    if kind == "cosine":
        pred_norm = prediction / (
            jnp.linalg.norm(prediction, axis=-1, keepdims=True) + 1e-8
        )
        target_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
        return jnp.mean(1.0 - jnp.sum(pred_norm * target_norm, axis=-1))
    raise ValueError(f"Unsupported prediction loss: {kind}")


def _variance_loss(x: jax.Array, gamma: float, eps: float) -> jax.Array:
    # Batch dimension is axis 0. VICReg needs a batch size greater than one.
    std = jnp.sqrt(jnp.var(x, axis=0) + eps)
    return jnp.mean(jax.nn.relu(gamma - std))


def _covariance_loss(x: jax.Array) -> jax.Array:
    batch_size, feature_dim = x.shape
    centered = x - jnp.mean(x, axis=0, keepdims=True)
    covariance = (centered.T @ centered) / jnp.maximum(batch_size - 1, 1)
    off_diagonal = covariance - jnp.diag(jnp.diag(covariance))
    return jnp.sum(off_diagonal**2) / feature_dim


def vicreg_loss(
    prediction_projection: jax.Array,
    target_projection: jax.Array,
    *,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """VICReg loss for predicted and target projections [B, D]."""

    if prediction_projection.ndim != 2 or target_projection.ndim != 2:
        raise ValueError("VICReg inputs must have shape [batch, feature_dim].")

    if prediction_projection.shape != target_projection.shape:
        raise ValueError(
            "VICReg inputs must have identical shapes; got "
            f"{prediction_projection.shape} and {target_projection.shape}."
        )

    # Target branch is not trained through gradients.
    target_projection = jax.lax.stop_gradient(target_projection)

    # 1. Invariance: bring predictions closer to targets.
    invariance = jnp.mean((prediction_projection - target_projection) ** 2)

    # 2. Variance: prevent collapse of the online predictions.
    variance = _variance_loss(prediction_projection, gamma, eps)

    # 3. Covariance: reduce redundancy between prediction dimensions.
    covariance = _covariance_loss(prediction_projection)

    total = sim_coeff * invariance + var_coeff * variance + cov_coeff * covariance

    return total, {
        "vicreg_invariance": invariance,
        "vicreg_variance": variance,
        "vicreg_covariance": covariance,
    }


def total_loss(
    prediction: jax.Array,
    target: jax.Array,
    prediction_projection: jax.Array,
    target_projection: jax.Array,
    config: LossConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    pred = prediction_loss(prediction, target, config.prediction)
    vicreg, vicreg_metrics = vicreg_loss(
        prediction_projection,
        target_projection,
        sim_coeff=config.vicreg_sim_coeff,
        var_coeff=config.vicreg_var_coeff,
        cov_coeff=config.vicreg_cov_coeff,
        gamma=config.vicreg_gamma,
        eps=config.eps,
    )
    total = config.prediction_weight * pred + config.vicreg_weight * vicreg
    metrics = {"prediction_loss": pred, "vicreg_loss": vicreg, **vicreg_metrics}
    return total, metrics
