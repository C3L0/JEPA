import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from src.ajeppa_m4.losses import prediction_loss, vicreg_loss


def test_prediction_loss_is_zero_for_identical_vectors():
    x = jnp.ones((4, 8))
    assert float(prediction_loss(x, x, "smooth_l1")) == pytest.approx(0.0)


def test_vicreg_returns_finite_components():
    x = jnp.arange(40, dtype=jnp.float32).reshape(5, 8) / 10.0
    y = x + 0.1
    loss, metrics = vicreg_loss(x, y)
    assert jnp.isfinite(loss)
    assert set(metrics) == {"vicreg_invariance", "vicreg_variance", "vicreg_covariance"}


def test_vicreg_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        vicreg_loss(jnp.ones((4, 8)), jnp.ones((4, 9)))
