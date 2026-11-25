import pytest

eqx = pytest.importorskip("equinox")
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from src.ajeppa_m4.ema import cosine_ema_momentum, ema_update


def test_cosine_schedule_starts_and_ends_at_requested_values():
    assert float(cosine_ema_momentum(0, 100, 0.99, 0.999)) == pytest.approx(
        0.99, abs=1e-6
    )
    assert float(cosine_ema_momentum(100, 100, 0.99, 0.999)) == pytest.approx(
        0.999, abs=1e-6
    )


def test_ema_update_moves_target_towards_online():
    target = jnp.array([0.0, 0.0])
    online = jnp.array([1.0, 2.0])
    updated = ema_update(target, online, 0.9)
    assert jnp.allclose(updated, jnp.array([0.1, 0.2]))
