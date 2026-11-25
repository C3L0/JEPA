import numpy as np

from src.ajeppa_m4.diagnostics import representation_diagnostics


def test_diagnostics_detect_constant_representation():
    result = representation_diagnostics(np.ones((16, 8), dtype=np.float32))
    assert result["collapsed_warning"] is True
    assert result["active_dim_fraction"] == 0.0


def test_diagnostics_accept_varying_representation():
    values = np.random.default_rng(3).normal(size=(64, 16)).astype(np.float32)
    result = representation_diagnostics(values)
    assert result["collapsed_warning"] is False
    assert result["effective_rank"] > 0.0
