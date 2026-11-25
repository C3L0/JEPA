from __future__ import annotations

import numpy as np


def representation_diagnostics(values: np.ndarray, std_threshold: float = 1e-2) -> dict[str, float | bool]:
    """Compute simple collapse indicators on a [N, D] representation matrix."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"Expected at least two representations with shape [N, D], got {x.shape}")
    std = np.std(x, axis=0)
    active = std > std_threshold

    centered = x - x.mean(axis=0, keepdims=True)
    covariance = (centered.T @ centered) / max(x.shape[0] - 1, 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), a_min=0.0, a_max=None)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / total
        probabilities = probabilities[probabilities > 0]
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))

    mean_std = float(std.mean())
    active_fraction = float(active.mean())
    collapsed = bool(mean_std < std_threshold or active_fraction < 0.10)
    return {
        "mean_feature_std": mean_std,
        "min_feature_std": float(std.min()),
        "active_dim_fraction": active_fraction,
        "effective_rank": effective_rank,
        "collapsed_warning": collapsed,
    }
