from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int = 14
    action_dim: int = 14
    latent_dim: int = 256
    projector_dim: int = 512
    hidden_dim: int = 512
    mlp_depth: int = 2
    max_horizon: int = 20


@dataclass(frozen=True)
class LossConfig:
    prediction: str = "smooth_l1"  # smooth_l1 | cosine
    prediction_weight: float = 1.0
    vicreg_weight: float = 1.0
    vicreg_sim_coeff: float = 1.0
    vicreg_var_coeff: float = 10.0
    vicreg_cov_coeff: float = 0.04
    vicreg_gamma: float = 1.0
    eps: float = 1e-4


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    ema_start: float = 0.99
    ema_end: float = 0.999
    seed: int = 42
    log_every_steps: int = 50
    checkpoint_every_epochs: int = 5


def save_config(
    path: str | Path, model: ModelConfig, loss: LossConfig, train: TrainConfig
) -> None:
    payload = {
        "model": asdict(model),
        "loss": asdict(loss),
        "train": asdict(train),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
