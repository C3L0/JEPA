from __future__ import annotations

import argparse

from .config import LossConfig, ModelConfig, TrainConfig
from .data import WindowArrays
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the M4 proprioceptive A-JEPA baseline")
    parser.add_argument("--train-npz", required=True, help="NPZ containing normalized M2 training windows")
    parser.add_argument("--val-npz", required=True, help="NPZ containing normalized M2 validation windows")
    parser.add_argument("--output-dir", default="runs/m4_baseline")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--projector-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prediction-loss", choices=["smooth_l1", "cosine"], default="smooth_l1")
    parser.add_argument("--disable-vicreg", action="store_true", help="Run a prediction-only baseline ablation")
    args = parser.parse_args()

    train_data = WindowArrays.from_npz(args.train_npz)
    val_data = WindowArrays.from_npz(args.val_npz)

    model_config = ModelConfig(
        state_dim=int(train_data.state_window.shape[-1]),
        action_dim=int(train_data.action_sequence.shape[-1]),
        latent_dim=args.latent_dim,
        projector_dim=args.projector_dim,
        hidden_dim=args.hidden_dim,
        max_horizon=int(train_data.action_sequence.shape[1]),
    )
    loss_config = LossConfig(
        prediction=args.prediction_loss,
        vicreg_weight=0.0 if args.disable_vicreg else 1.0,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    print("M4 dataset shapes:")
    for name, value in train_data.as_dict().items():
        print(f"  train/{name}: {value.shape}")
    print(f"  validation samples: {len(val_data)}")
    print(f"  VICReg: {'disabled' if args.disable_vicreg else 'enabled'}")

    train(
        train_data=train_data,
        validation_data=val_data,
        output_dir=args.output_dir,
        model_config=model_config,
        loss_config=loss_config,
        train_config=train_config,
    )
