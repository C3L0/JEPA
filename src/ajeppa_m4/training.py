from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import LossConfig, ModelConfig, TrainConfig, save_config
from .data import WindowArrays, iter_batches
from .diagnostics import representation_diagnostics
from .ema import cosine_ema_momentum, ema_update
from .losses import total_loss
from .model import ProprioAJEPA, initialize_online_and_target


def _forward_and_loss(
    online: ProprioAJEPA,
    target_encoder: eqx.nn.MLP,
    batch: dict[str, jax.Array],
    loss_config: LossConfig,
) -> tuple[jax.Array, dict[str, jax.Array], jax.Array, jax.Array]:
    state_window = batch["state_window"]
    action_sequence = batch["action_sequence"]
    action_mask = batch["action_mask"]
    horizon = batch["horizon"]
    target_state = batch["target_state"]

    context_latent = jax.vmap(online.encode_context)(state_window)
    prediction = jax.vmap(online.predict)(
        context_latent, action_sequence, action_mask, horizon
    )

    target_latent = jax.vmap(target_encoder)(target_state)
    target_latent = jax.lax.stop_gradient(target_latent)

    prediction_projection = jax.vmap(online.project)(prediction)
    target_projection = jax.vmap(online.project)(target_latent)

    loss, metrics = total_loss(
        prediction,
        target_latent,
        prediction_projection,
        target_projection,
        loss_config,
    )
    return loss, metrics, prediction, target_latent


def make_train_step(
    optimizer: optax.GradientTransformation,
    loss_config: LossConfig,
    train_config: TrainConfig,
    total_steps: int,
):
    """Build a JIT-compiled update; optimizer only sees the online model."""

    @eqx.filter_jit
    def train_step(online, target_encoder, optimizer_state, batch, step):
        def objective(online_model):
            loss, metrics, _, _ = _forward_and_loss(
                online_model, target_encoder, batch, loss_config
            )
            return loss, metrics

        (loss, metrics), gradients = eqx.filter_value_and_grad(objective, has_aux=True)(
            online
        )
        trainable_params = eqx.filter(online, eqx.is_inexact_array)

        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            trainable_params,
        )

        online = eqx.apply_updates(online, updates)

        momentum = cosine_ema_momentum(
            step,
            total_steps,
            start=train_config.ema_start,
            end=train_config.ema_end,
        )
        target_encoder = ema_update(target_encoder, online.encoder, momentum)
        metrics = {**metrics, "total_loss": loss, "ema_momentum": momentum}
        return online, target_encoder, optimizer_state, metrics

    return train_step


def make_eval_step(loss_config: LossConfig):
    @eqx.filter_jit
    def eval_step(online, target_encoder, batch):
        loss, metrics, prediction, target = _forward_and_loss(
            online, target_encoder, batch, loss_config
        )
        return {**metrics, "total_loss": loss}, prediction, target

    return eval_step


def _mean_metrics(metric_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = metric_rows[0].keys()
    return {
        key: float(np.mean([float(np.asarray(row[key])) for row in metric_rows]))
        for key in keys
    }


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(row.keys())

    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_fieldnames = next(csv.reader(handle))

        if existing_fieldnames != fieldnames:
            raise ValueError(
                "CSV schema mismatch. Use a fresh output directory for this experiment."
            )

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)

        if path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)


def _save_checkpoint(
    path: Path, online: ProprioAJEPA, target_encoder: eqx.nn.MLP
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, (online, target_encoder))


def _evaluate(
    online: ProprioAJEPA,
    target_encoder: eqx.nn.MLP,
    dataset: WindowArrays,
    batch_size: int,
    loss_config: LossConfig,
    seed: int,
) -> tuple[dict[str, float], dict[str, float | bool]]:
    eval_step = make_eval_step(loss_config)

    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    original_errors = []
    shuffled_errors = []
    prediction_deltas = []

    mean_action_errors = []
    mean_action_prediction_deltas = []

    latent_cosine_similarities = []
    latent_distances = []

    # 1. Shuffle actions globally, matching samples with the same horizon.
    rng = np.random.default_rng(seed)

    all_actions = dataset.action_sequence
    all_horizons = dataset.horizon.reshape(-1)

    globally_shuffled_actions = all_actions.copy()

    for h in np.unique(all_horizons):
        indices = np.flatnonzero(all_horizons == h)

        if len(indices) < 2:
            continue

        permuted_indices = rng.permutation(indices)
        donor_indices = np.roll(permuted_indices, shift=1)

        globally_shuffled_actions[permuted_indices] = all_actions[donor_indices]

    # 2. Initialize diagnostics collections ONCE.
    cursor = 0
    prediction_projections = []
    target_projections = []

    persistence_errors = []
    jepa_errors = []

    # 3. Evaluate the dataset ONCE.
    for batch in iter_batches(
        dataset, batch_size, shuffle=False, seed=seed, drop_last=True
    ):
        # Original prediction.
        jax_batch = {key: jnp.asarray(value) for key, value in batch.items()}

        metrics, prediction, target = eval_step(online, target_encoder, jax_batch)

        rows.append(jax.device_get(metrics))

        prediction_np = np.asarray(jax.device_get(prediction))
        target_np = np.asarray(jax.device_get(target))

        predictions.append(prediction_np)
        targets.append(target_np)

        # Latent persistence baseline
        last_state = jax_batch["state_window"][:, -1, :]
        persistence_prediction = jax.vmap(target_encoder)(last_state)

        # Compare current latent representation with future latent representation.
        current_latent_np = np.asarray(jax.device_get(persistence_prediction))

        dot_product = np.sum(current_latent_np * target_np, axis=-1)

        norm_product = np.linalg.norm(current_latent_np, axis=-1) * np.linalg.norm(
            target_np, axis=-1
        )

        cosine_similarity = dot_product / (norm_product + 1e-8)

        latent_cosine_similarities.append(cosine_similarity)

        latent_distances.append(np.linalg.norm(current_latent_np - target_np, axis=-1))

        persistence_error = jnp.mean(
            (persistence_prediction - target) ** 2,
            axis=-1,
        )

        jepa_error = jnp.mean(
            (prediction - target) ** 2,
            axis=-1,
        )

        persistence_errors.append(np.asarray(jax.device_get(persistence_error)))
        jepa_errors.append(np.asarray(jax.device_get(jepa_error)))

        # Projected representations.
        prediction_projection = jax.vmap(online.project)(prediction)
        target_projection = jax.vmap(online.project)(target)

        prediction_projections.append(np.asarray(jax.device_get(prediction_projection)))
        target_projections.append(np.asarray(jax.device_get(target_projection)))

        # Prediction using globally shuffled actions.
        batch_size_actual = len(batch["horizon"])

        shuffled_batch = dict(batch)
        shuffled_batch["action_sequence"] = globally_shuffled_actions[
            cursor : cursor + batch_size_actual
        ]

        cursor += batch_size_actual

        jax_shuffled_batch = {
            key: jnp.asarray(value) for key, value in shuffled_batch.items()
        }

        _, shuffled_prediction, _ = eval_step(
            online, target_encoder, jax_shuffled_batch
        )

        shuffled_prediction_np = np.asarray(jax.device_get(shuffled_prediction))

        # Compare predictions against the same targets.
        original_errors.append(np.mean((prediction_np - target_np) ** 2, axis=-1))

        # Prediction with mean actions (zero in normalized action space).
        mean_action_batch = dict(jax_batch)
        mean_action_batch["action_sequence"] = jnp.zeros_like(
            jax_batch["action_sequence"]
        )

        _, mean_action_prediction, _ = eval_step(
            online, target_encoder, mean_action_batch
        )

        mean_action_prediction_np = np.asarray(jax.device_get(mean_action_prediction))

        mean_action_errors.append(
            np.mean((mean_action_prediction_np - target_np) ** 2, axis=-1)
        )

        mean_action_prediction_deltas.append(
            np.linalg.norm(prediction_np - mean_action_prediction_np, axis=-1)
        )

        shuffled_errors.append(
            np.mean((shuffled_prediction_np - target_np) ** 2, axis=-1)
        )

        prediction_deltas.append(
            np.linalg.norm(prediction_np - shuffled_prediction_np, axis=-1)
        )

    if not rows:
        raise ValueError(
            "Validation split has fewer than one full batch. "
            "Reduce --batch-size or provide more validation windows."
        )

    # 4. Latent representation diagnostics.
    pred_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)

    diagnostics = {
        **{
            f"prediction_{k}": v
            for k, v in representation_diagnostics(pred_array).items()
        },
        **{
            f"target_{k}": v
            for k, v in representation_diagnostics(target_array).items()
        },
    }

    # 5. Projected representation diagnostics.
    prediction_projection_array = np.concatenate(prediction_projections, axis=0)
    target_projection_array = np.concatenate(target_projections, axis=0)

    diagnostics.update(
        {
            f"prediction_projection_{k}": v
            for k, v in representation_diagnostics(prediction_projection_array).items()
        }
    )

    diagnostics.update(
        {
            f"target_projection_{k}": v
            for k, v in representation_diagnostics(target_projection_array).items()
        }
    )

    # 6. Aggregate action-shuffling diagnostics.
    if original_errors:
        original_mse = np.concatenate(original_errors)
        shuffled_mse = np.concatenate(shuffled_errors)
        prediction_delta = np.concatenate(prediction_deltas)

        mean_action_mse = np.concatenate(mean_action_errors)
        mean_action_delta = np.concatenate(mean_action_prediction_deltas)

        cosine_values = np.concatenate(latent_cosine_similarities)
        latent_distance_values = np.concatenate(latent_distances)

        persistence_mse = float(np.mean(np.concatenate(persistence_errors)))
        jepa_mse = float(np.mean(np.concatenate(jepa_errors)))

        diagnostics.update(
            {
                "action_original_mse": float(np.mean(original_mse)),
                "action_shuffled_mse": float(np.mean(shuffled_mse)),
                "action_shuffle_mse_ratio": float(
                    np.mean(shuffled_mse) / (np.mean(original_mse) + 1e-8)
                ),
                "action_prediction_delta": float(np.mean(prediction_delta)),
                "action_test_samples": int(len(original_mse)),
                "val_latent_persistence_mse": persistence_mse,
                "val_latent_jepa_mse": jepa_mse,
                "val_latent_jepa_vs_persistence_ratio": (
                    jepa_mse / (persistence_mse + 1e-8)
                ),
                "action_mean_mse": float(np.mean(mean_action_mse)),
                "action_mean_prediction_delta": float(np.mean(mean_action_delta)),
                "latent_current_future_cosine": float(np.mean(cosine_values)),
                "latent_current_future_distance": float(
                    np.mean(latent_distance_values)
                ),
            }
        )

    return _mean_metrics(rows), diagnostics


def train(
    train_data: WindowArrays,
    validation_data: WindowArrays,
    output_dir: str | Path,
    model_config: ModelConfig,
    loss_config: LossConfig,
    train_config: TrainConfig,
) -> dict[str, Any]:
    """Train the proprioceptive baseline and save logs/checkpoints."""
    if train_config.batch_size < 2:
        raise ValueError("batch_size must be >= 2 for VICReg.")
    if len(train_data) < train_config.batch_size:
        raise ValueError("Training split must contain at least one full batch.")

    for split_name, data in (("train", train_data), ("validation", validation_data)):
        data.validate()
        if data.state_window.shape[-1] != model_config.state_dim:
            raise ValueError(f"{split_name}: state_dim does not match model config.")
        if data.action_sequence.shape[-1] != model_config.action_dim:
            raise ValueError(f"{split_name}: action_dim does not match model config.")
        if data.action_sequence.shape[1] != model_config.max_horizon:
            raise ValueError(
                f"{split_name}: padded max_horizon does not match model config."
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "config.json", model_config, loss_config, train_config)

    key = jax.random.key(train_config.seed)
    online, target_encoder = initialize_online_and_target(model_config, key)

    optimizer = optax.chain(
        optax.clip_by_global_norm(train_config.grad_clip_norm),
        optax.adamw(
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        ),
    )
    trainable_params = eqx.filter(online, eqx.is_inexact_array)
    optimizer_state = optimizer.init(trainable_params)

    steps_per_epoch = len(train_data) // train_config.batch_size
    total_steps = max(steps_per_epoch * train_config.epochs, 1)
    train_step = make_train_step(optimizer, loss_config, train_config, total_steps)

    best_validation_prediction = float("inf")
    global_step = 0
    history_path = output_dir / "history.csv"

    for epoch in range(1, train_config.epochs + 1):
        train_rows: list[dict[str, Any]] = []
        for batch in iter_batches(
            train_data,
            train_config.batch_size,
            shuffle=True,
            seed=train_config.seed + epoch,
            drop_last=False,
        ):
            jax_batch = {key: jnp.asarray(value) for key, value in batch.items()}
            online, target_encoder, optimizer_state, metrics = train_step(
                online,
                target_encoder,
                optimizer_state,
                jax_batch,
                jnp.asarray(global_step, dtype=jnp.int32),
            )
            train_rows.append(jax.device_get(metrics))
            global_step += 1

        train_metrics = _mean_metrics(train_rows)
        val_metrics, diagnostics = _evaluate(
            online,
            target_encoder,
            validation_data,
            train_config.batch_size,
            loss_config,
            train_config.seed,
        )

        row: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **diagnostics,
        }
        _append_csv(history_path, row)

        if val_metrics["prediction_loss"] < best_validation_prediction:
            best_validation_prediction = val_metrics["prediction_loss"]
            _save_checkpoint(output_dir / "best.eqx", online, target_encoder)

        if (
            epoch % train_config.checkpoint_every_epochs == 0
            or epoch == train_config.epochs
        ):
            _save_checkpoint(output_dir / "last.eqx", online, target_encoder)

        warning = (
            " COLLAPSE WARNING" if diagnostics["prediction_collapsed_warning"] else ""
        )
        print(
            f"epoch {epoch:03d}/{train_config.epochs} "
            f"train_loss={train_metrics['total_loss']:.5f} "
            f"val_pred={val_metrics['prediction_loss']:.5f} "
            f"active_dims={diagnostics['prediction_active_dim_fraction']:.3f}"
            f"{warning}"
        )

    return {
        "online": online,
        "target_encoder": target_encoder,
        "best_validation_prediction_loss": best_validation_prediction,
        "history_path": str(history_path),
        "output_dir": str(output_dir),
    }
