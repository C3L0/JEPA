from pathlib import Path

import equinox as eqx

from .config import LossConfig, ModelConfig
from .data import WindowArrays  # Check the actual loader name in your data.py
from .model import initialize_online_and_target
from .training import _evaluate

# Paths
CHECKPOINT = Path("runs/m4_smoke/best.eqx")
VAL_DATA = Path("data/m4_smoke/val_windows.npz")

# Configuration must match the checkpoint.
model_config = ModelConfig()
loss_config = LossConfig()

# Load validation data
validation_data = WindowArrays.from_npz(VAL_DATA)

# Recreate model architecture
key = __import__("jax").random.key(0)
online, target_encoder = initialize_online_and_target(model_config, key)

# Restore trained weights
online, target_encoder = eqx.tree_deserialise_leaves(
    CHECKPOINT,
    (online, target_encoder),
)

# Evaluate
metrics, diagnostics = _evaluate(
    online=online,
    target_encoder=target_encoder,
    dataset=validation_data,
    batch_size=32,
    loss_config=loss_config,
    seed=42,
)

print("\nEvaluation metrics:")
for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nDiagnostics:")
for name, value in diagnostics.items():
    print(f"{name}: {value}")
