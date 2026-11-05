from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Normalizer:
    method: str
    center: np.ndarray
    scale: np.ndarray
    eps: float = 1e-6
    clip: float | None = None

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        method: str = "standard",
        eps: float = 1e-6,
        clip: float | None = 10.0,
    ) -> "Normalizer":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Normalizer.fit expects [samples, features].")
        if values.shape[0] == 0:
            raise ValueError("Cannot fit a normalizer on an empty array.")
        if not np.isfinite(values).all():
            raise ValueError("Cannot fit a normalizer on NaN or infinite values.")

        if method == "standard":
            center = values.mean(axis=0)
            scale = values.std(axis=0)
        elif method == "minmax":
            minimum = values.min(axis=0)
            maximum = values.max(axis=0)
            center = (maximum + minimum) / 2.0
            scale = (maximum - minimum) / 2.0
        elif method == "none":
            center = np.zeros(values.shape[1], dtype=np.float32)
            scale = np.ones(values.shape[1], dtype=np.float32)
        else:
            raise ValueError("method must be one of: 'standard', 'minmax', 'none'.")

        scale = np.maximum(scale, eps)
        return cls(
            method=method,
            center=center.astype(np.float32),
            scale=scale.astype(np.float32),
            eps=eps,
            clip=clip,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        result = (values - self.center) / self.scale
        if self.clip is not None:
            result = np.clip(result, -self.clip, self.clip)
        return result.astype(np.float32, copy=False)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values * self.scale + self.center).astype(np.float32, copy=False)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "eps": self.eps,
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Normalizer":
        return cls(
            method=str(payload["method"]),
            center=np.asarray(payload["center"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            eps=float(payload.get("eps", 1e-6)),
            clip=payload.get("clip", 10.0),
        )


@dataclass
class NormalizationBundle:
    state: Normalizer
    action: Normalizer

    @classmethod
    def fit(cls, train_episodes, method: str = "standard", clip: float | None = 10.0):
        if not train_episodes:
            raise ValueError("train_episodes cannot be empty.")
        states = np.concatenate([ep.states for ep in train_episodes], axis=0)
        actions = np.concatenate([ep.actions for ep in train_episodes], axis=0)
        return cls(
            state=Normalizer.fit(states, method=method, clip=clip),
            action=Normalizer.fit(actions, method=method, clip=clip),
        )

    def save(self, path: str | Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state": self.state.to_dict(), "action": self.action.to_dict()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationBundle":
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            state=Normalizer.from_dict(payload["state"]),
            action=Normalizer.from_dict(payload["action"]),
        )
