from __future__ import annotations

from typing import Sequence
import numpy as np


class FrozenDINOv2:
    """
    Frozen Hugging Face/PyTorch DINOv2 feature extractor.

    This is intentionally separate from the JAX/Equinox trainable model:
    DINOv2 produces fixed patch tokens; A-JEPA consumes them as NumPy/JAX arrays.
    Input images should be RGB uint8 arrays [H, W, 3] or a sequence of them.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "Install vision dependencies with: uv add torch transformers pillow"
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        config_dim = getattr(self.model.config, "hidden_size", None)
        if config_dim != 384:
            raise ValueError(
                f"Expected DINOv2-Small hidden size 384, got {config_dim}. "
                "Check that you selected facebook/dinov2-small."
            )
        self.output_dim = config_dim

    def encode(self, images: Sequence[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if len(images) == 0:
            raise ValueError("images must not be empty")
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                pil_images = []
                for image in batch:
                    arr = np.asarray(image)
                    if arr.ndim != 3 or arr.shape[-1] != 3:
                        raise ValueError("Each image must have shape [H, W, 3] RGB")
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                    from PIL import Image
                    pil_images.append(Image.fromarray(arr, mode="RGB"))
                inputs = self.processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                result = self.model(**inputs)
                # Remove CLS token; keep spatial patch tokens.
                patch_tokens = result.last_hidden_state[:, 1:, :]
                outputs.append(patch_tokens.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)
