from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .retrieval import l2_normalize


class DinoEmbedder:
    """DINOv2 embedding extractor with MPS->CPU fallback."""

    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "auto") -> None:
        self.model_name = model_name
        self.device_pref = device
        self.device = None
        self.torch = None
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:  # pragma: no cover - dependency runtime check
            raise ImportError(
                "transformers and torch are required for DINO embeddings. "
                "Install backend/requirements.txt"
            ) from exc

        self.torch = torch
        device = self._resolve_device(torch)

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def _resolve_device(self, torch) -> str:
        if self.device_pref == "cpu":
            return "cpu"
        if self.device_pref == "mps":
            return "mps"
        if self.device_pref == "auto":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return "cpu"

    def _fallback_to_cpu(self) -> None:
        if self.device == "cpu":
            return
        self.model.to("cpu")
        self.device = "cpu"

    def _forward_pil_batch(self, pil_images: list[Image.Image]) -> np.ndarray:
        if not pil_images:
            return np.zeros((0, 384), dtype=np.float32)

        inputs = self.processor(images=pil_images, return_tensors="pt")
        torch = self.torch
        assert torch is not None

        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
        except RuntimeError as exc:
            # Typical on MPS when an op is unsupported.
            if self.device == "mps":
                self._fallback_to_cpu()
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
            else:
                raise exc

        cls_token = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
        return l2_normalize(cls_token)

    def embed_rgb(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3:
            raise ValueError("embed_rgb expects HxWxC image array.")
        pil_image = Image.fromarray(image_rgb)
        return self._forward_pil_batch([pil_image])

    def embed_path(self, path: str | Path) -> np.ndarray:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        return self._forward_pil_batch([rgb])

    def embed_many_paths(self, paths: Iterable[str | Path], batch_size: int = 16) -> np.ndarray:
        path_list = [Path(p) for p in paths]
        if not path_list:
            return np.zeros((0, 384), dtype=np.float32)

        bsz = max(1, int(batch_size))
        chunks: list[np.ndarray] = []
        for start in range(0, len(path_list), bsz):
            batch_paths = path_list[start : start + bsz]
            pil_images: list[Image.Image] = []
            for path in batch_paths:
                with Image.open(path) as img:
                    pil_images.append(img.convert("RGB"))
            chunks.append(self._forward_pil_batch(pil_images))
        return np.concatenate(chunks, axis=0)
