from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np


class WordOnnxModel:
    def __init__(
        self,
        *,
        model_path: str | Path,
        labels_path: str | Path,
        input_size: int = 224,
        ort_num_threads: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.input_size = max(16, int(input_size))
        self.ort_num_threads = max(1, int(ort_num_threads))

        if not self.model_path.exists():
            raise FileNotFoundError(f"Word model not found: {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Word labels not found: {self.labels_path}")

        self.labels = self._load_labels(self.labels_path)
        if not self.labels:
            raise ValueError(f"labels file is empty: {self.labels_path}")

        self._session: Any = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._input_layout: str | None = None
        self._init_session()

    @staticmethod
    def _load_labels(path: Path) -> list[str]:
        labels: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value:
                labels.append(value)
        return labels

    def _init_session(self) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError("onnxruntime is required for words mode inference") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.ort_num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def find_no_event_index(self, label_name: str) -> int | None:
        wanted = str(label_name).strip().lower()
        aliases = {
            wanted,
            wanted.replace("-", "_"),
            wanted.replace("_", "-"),
            "_no_event",
            "no_event",
            "none",
        }
        for idx, label in enumerate(self.labels):
            normalized = label.strip().lower()
            if normalized in aliases:
                return int(idx)
        return None

    def infer_probs(self, clip_bgr: list[np.ndarray]) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        x_cthw, x_tchw = self._preprocess_clip(clip_bgr)

        output: np.ndarray
        if self._input_layout in {None, "cthw"}:
            try:
                raw = self._session.run([self._output_name], {self._input_name: x_cthw})[0]
                self._input_layout = "cthw"
                output = np.asarray(raw)
            except Exception:
                raw = self._session.run([self._output_name], {self._input_name: x_tchw})[0]
                self._input_layout = "tchw"
                output = np.asarray(raw)
        else:
            raw = self._session.run([self._output_name], {self._input_name: x_tchw})[0]
            output = np.asarray(raw)

        latency_ms = (time.perf_counter() - started) * 1000.0

        logits = output.reshape(-1).astype(np.float32)
        if logits.size != len(self.labels):
            raise ValueError(f"model output size {logits.size} does not match labels size {len(self.labels)}")

        probs = self._to_probs(logits)
        return probs, float(latency_ms)

    def _preprocess_clip(self, clip_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not clip_bgr:
            raise ValueError("clip is empty")

        try:
            import cv2
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError("opencv-python is required for ONNX words preprocess") from exc

        # TODO(slovo): verify exact preprocess used by selected baseline checkpoint.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        frames_chw: list[np.ndarray] = []
        for frame in clip_bgr:
            resized = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            x = rgb.astype(np.float32) / 255.0
            x = (x - mean) / std
            x = np.transpose(x, (2, 0, 1))
            frames_chw.append(x)

        arr_tchw = np.stack(frames_chw, axis=0)  # [T, C, H, W]
        arr_cthw = np.transpose(arr_tchw, (1, 0, 2, 3))
        x_cthw = np.expand_dims(arr_cthw, axis=0).astype(np.float32, copy=False)
        x_tchw = np.expand_dims(arr_tchw, axis=0).astype(np.float32, copy=False)
        return np.ascontiguousarray(x_cthw), np.ascontiguousarray(x_tchw)

    @staticmethod
    def _to_probs(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(v)):
            clean = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            v = clean

        s = float(v.sum())
        if float(v.min()) >= 0.0 and float(v.max()) <= 1.0 and math.isclose(s, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return v

        m = float(np.max(v))
        exp = np.exp(v - m)
        denom = float(exp.sum())
        if denom <= 0.0:
            return np.zeros_like(v)
        return exp / denom
