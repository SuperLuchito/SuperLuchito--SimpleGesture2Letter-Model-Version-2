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
        mean: tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: tuple[float, float, float] = (58.395, 57.12, 57.375),
        letterbox: bool = True,
        pad_value: int = 114,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.input_size = max(16, int(input_size))
        self.ort_num_threads = max(1, int(ort_num_threads))
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean/std must have exactly 3 values (RGB)")
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if np.any(self.std == 0.0):
            raise ValueError("std values must be non-zero")
        self.letterbox = bool(letterbox)
        self.pad_value = int(np.clip(int(pad_value), 0, 255))

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
        self._input_shape: tuple[Any, ...] = ()
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
        # Suppress known ONNX graph optimizer warnings for released Slovo checkpoints.
        options.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._input_shape = tuple(model_input.shape)
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
            "---",
            "background",
        }
        for idx, label in enumerate(self.labels):
            normalized = label.strip().lower()
            if normalized in aliases:
                return int(idx)
        return None

    def infer_probs(self, clip_bgr: list[np.ndarray]) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        x_6d, x_cthw, x_tchw = self._preprocess_clip(clip_bgr)

        rank = len(self._input_shape)
        output: np.ndarray
        if rank == 6:
            raw = self._session.run([self._output_name], {self._input_name: x_6d})[0]
            output = np.asarray(raw)
        elif rank == 5:
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
        else:
            raise ValueError(
                f"Unsupported ONNX input rank {rank} for {self.model_path}. "
                "Expected 5D or 6D video tensor."
            )

        latency_ms = (time.perf_counter() - started) * 1000.0

        logits = output.reshape(-1).astype(np.float32)
        if logits.size != len(self.labels):
            raise ValueError(f"model output size {logits.size} does not match labels size {len(self.labels)}")

        probs = self._to_probs(logits)
        return probs, float(latency_ms)

    def _preprocess_clip(self, clip_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not clip_bgr:
            raise ValueError("clip is empty")

        try:
            import cv2
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError("opencv-python is required for ONNX words preprocess") from exc

        frames_chw: list[np.ndarray] = []
        for frame in clip_bgr:
            if self.letterbox:
                resized = self._letterbox(frame, (self.input_size, self.input_size), cv2)
            else:
                resized = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            x = rgb.astype(np.float32)
            x = (x - self.mean) / self.std
            x = np.transpose(x, (2, 0, 1))
            frames_chw.append(x)

        arr_cthw = np.stack(frames_chw, axis=1)  # [C, T, H, W]
        arr_tchw = np.transpose(arr_cthw, (1, 0, 2, 3))  # [T, C, H, W]

        x_6d = np.expand_dims(np.expand_dims(arr_cthw, axis=0), axis=0).astype(np.float32, copy=False)  # [1, 1, C, T, H, W]
        x_cthw = np.expand_dims(arr_cthw, axis=0).astype(np.float32, copy=False)
        x_tchw = np.expand_dims(arr_tchw, axis=0).astype(np.float32, copy=False)
        return (
            np.ascontiguousarray(x_6d),
            np.ascontiguousarray(x_cthw),
            np.ascontiguousarray(x_tchw),
        )

    def _letterbox(self, frame_bgr: np.ndarray, new_shape: tuple[int, int], cv2: Any) -> np.ndarray:
        shape = frame_bgr.shape[:2]  # (h, w)
        target_h, target_w = new_shape
        r = min(target_h / shape[0], target_w / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        dw = (target_w - new_unpad[0]) / 2.0
        dh = (target_h - new_unpad[1]) / 2.0

        if shape[::-1] != new_unpad:
            frame_bgr = cv2.resize(frame_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return cv2.copyMakeBorder(
            frame_bgr,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(self.pad_value, self.pad_value, self.pad_value),
        )

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
