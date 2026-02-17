from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import TopKItem, VLMDecision, build_inference_message

from .buffer import FrameRingBuffer
from .decoder import WordDecisionDecoder, WordThresholds
from .metrics import WordRuntimeMetrics
from .model_onnx import WordOnnxModel


@dataclass
class WordServiceConfig:
    window_frames: int
    frame_interval: int
    step: int
    topk: int
    max_fps_inference: float
    no_event_label: str
    ema_alpha: float
    hold_frames: int
    cooldown_frames: int
    dedup_same_word: bool
    thresholds: WordThresholds
    log_enabled: bool
    log_path: Path


class WordRecognitionService:
    def __init__(self, *, model: WordOnnxModel, config: WordServiceConfig) -> None:
        self.model = model
        self.config = config

        maxlen = max(64, (config.window_frames * max(1, config.frame_interval) * 2) + 8)
        self.buffer = FrameRingBuffer(maxlen=maxlen)
        self.decoder = WordDecisionDecoder(
            ema_alpha=config.ema_alpha,
            thresholds=config.thresholds,
            hold_frames=config.hold_frames,
            cooldown_frames=config.cooldown_frames,
            dedup_same_word=config.dedup_same_word,
        )
        self.metrics = WordRuntimeMetrics()

        self.no_event_index = self.model.find_no_event_index(config.no_event_label)
        self.frame_counter = 0
        self.last_infer_started_s: float = 0.0
        self._last_payload: dict[str, Any] | None = None

    def clear_text(self) -> None:
        self.decoder.clear_text()

    def update(self, frame_bgr, now_ms: int) -> dict[str, Any]:
        now_ms = int(now_ms)
        self.frame_counter += 1
        self.buffer.append(frame_bgr, timestamp_ms=now_ms)

        if not self.buffer.is_ready(
            window_frames=self.config.window_frames,
            frame_interval=self.config.frame_interval,
        ):
            payload = self._build_message(
                now_ms=now_ms,
                state="NONE",
                top1="NONE",
                top1_prob=0.0,
                topk=[],
                hold_count=0,
                hold_target=self.config.hold_frames,
                hold_progress=0.0,
                cooldown_left=0,
                latency_ms=None,
                margin=0.0,
                no_event_prob=0.0,
                committed=False,
            )
            self.metrics.record_state(timestamp_ms=now_ms, state="NONE", committed_word=None)
            self._last_payload = dict(payload)
            return payload

        if (self.frame_counter % max(1, self.config.step)) != 0:
            return self._carry_last_payload(now_ms=now_ms)

        if self.config.max_fps_inference > 0.0:
            min_dt = 1.0 / self.config.max_fps_inference
            now_s = time.perf_counter()
            if (now_s - self.last_infer_started_s) < min_dt:
                return self._carry_last_payload(now_ms=now_ms)

        clip, _ = self.buffer.sample_clip(
            window_frames=self.config.window_frames,
            frame_interval=self.config.frame_interval,
        )

        self.last_infer_started_s = time.perf_counter()
        probs, latency_ms = self.model.infer_probs(clip)
        self.metrics.record_inference(latency_ms)

        decoded = self.decoder.update(
            probs=probs,
            labels=self.model.labels,
            topk=self.config.topk,
            no_event_index=self.no_event_index,
        )

        topk = [(self.model.labels[idx], float(probs[idx])) for idx in decoded.topk_indices]

        payload = self._build_message(
            now_ms=now_ms,
            state=decoded.state,
            top1=decoded.top1_label,
            top1_prob=decoded.top1_prob,
            topk=topk,
            hold_count=decoded.hold_count,
            hold_target=decoded.hold_target,
            hold_progress=decoded.hold_progress,
            cooldown_left=decoded.cooldown_left,
            latency_ms=latency_ms,
            margin=decoded.margin,
            no_event_prob=decoded.no_event_prob,
            committed=decoded.committed,
        )

        committed_word = decoded.committed_word if decoded.committed else None
        self.metrics.record_state(timestamp_ms=now_ms, state=decoded.state, committed_word=committed_word)
        self._append_log(now_ms=now_ms, payload=payload)
        self._last_payload = dict(payload)
        return payload

    def _carry_last_payload(self, *, now_ms: int) -> dict[str, Any]:
        if self._last_payload is None:
            return self._build_message(
                now_ms=now_ms,
                state="NONE",
                top1="NONE",
                top1_prob=0.0,
                topk=[],
                hold_count=0,
                hold_target=self.config.hold_frames,
                hold_progress=0.0,
                cooldown_left=0,
                latency_ms=None,
                margin=0.0,
                no_event_prob=0.0,
                committed=False,
            )
        payload = dict(self._last_payload)
        payload["timestamp_ms"] = int(now_ms)
        return payload

    def _build_message(
        self,
        *,
        now_ms: int,
        state: str,
        top1: str,
        top1_prob: float,
        topk: list[tuple[str, float]],
        hold_count: int,
        hold_target: int,
        hold_progress: float,
        cooldown_left: int,
        latency_ms: float | None,
        margin: float,
        no_event_prob: float,
        committed: bool,
    ) -> dict[str, Any]:
        latency = self.metrics.latency_summary()
        fp_min = self.metrics.fp_per_minute()

        topk_items = [TopKItem(letter=label, score=score, exemplar_path="") for label, score in topk]

        payload = build_inference_message(
            status=state,
            letter=top1,
            word=top1,
            score=float(top1_prob),
            confidence=float(top1_prob),
            hand_present=state != "NONE",
            bbox_norm=[0.0, 0.0, 0.0, 0.0],
            hold_elapsed_ms=int(hold_count),
            hold_target_ms=int(max(1, hold_target)),
            text_value=self.decoder.text_value,
            committed_now=bool(committed),
            topk=topk_items,
            vlm=VLMDecision(),
            sim1=float(top1_prob),
            sim2=float(max(0.0, top1_prob - margin)),
            margin=float(margin),
            uncertain=state in {"UNKNOWN"},
            cooldown_left_ms=int(cooldown_left),
            mode="words",
            hold_unit="frames",
            latency_ms=latency_ms,
            fp_per_minute=fp_min,
            avg_infer_latency_ms=latency.avg_ms,
            p95_infer_latency_ms=latency.p95_ms,
        )

        payload["top1"] = {
            "label": top1,
            "prob": float(top1_prob),
            "no_event_prob": float(no_event_prob),
        }
        payload["latency_ms"] = float(latency_ms) if latency_ms is not None else None
        payload["timestamp_ms"] = int(now_ms)
        payload["state_detail"] = {
            "hold_frames": int(hold_count),
            "hold_target_frames": int(hold_target),
            "hold_progress": float(hold_progress),
            "cooldown_left_frames": int(cooldown_left),
        }
        return payload

    def _append_log(self, *, now_ms: int, payload: dict[str, Any]) -> None:
        if not self.config.log_enabled:
            return
        log_path = self.config.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        topk_serialized = [
            {
                "label": str(item.get("letter", "")),
                "score": float(item.get("score", 0.0)),
            }
            for item in payload.get("topk", [])
        ]

        line = {
            "timestamp_ms": int(now_ms),
            "state": payload.get("status", ""),
            "word": payload.get("word", payload.get("letter", "")),
            "score": float(payload.get("score", 0.0)),
            "latency_ms": payload.get("latency_ms"),
            "topk": topk_serialized,
            "fp_per_minute": payload.get("debug", {}).get("fp_per_minute"),
            "avg_infer_latency_ms": payload.get("debug", {}).get("avg_infer_latency_ms"),
            "p95_infer_latency_ms": payload.get("debug", {}).get("p95_infer_latency_ms"),
        }

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
