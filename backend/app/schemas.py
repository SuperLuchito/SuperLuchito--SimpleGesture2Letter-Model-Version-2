from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TopKItem:
    letter: str
    score: float
    exemplar_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "letter": self.letter,
            "score": float(self.score),
            "exemplar_path": self.exemplar_path,
        }


@dataclass
class VLMDecision:
    used: bool = False
    letter: str = "NONE"
    confidence: float = 0.0
    reason: str = ""
    trigger: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "used": self.used,
            "letter": self.letter,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "trigger": self.trigger,
        }


def build_inference_message(
    *,
    status: str,
    letter: str,
    score: float,
    confidence: float,
    hand_present: bool,
    bbox_norm: list[float] | None,
    hold_elapsed_ms: int,
    hold_target_ms: int,
    text_value: str,
    committed_now: bool,
    topk: list[TopKItem],
    vlm: VLMDecision,
    sim1: float,
    sim2: float,
    margin: float,
    uncertain: bool,
    cooldown_left_ms: int,
) -> dict[str, Any]:
    remaining = max(0, hold_target_ms - hold_elapsed_ms)
    progress = min(1.0, hold_elapsed_ms / hold_target_ms) if hold_target_ms > 0 else 0.0
    return {
        "status": status,
        "letter": letter,
        "score": float(score),
        "confidence": float(confidence),
        "hand_present": hand_present,
        "bbox_norm": bbox_norm or [0.0, 0.0, 0.0, 0.0],
        "hold": {
            "elapsed_ms": int(hold_elapsed_ms),
            "remaining_ms": int(remaining),
            "target_ms": int(hold_target_ms),
            "progress": float(progress),
        },
        "text_state": {
            "value": text_value,
            "committed": committed_now,
        },
        "topk": [item.to_dict() for item in topk],
        "vlm": vlm.to_dict(),
        "debug": {
            "sim1": float(sim1),
            "sim2": float(sim2),
            "margin": float(margin),
            "uncertain": uncertain,
            "cooldown_left_ms": int(cooldown_left_ms),
        },
    }
