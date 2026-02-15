from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    frontend_fps: int = 12
    jpeg_quality: float = 0.75
    hold_ms: int = 700
    cooldown_ms: int = 500
    retrieval_k: int = 5
    sim_none: float = 0.38
    sim_vlm_th: float = 0.52
    margin_th: float = 0.06
    precommit_ratio: float = 0.8
    min_bbox_area: float = 0.04
    enable_vlm_judge: bool = False
    vlm_min_confidence: float = 0.65
    vlm_timeout_ms: int = 1500
    vlm_base_url: str = "http://localhost:1234/v1"
    vlm_model: str = "qwen3-vl-4b-instruct"
    embedding_model: str = "facebook/dinov2-small"
    device: str = "auto"
    letters_allowlist: list[str] | None = None
    none_label_dir: str = "_none"
    debug_overlay: bool = True
    log_uncertain_events: bool = True
    uncertain_streak_frames: int = 4
    switch_min_frames: int = 3
    vlm_min_interval_ms: int = 1800
    hand_landmarker_model_path: str = "backend/artifacts/models/hand_landmarker.task"
    hand_bbox_padding: float = 0.2
    hand_focus_ratio: float = 1.0
    hand_wrist_extension_ratio: float = 0.18
    hand_bg_suppression: bool = True
    hand_bg_darken_factor: float = 0.45
    hand_mask_dilate_ratio: float = 0.08
    hand_mask_blur_sigma: float = 3.0
    max_num_hands: int = 1
    hand_min_detection_confidence: float = 0.55
    hand_min_presence_confidence: float = 0.55
    hand_min_tracking_confidence: float = 0.55

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        cfg = cls(**filtered)
        cfg.letters_allowlist = cfg.letters_allowlist or []
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            return cls()
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config at {cfg_path} must be a mapping.")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["letters_allowlist"] = self.letters_allowlist or []
        return result

    def write_yaml(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


def load_config(config_path: str | Path) -> AppConfig:
    return AppConfig.from_yaml(config_path)


def merge_config_values(config_path: str | Path, updates: dict[str, Any]) -> AppConfig:
    current = load_config(config_path)
    payload = current.to_dict()
    payload.update(updates)
    merged = AppConfig.from_dict(payload)
    merged.write_yaml(config_path)
    return merged
