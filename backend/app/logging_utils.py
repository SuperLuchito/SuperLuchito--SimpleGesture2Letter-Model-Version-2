from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - optional import at module load
    cv2 = None


class UncertainEventLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        *,
        query_crop_bgr,
        collage_jpeg_bytes: bytes | None,
        payload: dict[str, Any],
    ) -> dict[str, str]:
        if cv2 is None:
            raise ImportError("opencv-python is required for uncertain event logging.")
        now = datetime.now(timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
        event_dir = self.output_dir / stamp
        event_dir.mkdir(parents=True, exist_ok=True)

        query_path = event_dir / "query.jpg"
        json_path = event_dir / "event.json"
        collage_path = event_dir / "collage.jpg"

        cv2.imwrite(str(query_path), query_crop_bgr)
        if collage_jpeg_bytes:
            collage_path.write_bytes(collage_jpeg_bytes)

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return {
            "event_dir": str(event_dir),
            "query": str(query_path),
            "collage": str(collage_path if collage_jpeg_bytes else ""),
            "json": str(json_path),
        }
