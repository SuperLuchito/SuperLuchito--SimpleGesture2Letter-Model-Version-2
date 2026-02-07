from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

try:
    import cv2
except Exception:  # pragma: no cover - optional import at module load
    cv2 = None
import numpy as np

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class HandDetection:
    hand_present: bool
    bbox_norm: tuple[float, float, float, float]
    bbox_px: tuple[int, int, int, int]
    crop_bgr: np.ndarray | None


class HandDetector:
    def __init__(
        self,
        model_path: str,
        *,
        min_bbox_area: float = 0.04,
        bbox_padding: float = 0.15,
        focus_ratio: float = 0.92,
        bg_suppression: bool = True,
        bg_darken_factor: float = 0.45,
        mask_dilate_ratio: float = 0.08,
        mask_blur_sigma: float = 3.0,
        max_num_hands: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.min_bbox_area = min_bbox_area
        self.bbox_padding = bbox_padding
        self.focus_ratio = float(np.clip(focus_ratio, 0.7, 1.0))
        self.bg_suppression = bool(bg_suppression)
        self.bg_darken_factor = float(np.clip(bg_darken_factor, 0.0, 1.0))
        self.mask_dilate_ratio = float(max(0.0, mask_dilate_ratio))
        self.mask_blur_sigma = float(max(0.0, mask_blur_sigma))
        self.max_num_hands = max_num_hands

        self.mp = None
        self.landmarker = None
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists():
            urlretrieve(HAND_LANDMARKER_URL, self.model_path)

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "mediapipe is required for Hand Landmarker. Install backend/requirements.txt"
            ) from exc

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.mp = mp

    def _tighten_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> tuple[float, float, float, float]:
        if self.focus_ratio >= 0.999:
            return x1, y1, x2, y2

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1) * self.focus_ratio
        h = (y2 - y1) * self.focus_ratio

        tx1 = float(np.clip(cx - (w * 0.5), 0.0, 1.0))
        ty1 = float(np.clip(cy - (h * 0.5), 0.0, 1.0))
        tx2 = float(np.clip(cx + (w * 0.5), 0.0, 1.0))
        ty2 = float(np.clip(cy + (h * 0.5), 0.0, 1.0))
        return tx1, ty1, tx2, ty2

    @staticmethod
    def _landmarks_to_crop_points(
        landmarks: Any,
        *,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        crop_w: int,
        crop_h: int,
    ) -> np.ndarray:
        bw = max(1e-6, x2 - x1)
        bh = max(1e-6, y2 - y1)
        pts: list[tuple[int, int]] = []
        for lm in landmarks:
            cx = int(round(((lm.x - x1) / bw) * crop_w))
            cy = int(round(((lm.y - y1) / bh) * crop_h))
            cx = int(np.clip(cx, 0, max(0, crop_w - 1)))
            cy = int(np.clip(cy, 0, max(0, crop_h - 1)))
            pts.append((cx, cy))
        if not pts:
            return np.empty((0, 2), dtype=np.int32)
        return np.array(pts, dtype=np.int32)

    def _suppress_crop_background(self, crop_bgr: np.ndarray, crop_points: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return crop_bgr
        if crop_points.shape[0] < 3:
            return crop_bgr

        ch, cw = crop_bgr.shape[:2]
        if ch == 0 or cw == 0:
            return crop_bgr

        hull = cv2.convexHull(crop_points.reshape(-1, 1, 2))
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        dilate_px = int(round(max(ch, cw) * self.mask_dilate_ratio))
        dilate_px = max(1, dilate_px)
        if dilate_px > 1:
            kernel = np.ones((dilate_px, dilate_px), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        if self.mask_blur_sigma > 0.0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=self.mask_blur_sigma, sigmaY=self.mask_blur_sigma)

        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        bg = (crop_bgr.astype(np.float32) * self.bg_darken_factor).astype(np.float32)
        fg = crop_bgr.astype(np.float32)
        fused = (fg * alpha) + (bg * (1.0 - alpha))
        return np.clip(fused, 0, 255).astype(np.uint8)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandDetection:
        if cv2 is None:
            raise ImportError("opencv-python is required for hand detection.")
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return HandDetection(
                hand_present=False,
                bbox_norm=(0.0, 0.0, 0.0, 0.0),
                bbox_px=(0, 0, 0, 0),
                crop_bgr=None,
            )

        landmarks = result.hand_landmarks[0]
        xs = np.array([lm.x for lm in landmarks], dtype=np.float32)
        ys = np.array([lm.y for lm in landmarks], dtype=np.float32)

        x1 = float(np.clip(xs.min() - self.bbox_padding, 0.0, 1.0))
        y1 = float(np.clip(ys.min() - self.bbox_padding, 0.0, 1.0))
        x2 = float(np.clip(xs.max() + self.bbox_padding, 0.0, 1.0))
        y2 = float(np.clip(ys.max() + self.bbox_padding, 0.0, 1.0))
        x1, y1, x2, y2 = self._tighten_bbox(x1, y1, x2, y2)

        area = max(0.0, (x2 - x1) * (y2 - y1))
        if area < self.min_bbox_area:
            return HandDetection(
                hand_present=False,
                bbox_norm=(x1, y1, x2, y2),
                bbox_px=(0, 0, 0, 0),
                crop_bgr=None,
            )

        px1 = int(round(x1 * w))
        py1 = int(round(y1 * h))
        px2 = int(round(x2 * w))
        py2 = int(round(y2 * h))
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)

        if px2 <= px1 or py2 <= py1:
            return HandDetection(
                hand_present=False,
                bbox_norm=(x1, y1, x2, y2),
                bbox_px=(0, 0, 0, 0),
                crop_bgr=None,
            )

        crop = frame_bgr[py1:py2, px1:px2].copy()
        if self.bg_suppression:
            points = self._landmarks_to_crop_points(
                landmarks,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                crop_w=(px2 - px1),
                crop_h=(py2 - py1),
            )
            crop = self._suppress_crop_background(crop, points)

        return HandDetection(
            hand_present=True,
            bbox_norm=(x1, y1, x2, y2),
            bbox_px=(px1, py1, px2, py2),
            crop_bgr=crop,
        )
