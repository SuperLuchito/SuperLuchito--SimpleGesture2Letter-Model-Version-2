#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import load_config
from app.hand_detector import HandDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture gallery samples from webcam")
    parser.add_argument("--label", required=True, help="Буква или _none")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("--auto", action="store_true", help="Автосохранение по таймеру")
    parser.add_argument("--config", default=str(BACKEND / "config.yaml"))
    parser.add_argument("--gallery", default=str(BACKEND / "gallery"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.gallery) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("[capture_gallery] Не удалось открыть камеру")
        return 1

    detector = None
    if args.label != cfg.none_label_dir:
        model_path = Path(cfg.hand_landmarker_model_path)
        if not model_path.is_absolute():
            model_path = (ROOT / model_path).resolve()
        detector = HandDetector(
            str(model_path),
            min_bbox_area=cfg.min_bbox_area,
            bbox_padding=cfg.hand_bbox_padding,
            focus_ratio=cfg.hand_focus_ratio,
            bg_suppression=cfg.hand_bg_suppression,
            bg_darken_factor=cfg.hand_bg_darken_factor,
            mask_dilate_ratio=cfg.hand_mask_dilate_ratio,
            mask_blur_sigma=cfg.hand_mask_blur_sigma,
            max_num_hands=cfg.max_num_hands,
        )

    auto_mode = args.auto
    last_save_ms = 0
    saved = 0

    print("[capture_gallery] Горячие клавиши: s=save, a=auto on/off, q=quit")

    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            continue

        ts_ms = int(time.monotonic() * 1000)
        crop = None

        if detector is not None:
            det = detector.detect(frame, ts_ms)
            if det.hand_present and det.crop_bgr is not None:
                crop = det.crop_bgr
                x1, y1, x2, y2 = det.bbox_px
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 80), 2)
            cv2.putText(
                frame,
                f"hand={det.hand_present}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 230, 80) if det.hand_present else (60, 60, 220),
                2,
            )
        else:
            crop = frame
            cv2.putText(
                frame,
                "NONE capture mode",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 200, 80),
                2,
            )

        cv2.putText(
            frame,
            f"label={args.label} saved={saved}/{args.count} auto={auto_mode}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if auto_mode and crop is not None and (ts_ms - last_save_ms) >= args.interval_ms:
            saved += 1
            filename = out_dir / f"{int(time.time() * 1000)}_{saved:03d}.jpg"
            cv2.imwrite(str(filename), crop)
            last_save_ms = ts_ms
            print(f"[capture_gallery] saved: {filename}")

        cv2.imshow("capture_gallery", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("a"):
            auto_mode = not auto_mode
        if key == ord("s") and crop is not None:
            saved += 1
            filename = out_dir / f"{int(time.time() * 1000)}_{saved:03d}.jpg"
            cv2.imwrite(str(filename), crop)
            last_save_ms = ts_ms
            print(f"[capture_gallery] saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"[capture_gallery] Завершено. Сохранено {saved} файлов в {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
