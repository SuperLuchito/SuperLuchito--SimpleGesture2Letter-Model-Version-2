#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import load_config
from app.hand_detector import HandDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MANIFEST_COLUMNS = [
    "label",
    "session_id",
    "capture_mode",
    "timestamp_utc",
    "timestamp_ms",
    "camera_id",
    "bbox_area",
    "sharpness",
    "bright_ratio",
    "distance_bucket",
    "mirror",
    "frame_delta",
    "bbox_delta",
    "dedup_strategy",
    "path",
]

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
    "DejaVuSans.ttf",
]

TEXT_WHITE = (255, 255, 255)
TEXT_OK = (80, 230, 80)
TEXT_WARN = (255, 210, 80)
TEXT_BAD = (60, 60, 220)
TextItem = tuple[str, tuple[int, int], tuple[int, int, int], int]


def build_session_id(explicit: str | None) -> str:
    if explicit is None or not explicit.strip():
        return datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ")

    value = explicit.strip().replace("/", "-").replace("\\", "-").replace(" ", "_")
    if not value:
        raise ValueError("session_id must not be empty")
    return value


def similarity_gray(image_bgr: np.ndarray, size: int = 160) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def mean_abs_frame_delta(curr_ref: np.ndarray, prev_ref: np.ndarray) -> float:
    return float(np.mean(np.abs(curr_ref - prev_ref)))


def compute_quality_metrics(crop_bgr: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    bright_ratio = float(np.mean(gray >= 245))
    return sharpness, bright_ratio


def distance_bucket_from_bbox(bbox_area: float) -> str:
    if bbox_area <= 0.0:
        return "unknown"
    if bbox_area < 0.08:
        return "far"
    if bbox_area <= 0.18:
        return "mid"
    return "near"


def append_manifest(jsonl_path: Path, csv_path: Path, row: dict[str, str | int | float]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if not csv_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in MANIFEST_COLUMNS})


@lru_cache(maxsize=16)
def get_unicode_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_size = max(10, int(size))
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text_items(frame_bgr: np.ndarray, items: list[TextItem]) -> np.ndarray:
    if not items:
        return frame_bgr

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for text, (x, y), color_bgr, size in items:
        font = get_unicode_font(size)
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        draw.text(
            (int(x), int(y)),
            text,
            fill=color_rgb,
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    out_rgb = np.asarray(image)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def build_operator_hints(
    *,
    detector_enabled: bool,
    hand_present: bool,
    bbox_area: float,
    bbox_norm: tuple[float, float, float, float],
    frame_bgr: np.ndarray,
    min_capture_bbox_area: float,
) -> list[str]:
    hints: list[str] = []

    scene_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    scene_luma = float(np.mean(scene_gray))
    if scene_luma < 55.0:
        hints.append("темно")

    if detector_enabled:
        if not hand_present:
            hints.append("покажите руку в кадре")
        else:
            if bbox_area > 0.30:
                hints.append("слишком близко")
            if 0.0 < bbox_area < min_capture_bbox_area:
                hints.append("поднесите руку ближе")

            x1, y1, x2, y2 = bbox_norm
            bw = max(1e-6, x2 - x1)
            bh = max(1e-6, y2 - y1)
            aspect = bw / bh
            if aspect < 0.55 or aspect > 1.8:
                hints.append("поверните руку фронтально")

    if not hints:
        return ["ок"]
    return list(dict.fromkeys(hints))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture gallery samples from webcam")
    parser.add_argument("--label", required=True, help="Буква или _none")
    parser.add_argument("--session-id", default="", help="ID сессии (по умолчанию авто session_YYYYMMDD...)")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--interval-ms", type=int, default=1000, help="Интервал авто-попыток сохранения")
    parser.add_argument("--auto", action="store_true", help="Автосохранение по таймеру")
    parser.add_argument("--mirror", action="store_true", help="Зеркалить кадры (preview + сохранение)")
    parser.add_argument("--config", default=str(BACKEND / "config.yaml"))
    parser.add_argument("--gallery", default=str(BACKEND / "gallery"))
    parser.add_argument("--min-capture-bbox-area", type=float, default=0.06, help="Мин. площадь bbox руки")
    parser.add_argument(
        "--min-save-interval-ms",
        type=int,
        default=1000,
        help="Минимальная пауза между сохранениями (manual + auto)",
    )
    parser.add_argument(
        "--min-frame-delta",
        type=float,
        default=8.0,
        help="Мин. изменение кадра (mean abs gray diff) для сохранения",
    )
    parser.add_argument(
        "--min-bbox-delta",
        type=float,
        default=0.015,
        help="Мин. изменение площади bbox для режима с рукой",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    label_dir = Path(args.gallery) / args.label
    label_dir.mkdir(parents=True, exist_ok=True)
    session_id = build_session_id(args.session_id)
    session_dir = label_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest_jsonl = session_dir / "manifest.jsonl"
    manifest_csv = session_dir / "manifest.csv"
    min_capture_bbox_area = max(float(cfg.min_bbox_area), float(args.min_capture_bbox_area))

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
            wrist_extension_ratio=cfg.hand_wrist_extension_ratio,
            bg_suppression=cfg.hand_bg_suppression,
            bg_darken_factor=cfg.hand_bg_darken_factor,
            mask_dilate_ratio=cfg.hand_mask_dilate_ratio,
            mask_blur_sigma=cfg.hand_mask_blur_sigma,
            max_num_hands=cfg.max_num_hands,
        )

    auto_mode = args.auto
    last_save_ms = 0
    last_auto_attempt_ms = 0
    saved = 0
    last_saved_ref: np.ndarray | None = None
    last_saved_bbox_area: float | None = None
    min_save_interval_ms = max(0, int(args.min_save_interval_ms))
    min_frame_delta = max(0.0, float(args.min_frame_delta))
    min_bbox_delta = max(0.0, float(args.min_bbox_delta))

    print("[capture_gallery] Горячие клавиши: s=save, a=auto on/off, q=quit")
    print(f"[capture_gallery] Session: {session_id}")
    print(f"[capture_gallery] Output dir: {session_dir}")
    print(f"[capture_gallery] Mirror: {bool(args.mirror)}")
    print(
        "[capture_gallery] Min dedup: "
        f"min_save_interval_ms={min_save_interval_ms}, "
        f"min_frame_delta={min_frame_delta:.2f}, "
        f"min_bbox_delta={min_bbox_delta:.4f}"
    )
    print(
        "[capture_gallery] Hint thresholds: "
        f"min_capture_bbox_area={min_capture_bbox_area:.4f}"
    )

    def try_save(crop_bgr: np.ndarray | None, *, capture_mode: str, ts_ms: int, bbox_area: float) -> bool:
        nonlocal saved, last_save_ms, last_saved_ref, last_saved_bbox_area

        if crop_bgr is None:
            print(f"[capture_gallery] skip ({capture_mode}): no_crop")
            return False

        if (ts_ms - last_save_ms) < min_save_interval_ms:
            wait_left = int(min_save_interval_ms - (ts_ms - last_save_ms))
            print(f"[capture_gallery] skip ({capture_mode}): min_interval ({wait_left}ms left)")
            return False

        # Русский комментарий: минимальный дедуп.
        # Сохраняем кадр только если прошло нужное время и поза/кадр изменились.
        current_ref = similarity_gray(crop_bgr)
        frame_delta = -1.0
        bbox_delta = -1.0
        changed_by_frame = True
        changed_by_bbox = True

        if last_saved_ref is not None:
            frame_delta = mean_abs_frame_delta(current_ref, last_saved_ref)
            changed_by_frame = frame_delta >= min_frame_delta

        if detector is not None:
            if last_saved_bbox_area is None:
                changed_by_bbox = True
            else:
                bbox_delta = abs(float(bbox_area) - float(last_saved_bbox_area))
                changed_by_bbox = bbox_delta >= min_bbox_delta

        pose_changed = changed_by_frame or changed_by_bbox
        if not pose_changed:
            print(
                f"[capture_gallery] skip ({capture_mode}): no_pose_change "
                f"(frame_delta={frame_delta:.3f}, bbox_delta={bbox_delta:.5f})"
            )
            return False

        saved += 1
        now_utc = datetime.now(timezone.utc)
        ts_wall_ms = int(now_utc.timestamp() * 1000)
        filename = session_dir / f"{ts_wall_ms}_{saved:03d}.jpg"
        cv2.imwrite(str(filename), crop_bgr)
        last_save_ms = ts_ms

        sharpness, bright_ratio = compute_quality_metrics(crop_bgr)
        last_saved_ref = current_ref
        if detector is not None:
            last_saved_bbox_area = float(bbox_area)
        distance_bucket = distance_bucket_from_bbox(float(bbox_area))

        row = {
            "label": args.label,
            "session_id": session_id,
            "capture_mode": capture_mode,
            "timestamp_utc": now_utc.isoformat(),
            "timestamp_ms": ts_wall_ms,
            "camera_id": args.camera_id,
            "bbox_area": round(float(bbox_area), 6),
            "sharpness": round(sharpness, 4),
            "bright_ratio": round(bright_ratio, 6),
            "distance_bucket": distance_bucket,
            "mirror": bool(args.mirror),
            "frame_delta": round(frame_delta, 6) if frame_delta >= 0.0 else "",
            "bbox_delta": round(bbox_delta, 6) if bbox_delta >= 0.0 else "",
            "dedup_strategy": "min_interval+pose_change",
            "path": str(filename.resolve()),
        }
        append_manifest(manifest_jsonl, manifest_csv, row)
        print(f"[capture_gallery] saved: {filename}")
        return True

    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            continue
        if args.mirror:
            frame = cv2.flip(frame, 1)

        ts_ms = int(time.monotonic() * 1000)
        crop = None
        bbox_area = 0.0
        bbox_norm = (0.0, 0.0, 0.0, 0.0)
        hand_present = False
        text_items: list[TextItem] = []

        if detector is not None:
            det = detector.detect(frame, ts_ms)
            hand_present = bool(det.hand_present)
            if det.hand_present and det.crop_bgr is not None:
                crop = det.crop_bgr
                x1, y1, x2, y2 = det.bbox_px
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 80), 2)
                nx1, ny1, nx2, ny2 = det.bbox_norm
                bbox_area = float(max(0.0, (nx2 - nx1) * (ny2 - ny1)))
                bbox_norm = (nx1, ny1, nx2, ny2)
            text_items.append(
                (
                    f"рука={det.hand_present} area={bbox_area:.4f}",
                    (12, 12),
                    TEXT_OK if det.hand_present else TEXT_BAD,
                    20,
                )
            )
        else:
            crop = frame
            text_items.append(("режим NONE", (12, 12), TEXT_WARN, 22))

        hints = build_operator_hints(
            detector_enabled=(detector is not None),
            hand_present=hand_present,
            bbox_area=bbox_area,
            bbox_norm=bbox_norm,
            frame_bgr=frame,
            min_capture_bbox_area=min_capture_bbox_area,
        )
        primary_hint = hints[0]
        text_items.append(
            (
                f"подсказка: {primary_hint}",
                (12, 44),
                TEXT_OK if primary_hint == "ок" else TEXT_WARN,
                21,
            )
        )
        for idx, hint in enumerate(hints[1:3], start=1):
            text_items.append(
                (
                    f"подсказка{idx + 1}: {hint}",
                    (12, 44 + (idx * 24)),
                    TEXT_WARN,
                    19,
                )
            )

        distance_bucket = distance_bucket_from_bbox(float(bbox_area))
        text_items.append(
            (
                f"дистанция={distance_bucket} interval={min_save_interval_ms}ms",
                (12, 98 if len(hints) <= 1 else 142),
                TEXT_WHITE,
                20,
            )
        )
        text_items.append(
            (
                f"буква={args.label} saved={saved}/{args.count} auto={auto_mode}",
                (12, 124 if len(hints) <= 1 else 168),
                TEXT_WHITE,
                20,
            )
        )

        frame = draw_text_items(frame, text_items)

        if auto_mode and crop is not None and (ts_ms - last_auto_attempt_ms) >= args.interval_ms:
            last_auto_attempt_ms = ts_ms
            try_save(crop, capture_mode="auto", ts_ms=ts_ms, bbox_area=bbox_area)

        cv2.imshow("capture_gallery", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("a"):
            auto_mode = not auto_mode
        if key == ord("s") and crop is not None:
            try_save(crop, capture_mode="manual", ts_ms=ts_ms, bbox_area=bbox_area)

    cap.release()
    cv2.destroyAllWindows()

    images_session = [p for p in session_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    images_label = [p for p in label_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"[capture_gallery] Завершено. Сохранено {saved} файлов в {session_dir}")
    print(f"[capture_gallery] Итог файлов в сессии: {len(images_session)}")
    print(f"[capture_gallery] Итог файлов по букве '{args.label}': {len(images_label)}")
    print(f"[capture_gallery] Manifest: {manifest_jsonl}")
    print(f"[capture_gallery] Manifest: {manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
