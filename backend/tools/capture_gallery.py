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
from app.embedding import DinoEmbedder
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
    "mirror",
    "dhash",
    "dino_cosine_max",
    "ssim_best",
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


def dhash(image_bgr: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = vec_a.astype(np.float32, copy=False)
    b = vec_b.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def similarity_gray(image_bgr: np.ndarray, size: int = 160) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_a = cv2.GaussianBlur(gray_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gray_b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(gray_a * gray_a, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(gray_b * gray_b, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, (11, 11), 1.5) - mu_ab

    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


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


def quality_check(
    crop_bgr: np.ndarray,
    *,
    min_sharpness: float,
    max_bright_ratio: float,
    bbox_area: float,
    min_capture_bbox_area: float,
    require_bbox_area: bool,
) -> tuple[bool, float, float, str]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    bright_ratio = float(np.mean(gray >= 245))

    if sharpness < min_sharpness:
        return False, sharpness, bright_ratio, f"blur ({sharpness:.1f} < {min_sharpness:.1f})"
    if bright_ratio > max_bright_ratio:
        return False, sharpness, bright_ratio, f"overexposed ({bright_ratio:.2f} > {max_bright_ratio:.2f})"
    if require_bbox_area and bbox_area < min_capture_bbox_area:
        return False, sharpness, bright_ratio, f"small_bbox ({bbox_area:.4f} < {min_capture_bbox_area:.4f})"
    return True, sharpness, bright_ratio, ""


def build_operator_hints(
    *,
    detector_enabled: bool,
    hand_present: bool,
    bbox_area: float,
    bbox_norm: tuple[float, float, float, float],
    frame_bgr: np.ndarray,
    crop_bgr: np.ndarray | None,
    min_capture_bbox_area: float,
    min_sharpness: float,
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

            if crop_bgr is not None:
                crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                sharpness_preview = float(cv2.Laplacian(crop_gray, cv2.CV_32F).var())
                if sharpness_preview < (min_sharpness * 0.75):
                    hints.append("стабилизируйте руку")

    if not hints:
        return ["ок"]
    return list(dict.fromkeys(hints))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture gallery samples from webcam")
    parser.add_argument("--label", required=True, help="Буква или _none")
    parser.add_argument("--session-id", default="", help="ID сессии (по умолчанию авто session_YYYYMMDD...)")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("--auto", action="store_true", help="Автосохранение по таймеру")
    parser.add_argument("--mirror", action="store_true", help="Зеркалить кадры (preview + сохранение)")
    parser.add_argument("--config", default=str(BACKEND / "config.yaml"))
    parser.add_argument("--gallery", default=str(BACKEND / "gallery"))
    parser.add_argument("--min-sharpness", type=float, default=45.0, help="Порог резкости (Laplacian var)")
    parser.add_argument("--max-bright-ratio", type=float, default=0.35, help="Макс. доля пересвеченных пикселей")
    parser.add_argument("--min-capture-bbox-area", type=float, default=0.06, help="Мин. площадь bbox руки")
    parser.add_argument(
        "--dedup-hamming-th",
        type=int,
        default=2,
        help="Порог dHash для быстрого отсева почти идентичных кадров",
    )
    parser.add_argument(
        "--dedup-cosine-th",
        type=float,
        default=0.995,
        help="Основной порог дедупа по DINOv2 cosine (высокий = меньше агрессии)",
    )
    parser.add_argument(
        "--dedup-cosine-margin",
        type=float,
        default=0.004,
        help="Ширина пограничной зоны для SSIM tie-break",
    )
    parser.add_argument("--dedup-ssim-th", type=float, default=0.985, help="Порог SSIM дедупликации")
    parser.add_argument("--dedup-ref-limit", type=int, default=80, help="Сколько последних кадров сравнивать")
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
    saved = 0
    dedup_hashes: list[int] = []
    dedup_embeddings: list[np.ndarray] = []
    dedup_refs: list[np.ndarray] = []
    dedup_ref_limit = max(1, int(args.dedup_ref_limit))
    dedup_cosine_margin = max(0.0, float(args.dedup_cosine_margin))
    dedup_embedder: DinoEmbedder | None = None
    try:
        dedup_embedder = DinoEmbedder(model_name=cfg.embedding_model, device=cfg.device)
        print(f"[capture_gallery] DINO дедуп модель: {cfg.embedding_model} ({cfg.device})")
    except Exception as exc:
        print(
            "[capture_gallery][warn] DINOv2 для дедупликации недоступен, "
            f"использую fallback dHash+SSIM. Причина: {exc}"
        )

    print("[capture_gallery] Горячие клавиши: s=save, a=auto on/off, q=quit")
    print(f"[capture_gallery] Session: {session_id}")
    print(f"[capture_gallery] Output dir: {session_dir}")
    print(f"[capture_gallery] Mirror: {bool(args.mirror)}")
    print(
        "[capture_gallery] QC: "
        f"min_sharpness={args.min_sharpness}, "
        f"max_bright_ratio={args.max_bright_ratio}, "
        f"min_capture_bbox_area={min_capture_bbox_area:.4f}"
    )
    print(
        "[capture_gallery] Dedup: "
        f"dHash<={args.dedup_hamming_th} -> "
        f"DINOv2 cosine>={args.dedup_cosine_th:.4f} "
        f"(margin={dedup_cosine_margin:.4f}) -> "
        f"SSIM>={args.dedup_ssim_th:.4f} tie-break"
    )

    def try_save(crop_bgr: np.ndarray | None, *, capture_mode: str, ts_ms: int, bbox_area: float) -> bool:
        nonlocal saved, last_save_ms, dedup_embedder

        if crop_bgr is None:
            print(f"[capture_gallery] skip ({capture_mode}): no_crop")
            return False

        ok, sharpness, bright_ratio, reason = quality_check(
            crop_bgr,
            min_sharpness=float(args.min_sharpness),
            max_bright_ratio=float(args.max_bright_ratio),
            bbox_area=float(bbox_area),
            min_capture_bbox_area=min_capture_bbox_area,
            require_bbox_area=(detector is not None),
        )
        if not ok:
            print(f"[capture_gallery] skip ({capture_mode}): {reason}")
            return False

        # Русский комментарий: дедуп работает в 3 шага.
        # 1) Быстрый фильтр dHash, 2) основной критерий DINOv2 cosine,
        # 3) SSIM включается только в пограничной зоне около порога cosine.
        current_hash = dhash(crop_bgr)
        for prev_hash in dedup_hashes[-dedup_ref_limit:]:
            dist = hamming_distance(current_hash, prev_hash)
            if dist <= args.dedup_hamming_th:
                print(f"[capture_gallery] skip ({capture_mode}): duplicate_dhash (distance={dist})")
                return False

        current_ref = similarity_gray(crop_bgr)
        cosine_max = -1.0
        ssim_best = -1.0
        dedup_strategy = "dhash+dino+ssim_tiebreak"
        current_embedding: np.ndarray | None = None

        recent_refs = dedup_refs[-dedup_ref_limit:]
        recent_embeddings = dedup_embeddings[-dedup_ref_limit:]

        if dedup_embedder is not None and recent_embeddings:
            try:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                current_embedding = dedup_embedder.embed_rgb(crop_rgb)[0].astype(np.float32)

                best_idx = -1
                for idx, prev_embedding in enumerate(recent_embeddings):
                    score = cosine_similarity(current_embedding, prev_embedding)
                    if score > cosine_max:
                        cosine_max = score
                        best_idx = idx

                strict_cosine_th = float(args.dedup_cosine_th) + dedup_cosine_margin
                if cosine_max >= strict_cosine_th:
                    print(
                        f"[capture_gallery] skip ({capture_mode}): "
                        f"duplicate_dino_strict (cos={cosine_max:.4f})"
                    )
                    return False

                if cosine_max >= float(args.dedup_cosine_th) and best_idx >= 0 and best_idx < len(recent_refs):
                    ssim_best = ssim(current_ref, recent_refs[best_idx])
                    if ssim_best >= args.dedup_ssim_th:
                        print(
                            f"[capture_gallery] skip ({capture_mode}): "
                            f"duplicate_dino_ssim (cos={cosine_max:.4f}, ssim={ssim_best:.4f})"
                        )
                        return False
            except Exception as exc:
                dedup_embedder = None
                dedup_strategy = "dhash+ssim_fallback"
                print(
                    "[capture_gallery][warn] DINO-дедуп отключен во время съёмки, "
                    f"перехожу на fallback dHash+SSIM. Причина: {exc}"
                )
                for prev_ref in recent_refs:
                    score = ssim(current_ref, prev_ref)
                    if score > ssim_best:
                        ssim_best = score
                    if score >= args.dedup_ssim_th:
                        print(f"[capture_gallery] skip ({capture_mode}): duplicate_ssim (score={score:.4f})")
                        return False
        elif recent_refs:
            dedup_strategy = "dhash+ssim_fallback"
            for prev_ref in recent_refs:
                score = ssim(current_ref, prev_ref)
                if score > ssim_best:
                    ssim_best = score
                if score >= args.dedup_ssim_th:
                    print(f"[capture_gallery] skip ({capture_mode}): duplicate_ssim (score={score:.4f})")
                    return False

        saved += 1
        now_utc = datetime.now(timezone.utc)
        ts_wall_ms = int(now_utc.timestamp() * 1000)
        filename = session_dir / f"{ts_wall_ms}_{saved:03d}.jpg"
        cv2.imwrite(str(filename), crop_bgr)
        last_save_ms = ts_ms

        dedup_hashes.append(current_hash)
        if dedup_embedder is not None:
            if current_embedding is None:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                current_embedding = dedup_embedder.embed_rgb(crop_rgb)[0].astype(np.float32)
            dedup_embeddings.append(current_embedding)
        dedup_refs.append(current_ref)

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
            "mirror": bool(args.mirror),
            "dhash": format(current_hash, "016x"),
            "dino_cosine_max": round(cosine_max, 6) if cosine_max >= 0.0 else "",
            "ssim_best": round(ssim_best, 6) if ssim_best >= 0.0 else "",
            "dedup_strategy": dedup_strategy,
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
            crop_bgr=crop,
            min_capture_bbox_area=min_capture_bbox_area,
            min_sharpness=float(args.min_sharpness),
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

        text_items.append(
            (
                f"буква={args.label} saved={saved}/{args.count} auto={auto_mode}",
                (12, 98 if len(hints) <= 1 else 142),
                TEXT_WHITE,
                21,
            )
        )

        frame = draw_text_items(frame, text_items)

        if auto_mode and crop is not None and (ts_ms - last_save_ms) >= args.interval_ms:
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
