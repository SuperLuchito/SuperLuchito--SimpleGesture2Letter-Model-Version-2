#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import load_config, merge_config_values
from app.embedding import DinoEmbedder
from app.retrieval import GalleryIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate SIM_NONE by _none samples")
    parser.add_argument("--config", default=str(BACKEND / "config.yaml"))
    parser.add_argument("--gallery", default=str(BACKEND / "gallery"))
    parser.add_argument("--artifacts", default=str(BACKEND / "artifacts"))
    parser.add_argument("--none-label", default="_none")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    try:
        index = GalleryIndex.load(Path(args.artifacts) / "faiss.index", Path(args.artifacts) / "meta.json")
    except Exception as exc:
        print(f"[calibrate] Индекс недоступен: {exc}")
        print("[calibrate] Сначала запустите: python backend/tools/build_index.py")
        return 1

    if index.size == 0:
        print("[calibrate] Индекс пуст. Сначала запустите build_index.py")
        return 1

    none_dir = Path(args.gallery) / args.none_label
    if not none_dir.exists():
        print(f"[calibrate] Директория с NONE-кадрами не найдена: {none_dir}")
        return 1

    images = [p for p in sorted(none_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not images:
        print("[calibrate] Нет кадров в _none")
        return 1

    print(f"[calibrate] Загружаю DINOv2 ({cfg.embedding_model})")
    embedder = DinoEmbedder(model_name=cfg.embedding_model, device=cfg.device)

    max_scores: list[float] = []
    margins: list[float] = []

    for i, img_path in enumerate(images, start=1):
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        vec = embedder.embed_rgb(rgb)
        hits = index.search(vec, k=2)
        if not hits:
            continue
        max_scores.append(float(hits[0].score))
        if len(hits) > 1:
            margins.append(float(hits[0].score - hits[1].score))
        if i % 10 == 0 or i == len(images):
            print(f"[calibrate] {i}/{len(images)}")

    if not max_scores:
        print("[calibrate] Не удалось вычислить similarity для NONE-кадров")
        return 1

    max_arr = np.asarray(max_scores)
    sim_none_p95 = float(np.percentile(max_arr, 95))
    sim_none_p99 = float(np.percentile(max_arr, 99))
    # Keep a conservative upper bound; p99 can be too strict on small/noisy NONE sets.
    sim_none_recommended = float(np.clip(sim_none_p95, 0.45, 0.65))
    rec_sim_vlm = float(min(0.95, sim_none_recommended + 0.12))
    rec_margin = float(np.percentile(np.asarray(margins), 75)) if margins else cfg.margin_th

    merge_config_values(args.config, {"sim_none": round(sim_none_recommended, 4)})

    print("[calibrate] Обновлено в config.yaml:")
    print(f"  sim_none = {sim_none_recommended:.4f} (bounded p95)")
    print(f"  reference p95={sim_none_p95:.4f}, p99={sim_none_p99:.4f}")
    print("[calibrate] Рекомендации (не перезаписаны автоматически):")
    print(f"  sim_vlm_th ~= {rec_sim_vlm:.4f}")
    print(f"  margin_th ~= {rec_margin:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
