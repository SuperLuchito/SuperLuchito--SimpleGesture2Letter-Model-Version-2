#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import load_config
from app.embedding import DinoEmbedder
from app.retrieval import build_faiss_index, save_metadata, scan_gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from backend/gallery")
    parser.add_argument("--config", default=str(BACKEND / "config.yaml"))
    parser.add_argument("--gallery", default=str(BACKEND / "gallery"))
    parser.add_argument("--artifacts", default=str(BACKEND / "artifacts"))
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    gallery_dir = Path(args.gallery)
    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    entries = scan_gallery(
        gallery_dir=gallery_dir,
        none_label_dir=cfg.none_label_dir,
        letters_allowlist=cfg.letters_allowlist,
    )
    if not entries:
        print("[build_index] Нет эталонов. Добавьте изображения в backend/gallery/<БУКВА>/*.jpg")
        return 1

    print(f"[build_index] Найдено {len(entries)} эталонов. Загружаю DINOv2 ({cfg.embedding_model})...")
    embedder = DinoEmbedder(model_name=cfg.embedding_model, device=cfg.device)

    vectors = []
    metadata = []
    for i, entry in enumerate(entries, start=1):
        vec = embedder.embed_path(entry.path)
        vectors.append(vec)
        metadata.append(
            {
                "path": entry.path,
                "letter": entry.letter,
                "row_id": i - 1,
            }
        )
        if i % 10 == 0 or i == len(entries):
            print(f"[build_index] {i}/{len(entries)}")

    matrix = np.concatenate(vectors, axis=0).astype(np.float32)
    index = build_faiss_index(matrix)

    import faiss

    index_path = artifacts_dir / "faiss.index"
    meta_path = artifacts_dir / "meta.json"

    faiss.write_index(index, str(index_path))
    save_metadata(meta_path, metadata)

    print(f"[build_index] Готово: {index_path}")
    print(f"[build_index] Готово: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
