#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import load_config
from app.embedding import DinoEmbedder
from app.retrieval import GalleryEntry, build_faiss_index, save_metadata, scan_gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from backend/gallery")
    parser.add_argument("--config", default=str(BACKEND / "config.yaml"))
    parser.add_argument("--gallery", default=str(BACKEND / "gallery"))
    parser.add_argument("--artifacts", default=str(BACKEND / "artifacts"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча для эмбеддингов")
    parser.add_argument(
        "--imbalance-ratio-warn",
        type=float,
        default=0.5,
        help="Порог предупреждения по дисбалансу: min_count/max_count",
    )
    return parser.parse_args()


def print_dataset_report(entries: list[GalleryEntry], ratio_warn: float) -> None:
    counts = Counter(e.letter for e in entries)
    labels = sorted(counts.keys())

    print("[build_index] Отчёт по датасету:")
    for label in labels:
        print(f"  - {label}: {counts[label]}")

    min_label, min_count = min(counts.items(), key=lambda kv: kv[1])
    max_label, max_count = max(counts.items(), key=lambda kv: kv[1])
    ratio = float(min_count / max_count) if max_count > 0 else 1.0
    print(
        "[build_index] Статистика: "
        f"classes={len(counts)}, total={len(entries)}, "
        f"min={min_label}:{min_count}, max={max_label}:{max_count}, "
        f"ratio={ratio:.3f}"
    )

    if ratio < ratio_warn:
        print(
            "[build_index][warn] Сильный дисбаланс классов: "
            f"ratio={ratio:.3f} < {ratio_warn:.3f}. "
            "Рекомендуется доснять кадры для малочисленных букв."
        )
    if min_count < 20:
        print(
            "[build_index][warn] Для некоторых букв меньше 20 эталонов. "
            "Это может ухудшать стабильность распознавания."
        )


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

    print_dataset_report(entries, float(args.imbalance_ratio_warn))
    print(f"[build_index] Найдено {len(entries)} эталонов. Загружаю DINOv2 ({cfg.embedding_model})...")
    embedder = DinoEmbedder(model_name=cfg.embedding_model, device=cfg.device)
    batch_size = max(1, int(args.batch_size))

    vectors: list[np.ndarray] = []
    metadata = []
    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        chunk_paths = [entry.path for entry in chunk]
        chunk_vectors = embedder.embed_many_paths(chunk_paths, batch_size=batch_size)
        if chunk_vectors.shape[0] != len(chunk):
            raise RuntimeError(
                "Embedding batch size mismatch: "
                f"got={chunk_vectors.shape[0]} expected={len(chunk)}"
            )
        vectors.append(chunk_vectors)

        for entry in chunk:
            metadata.append(
                {
                    "path": entry.path,
                    "letter": entry.letter,
                    "row_id": len(metadata),
                }
            )
        done = min(len(entries), start + len(chunk))
        print(f"[build_index] embeddings {done}/{len(entries)} (batch={len(chunk)})")

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
