#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import load_config
from app.embedding import DinoEmbedder
from app.retrieval import GalleryEntry, build_faiss_index, l2_normalize, scan_gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sanity split on gallery (train-like vs val-like)")
    parser.add_argument("--config", default=str(BACKEND / "config.yaml"))
    parser.add_argument("--gallery", default=str(BACKEND / "gallery"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-val-per-class", type=int, default=1)
    parser.add_argument("--min-train-per-class", type=int, default=2)
    parser.add_argument("--report", default=str(BACKEND / "artifacts" / "sanity_eval.json"))
    parser.add_argument("--split-file", default=str(BACKEND / "artifacts" / "sanity_split.json"))
    return parser.parse_args()


def make_sanity_split(
    entries: list[GalleryEntry],
    *,
    val_ratio: float,
    seed: int,
    min_val_per_class: int,
    min_train_per_class: int,
) -> tuple[list[GalleryEntry], list[GalleryEntry], dict[str, str]]:
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError("val_ratio must be in (0, 1).")
    if min_val_per_class < 1:
        raise ValueError("min_val_per_class must be >= 1.")
    if min_train_per_class < 1:
        raise ValueError("min_train_per_class must be >= 1.")

    by_label: dict[str, list[GalleryEntry]] = defaultdict(list)
    for entry in entries:
        by_label[entry.letter].append(entry)

    train_entries: list[GalleryEntry] = []
    val_entries: list[GalleryEntry] = []
    skipped_labels: dict[str, str] = {}

    for label_idx, label in enumerate(sorted(by_label.keys())):
        items = sorted(by_label[label], key=lambda e: e.path)
        rnd = random.Random(int(seed) + ((label_idx + 1) * 7919))
        rnd.shuffle(items)

        count = len(items)
        required = int(min_train_per_class + min_val_per_class)
        if count < required:
            skipped_labels[label] = f"insufficient_samples:{count}<{required}"
            train_entries.extend(items)
            continue

        proposed_val = int(round(count * float(val_ratio)))
        val_count = max(int(min_val_per_class), proposed_val)
        val_count = min(val_count, count - int(min_train_per_class))

        if val_count < int(min_val_per_class):
            skipped_labels[label] = "cannot_allocate_validation"
            train_entries.extend(items)
            continue

        val_part = items[:val_count]
        train_part = items[val_count:]
        if len(train_part) < int(min_train_per_class):
            skipped_labels[label] = "cannot_allocate_train"
            train_entries.extend(items)
            continue

        train_entries.extend(train_part)
        val_entries.extend(val_part)

    return train_entries, val_entries, skipped_labels


def evaluate_sanity_split(
    *,
    train_entries: list[GalleryEntry],
    val_entries: list[GalleryEntry],
    embedder: DinoEmbedder,
    k: int,
    batch_size: int,
) -> dict[str, Any]:
    if not train_entries:
        raise ValueError("Train split is empty; cannot evaluate.")

    train_paths = [entry.path for entry in train_entries]
    train_labels = [entry.letter for entry in train_entries]
    train_vectors = embedder.embed_many_paths(train_paths, batch_size=batch_size)
    if train_vectors.shape[0] != len(train_entries):
        raise RuntimeError(
            f"Train embedding size mismatch: got={train_vectors.shape[0]} expected={len(train_entries)}"
        )

    index = build_faiss_index(train_vectors)
    search_k = min(max(1, int(k)), len(train_entries))

    if not val_entries:
        return {
            "summary": {
                "train_samples": len(train_entries),
                "val_samples": 0,
                "k": search_k,
                "top1_acc": 0.0,
                "topk_acc": 0.0,
                "top1_correct": 0,
                "topk_correct": 0,
            },
            "per_class": {},
        }

    val_paths = [entry.path for entry in val_entries]
    val_vectors = embedder.embed_many_paths(val_paths, batch_size=batch_size)
    if val_vectors.shape[0] != len(val_entries):
        raise RuntimeError(
            f"Val embedding size mismatch: got={val_vectors.shape[0]} expected={len(val_entries)}"
        )

    query = l2_normalize(val_vectors.astype(np.float32))
    scores, indices = index.search(query, search_k)

    top1_correct = 0
    topk_correct = 0
    per_class_stats: dict[str, dict[str, int]] = {}

    for row_idx, target in enumerate(val_entries):
        predicted_labels = [train_labels[idx] for idx in indices[row_idx] if idx >= 0]
        is_top1 = bool(predicted_labels) and predicted_labels[0] == target.letter
        is_topk = target.letter in predicted_labels

        if is_top1:
            top1_correct += 1
        if is_topk:
            topk_correct += 1

        bucket = per_class_stats.setdefault(
            target.letter,
            {"samples": 0, "top1_correct": 0, "topk_correct": 0},
        )
        bucket["samples"] += 1
        bucket["top1_correct"] += int(is_top1)
        bucket["topk_correct"] += int(is_topk)

    per_class: dict[str, dict[str, float | int]] = {}
    for label, stats in sorted(per_class_stats.items()):
        samples = max(1, stats["samples"])
        per_class[label] = {
            "samples": stats["samples"],
            "top1_correct": stats["top1_correct"],
            "topk_correct": stats["topk_correct"],
            "top1_acc": float(stats["top1_correct"] / samples),
            "topk_acc": float(stats["topk_correct"] / samples),
        }

    val_count = len(val_entries)
    return {
        "summary": {
            "train_samples": len(train_entries),
            "val_samples": val_count,
            "k": search_k,
            "top1_acc": float(top1_correct / max(1, val_count)),
            "topk_acc": float(topk_correct / max(1, val_count)),
            "top1_correct": top1_correct,
            "topk_correct": topk_correct,
        },
        "per_class": per_class,
    }


def run_sanity_evaluation(
    *,
    entries: list[GalleryEntry],
    embedding_model: str,
    device: str,
    val_ratio: float,
    seed: int,
    min_val_per_class: int,
    min_train_per_class: int,
    k: int,
    batch_size: int,
    report_path: Path,
    split_path: Path,
    embedder: DinoEmbedder | None = None,
) -> dict[str, Any]:
    train_entries, val_entries, skipped_labels = make_sanity_split(
        entries,
        val_ratio=val_ratio,
        seed=seed,
        min_val_per_class=min_val_per_class,
        min_train_per_class=min_train_per_class,
    )

    runtime_embedder = embedder or DinoEmbedder(model_name=embedding_model, device=device)
    metrics = evaluate_sanity_split(
        train_entries=train_entries,
        val_entries=val_entries,
        embedder=runtime_embedder,
        k=k,
        batch_size=batch_size,
    )

    split_payload = {
        "settings": {
            "val_ratio": float(val_ratio),
            "seed": int(seed),
            "min_val_per_class": int(min_val_per_class),
            "min_train_per_class": int(min_train_per_class),
        },
        "counts": {
            "total": len(entries),
            "train": len(train_entries),
            "val": len(val_entries),
        },
        "skipped_labels": skipped_labels,
        "train": [{"path": e.path, "letter": e.letter} for e in train_entries],
        "val": [{"path": e.path, "letter": e.letter} for e in val_entries],
    }

    report_payload = {
        "settings": {
            "embedding_model": embedding_model,
            "device": device,
            "val_ratio": float(val_ratio),
            "seed": int(seed),
            "min_val_per_class": int(min_val_per_class),
            "min_train_per_class": int(min_train_per_class),
            "k": int(k),
            "batch_size": int(batch_size),
        },
        "summary": metrics["summary"],
        "per_class": metrics["per_class"],
        "skipped_labels": skipped_labels,
        "paths": {
            "split_file": str(split_path),
            "report_file": str(report_path),
        },
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_payload


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    entries = scan_gallery(
        gallery_dir=Path(args.gallery),
        none_label_dir=cfg.none_label_dir,
        letters_allowlist=cfg.letters_allowlist,
    )
    if not entries:
        print("[sanity_eval] Нет эталонов для оценки.")
        return 1

    report = run_sanity_evaluation(
        entries=entries,
        embedding_model=cfg.embedding_model,
        device=cfg.device,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        min_val_per_class=int(args.min_val_per_class),
        min_train_per_class=int(args.min_train_per_class),
        k=int(args.k),
        batch_size=max(1, int(args.batch_size)),
        report_path=Path(args.report),
        split_path=Path(args.split_file),
    )

    summary = report["summary"]
    print("[sanity_eval] Готово:")
    print(f"  train={summary['train_samples']} val={summary['val_samples']} k={summary['k']}")
    print(f"  top1_acc={summary['top1_acc']:.4f} topk_acc={summary['topk_acc']:.4f}")
    print(f"  report={report['paths']['report_file']}")
    print(f"  split={report['paths']['split_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
