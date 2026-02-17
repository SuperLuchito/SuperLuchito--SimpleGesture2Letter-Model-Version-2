#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PATH_KEYS = ("video_path", "path", "filepath", "sample_path", "video", "video_file")
LABEL_KEYS = ("label", "word", "gloss", "class", "class_name")
SIGNER_KEYS = ("user_id", "signer_id", "signer", "person_id", "subject_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare signer-safe train/val/test split for Slovo")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--path-key", default="")
    parser.add_argument("--label-key", default="")
    parser.add_argument("--signer-key", default="")
    return parser.parse_args()


def pick_key(headers: list[str], candidates: tuple[str, ...], forced: str) -> str:
    if forced:
        if forced not in headers:
            raise ValueError(f"forced key '{forced}' is absent in annotations")
        return forced
    for key in candidates:
        if key in headers:
            return key
    raise ValueError(f"None of keys {candidates} found in annotations headers={headers}")


def main() -> int:
    args = parse_args()
    val_ratio = float(args.val_ratio)
    test_ratio = float(args.test_ratio)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")
    if (val_ratio + test_ratio) >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be < 1")

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = list(reader.fieldnames or [])

    if not rows:
        raise ValueError("annotations file is empty")

    path_key = pick_key(headers, PATH_KEYS, args.path_key)
    label_key = pick_key(headers, LABEL_KEYS, args.label_key)
    signer_key = pick_key(headers, SIGNER_KEYS, args.signer_key)

    signer_to_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        signer = str(row.get(signer_key, "")).strip()
        if not signer:
            signer = "_unknown_signer"
        signer_to_rows.setdefault(signer, []).append(row)

    signer_ids = sorted(signer_to_rows.keys())
    rnd = random.Random(int(args.seed))
    rnd.shuffle(signer_ids)

    total_rows = len(rows)
    target_val = max(1, int(round(total_rows * val_ratio)))
    target_test = int(round(total_rows * test_ratio))
    if test_ratio > 0.0:
        target_test = max(1, target_test)

    test_signers: list[str] = []
    val_signers: list[str] = []
    remaining_signers: list[str] = []
    test_count = 0
    val_count = 0
    for signer in signer_ids:
        signer_samples = len(signer_to_rows[signer])
        if test_count < target_test:
            test_signers.append(signer)
            test_count += signer_samples
            continue
        if val_count < target_val:
            val_signers.append(signer)
            val_count += signer_samples
            continue
        remaining_signers.append(signer)

    test_signers_set = set(test_signers)
    val_signers_set = set(val_signers)
    train_signers_set = set(remaining_signers)

    train_items: list[dict[str, str]] = []
    val_items: list[dict[str, str]] = []
    test_items: list[dict[str, str]] = []

    for row in rows:
        item = {
            "path": str(row.get(path_key, "")).strip(),
            "label": str(row.get(label_key, "")).strip(),
            "signer_id": str(row.get(signer_key, "")).strip() or "_unknown_signer",
        }
        if not item["path"] or not item["label"]:
            continue
        if item["signer_id"] in test_signers_set:
            test_items.append(item)
        elif item["signer_id"] in val_signers_set:
            val_items.append(item)
        else:
            train_items.append(item)

    train_signers = {x["signer_id"] for x in train_items}
    val_signers_actual = {x["signer_id"] for x in val_items}
    test_signers_actual = {x["signer_id"] for x in test_items}

    leakage_train_val = sorted(train_signers.intersection(val_signers_actual))
    leakage_train_test = sorted(train_signers.intersection(test_signers_actual))
    leakage_val_test = sorted(val_signers_actual.intersection(test_signers_actual))
    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise RuntimeError(
            "Signer leakage detected: "
            f"train-val={leakage_train_val[:5]}, "
            f"train-test={leakage_train_test[:5]}, "
            f"val-test={leakage_val_test[:5]}"
        )

    payload = {
        "meta": {
            "annotations": str(ann_path),
            "path_key": path_key,
            "label_key": label_key,
            "signer_key": signer_key,
            "seed": int(args.seed),
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "total_samples": len(train_items) + len(val_items) + len(test_items),
            "train_samples": len(train_items),
            "val_samples": len(val_items),
            "test_samples": len(test_items),
            "train_signers": len(train_signers),
            "val_signers": len(val_signers_actual),
            "test_signers": len(test_signers_actual),
        },
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    train_counts = Counter(item["label"] for item in train_items)
    val_counts = Counter(item["label"] for item in val_items)
    test_counts = Counter(item["label"] for item in test_items)

    print("[prepare_slovo_splits] done")
    print(f"  out={out_path}")
    print(f"  train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(
        "  signers "
        f"train={len(train_signers)} val={len(val_signers_actual)} test={len(test_signers_actual)}"
    )
    print(
        "  labels "
        f"train={len(train_counts)} val={len(val_counts)} test={len(test_counts)}"
    )
    if not train_signers_set:
        print("  [warn] train split has no signer ids (check ratios and source annotations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
