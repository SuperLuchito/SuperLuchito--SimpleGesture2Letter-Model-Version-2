#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subset labels/splits from prepared splits.json")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--labels-file", default="", help="Optional file with labels (one per line)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.splits)
    if not in_path.exists():
        raise FileNotFoundError(f"splits file not found: {in_path}")

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    train = list(payload.get("train", []))
    val = list(payload.get("val", []))
    test = list(payload.get("test", []))

    selected: list[str]
    if args.labels_file:
        labels_path = Path(args.labels_file)
        selected = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        counts = Counter(item.get("label", "") for item in train)
        selected = [label for label, _ in counts.most_common(max(1, int(args.top_k))) if label]

    selected_set = set(selected)
    train_subset = [item for item in train if item.get("label") in selected_set]
    val_subset = [item for item in val if item.get("label") in selected_set]
    test_subset = [item for item in test if item.get("label") in selected_set]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "labels.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    (out_dir / "subset_splits.json").write_text(
        json.dumps(
            {
                "meta": {
                    "source": str(in_path),
                    "labels": len(selected),
                    "train_samples": len(train_subset),
                    "val_samples": len(val_subset),
                    "test_samples": len(test_subset),
                },
                "labels": selected,
                "train": train_subset,
                "val": val_subset,
                "test": test_subset,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[build_subset] done")
    print(f"  out_dir={out_dir}")
    print(
        f"  labels={len(selected)} "
        f"train={len(train_subset)} val={len(val_subset)} test={len(test_subset)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
