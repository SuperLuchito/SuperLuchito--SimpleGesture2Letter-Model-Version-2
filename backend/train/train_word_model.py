#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training entrypoint scaffold for word-level model")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits_path = Path(args.splits)
    if not splits_path.exists():
        raise FileNotFoundError(f"splits not found: {splits_path}")

    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    train = payload.get("train", [])
    val = payload.get("val", [])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[train_word_model] scaffold")
    print(f"  train_samples={len(train)}")
    print(f"  val_samples={len(val)}")
    print(f"  epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")
    print("  TODO: подключить baseline Slovo модель и dataloader клипов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
