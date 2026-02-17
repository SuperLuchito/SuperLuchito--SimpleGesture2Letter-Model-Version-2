#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZIP_PATH="$ROOT_DIR/backend/data/slovo/slovo.zip"
EXTRACT_DIR="$ROOT_DIR/backend/data/slovo/extracted"
SPLITS_PATH="$ROOT_DIR/backend/data/slovo/splits.json"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "[finalize_slovo_dataset] zip not found: $ZIP_PATH"
  exit 1
fi

echo "[finalize_slovo_dataset] extracting $ZIP_PATH ..."
mkdir -p "$EXTRACT_DIR"
unzip -o "$ZIP_PATH" -d "$EXTRACT_DIR" >/tmp/finalize_slovo_unzip.log

ANN_PATH="$(find "$EXTRACT_DIR" -type f -iname 'annotations.csv' | head -n 1)"
if [[ -z "$ANN_PATH" ]]; then
  echo "[finalize_slovo_dataset] annotations.csv not found after extraction"
  exit 1
fi

ANN_DIR="$(cd "$(dirname "$ANN_PATH")" && pwd)"
if [[ -d "$ANN_DIR/videos" ]]; then
  PATH_PREFIX="$ANN_DIR/videos"
else
  PATH_PREFIX="$ANN_DIR"
fi

echo "[finalize_slovo_dataset] building signer-safe split ..."
"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/backend/scripts/prepare_slovo_splits.py" \
  --annotations "$ANN_PATH" \
  --out "$SPLITS_PATH" \
  --path-prefix "$PATH_PREFIX" \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42

echo "[finalize_slovo_dataset] done"
echo "  annotations=$ANN_PATH"
echo "  splits=$SPLITS_PATH"

