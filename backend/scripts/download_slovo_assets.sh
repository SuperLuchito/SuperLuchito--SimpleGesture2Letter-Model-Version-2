#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACTS_DIR="$ROOT_DIR/backend/artifacts"
SLOVO_DIR="$ROOT_DIR/backend/data/slovo"

MODEL_URL="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/models/mvit/onnx/mvit32-2.onnx"
DATASET_URL="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip"

WITH_DATASET=false
if [[ "${1:-}" == "--with-dataset" ]]; then
  WITH_DATASET=true
fi

mkdir -p "$ARTIFACTS_DIR" "$SLOVO_DIR"

echo "[download_slovo_assets] downloading baseline ONNX model..."
curl -L --fail --retry 3 -C - -o "$ARTIFACTS_DIR/slovo_word_model.onnx" "$MODEL_URL"
ls -lh "$ARTIFACTS_DIR/slovo_word_model.onnx"

if [[ "$WITH_DATASET" == "true" ]]; then
  echo "[download_slovo_assets] downloading Slovo dataset zip (16GB)..."
  curl -L --fail --retry 3 -C - -o "$SLOVO_DIR/slovo.zip" "$DATASET_URL"
  ls -lh "$SLOVO_DIR/slovo.zip"
else
  echo "[download_slovo_assets] dataset download skipped (use --with-dataset to enable)."
fi

