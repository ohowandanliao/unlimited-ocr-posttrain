#!/usr/bin/env bash
set -euo pipefail

if [[ "${UOCR_ALLOW_LEGACY_20260825:-0}" != "1" ]]; then
  echo "refusing legacy 2026-08-25 inference service; set UOCR_ALLOW_LEGACY_20260825=1 to opt in" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/hyx_env.sh"

FULL_CE_ADAPTER_PATH=${FULL_CE_ADAPTER_PATH:-$UOCR_WORK_ROOT/outputs/reviewed-full-ce-dual-32k/v0-20260824-233410/checkpoint-443}
TITLE_WEIGHTED_ADAPTER_PATH=${TITLE_WEIGHTED_ADAPTER_PATH:-$UOCR_WORK_ROOT/outputs/reviewed-title-weighted-dual-32k/v0-20260825-031657/checkpoint-443}
SERVICE_HOST=${SERVICE_HOST:-0.0.0.0}
SERVICE_PORT=${SERVICE_PORT:-18080}
SERVICE_DEVICE=${SERVICE_DEVICE:-cuda:0}

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
  "$PYTHON_BIN" "$SCRIPT_DIR/serve_unlimited_ocr.py" \
  --base-model "$MODEL_PATH" \
  --full-ce-adapter "$FULL_CE_ADAPTER_PATH" \
  --title-weighted-adapter "$TITLE_WEIGHTED_ADAPTER_PATH" \
  --device "$SERVICE_DEVICE" \
  --host "$SERVICE_HOST" \
  --port "$SERVICE_PORT"
