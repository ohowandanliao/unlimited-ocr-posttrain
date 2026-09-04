#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/hyx_env.sh"

: "${MODEL_PATH:?set MODEL_PATH to the Unlimited-OCR base model directory}"
: "${FULL_CE_ADAPTER_PATH:?set FULL_CE_ADAPTER_PATH to the full_ce adapter directory}"
: "${TITLE_WEIGHTED_ADAPTER_PATH:?set TITLE_WEIGHTED_ADAPTER_PATH to the title_weighted adapter directory}"
SERVICE_HOST=${SERVICE_HOST:-127.0.0.1}
SERVICE_PORT=${SERVICE_PORT:-18080}
SERVICE_DEVICE=${SERVICE_DEVICE:-cuda:0}
for path_name in MODEL_PATH FULL_CE_ADAPTER_PATH TITLE_WEIGHTED_ADAPTER_PATH; do
  if [[ ! -d "${!path_name}" ]]; then
    echo "$path_name must name an existing directory: ${!path_name}" >&2
    exit 2
  fi
done

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
  "$PYTHON_BIN" "$SCRIPT_DIR/serve_unlimited_ocr.py" \
  --base-model "$MODEL_PATH" \
  --full-ce-adapter "$FULL_CE_ADAPTER_PATH" \
  --title-weighted-adapter "$TITLE_WEIGHTED_ADAPTER_PATH" \
  --device "$SERVICE_DEVICE" \
  --host "$SERVICE_HOST" \
  --port "$SERVICE_PORT"
