#!/usr/bin/env bash
set -euo pipefail

if [[ "${UOCR_ALLOW_LEGACY_20260825:-0}" != "1" ]]; then
  echo "refusing legacy 2026-08-25 parallel evaluation; set UOCR_ALLOW_LEGACY_20260825=1 to opt in" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POSTTRAIN_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
HYX_ROOT=$(cd "$POSTTRAIN_ROOT/.." && pwd)
UOCR_ROOT=${UOCR_ROOT:-$HYX_ROOT/uocr-ms-swift-title-mask}
PYTHON_BIN=${PYTHON_BIN:-$UOCR_ROOT/env/ms-swift-venv/bin/python}
PYTHON_SCRIPT="$POSTTRAIN_ROOT/ms_swift_title_mask/scripts/probe_evaluation_pdfs_parallel.py"
PDF_ROOT=/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/source
GT_ROOT=/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth
RUN_ROOT=$UOCR_ROOT/runs/20260823
OLD_DIR=$RUN_ROOT/inference_eval_20260825_all129
FINAL_DIR=$RUN_ROOT/inference_eval_20260825_all129_parallel_v3
PARALLEL_LOG=$RUN_ROOT/inference_eval_20260825_all129_parallel_v3.log

if [[ ! -s "$OLD_DIR/responses.jsonl" ]]; then
  echo "partial result file is missing: $OLD_DIR/responses.jsonl" >&2
  exit 2
fi
mkdir -p "$FINAL_DIR" "$FINAL_DIR/pages"
if [[ ! -s "$FINAL_DIR/responses.jsonl" ]]; then
  cp "$OLD_DIR/responses.jsonl" "$FINAL_DIR/responses.jsonl"
  cp "$OLD_DIR/evaluation_names.txt" "$FINAL_DIR/evaluation_names.txt"
  find "$OLD_DIR" -maxdepth 1 -type f -name '*.pdf__unlimited-ocr-*.md' -exec cp {} "$FINAL_DIR/" \;
fi

for endpoint in http://127.0.0.1:18080/infer http://127.0.0.1:18081/infer; do
  curl --fail --silent --show-error --max-time 30 "${endpoint%/infer}/health" >/dev/null
done

cd "$POSTTRAIN_ROOT"
exec "$PYTHON_BIN" "$PYTHON_SCRIPT" \
  --pdf-root "$PDF_ROOT" \
  --gt-root "$GT_ROOT" \
  --output-dir "$FINAL_DIR" \
  --all-files \
  --url http://127.0.0.1:18080/infer \
  --url http://127.0.0.1:18081/infer \
  --model unlimited-ocr-base \
  --model unlimited-ocr-full-ce \
  --model unlimited-ocr-title-weighted \
  --dpi 144 --max-length 32768 --timeout 3600 --attempts 1 \
  --note "2026-08-25 dynamic two-GPU queue; resumed from prior partial output; errors are not counted as completion"
