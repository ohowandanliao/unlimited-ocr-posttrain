#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "usage: source $0" >&2
  exit 2
fi

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POSTTRAIN_ROOT=${POSTTRAIN_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
HYX_ROOT=$(cd "$POSTTRAIN_ROOT/.." && pwd)
UOCR_ROOT=${UOCR_ROOT:-$HYX_ROOT/uocr-ms-swift-title-mask}
UOCR_VENV=${UOCR_VENV:-$UOCR_ROOT/env/ms-swift-venv}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-$UOCR_ROOT/repos/ms-swift-uocr}
UOCR_SHIMS_ROOT=${UOCR_SHIMS_ROOT:-$POSTTRAIN_ROOT/ms_swift_title_mask/shims}
# UOCR_ROOT is the run workspace (data/output/evaluation/logs/service live under it);
# set it explicitly to relocate artifacts. All runnable code lives in this repo.
# Compatibility alias for older callers. New artifacts are split by purpose
# under DATA_ROOT, MODEL_OUTPUT_ROOT, EVALUATION_ROOT, LOG_ROOT and SERVICE_ROOT.
UOCR_WORK_ROOT=${UOCR_WORK_ROOT:-$UOCR_ROOT}
WORK_ROOT=${WORK_ROOT:-$UOCR_WORK_ROOT}
DATA_ROOT=${DATA_ROOT:-$UOCR_ROOT/data}
REVIEW_DATA_ROOT=${REVIEW_DATA_ROOT:-$DATA_ROOT/review}
DATA_RECIPE_ROOT=${DATA_RECIPE_ROOT:-$DATA_ROOT/recipes}
MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT:-$UOCR_ROOT/output}
EVALUATION_ROOT=${EVALUATION_ROOT:-$UOCR_ROOT/evaluation}
EVAL_ROOT=${EVAL_ROOT:-$EVALUATION_ROOT}
LOG_ROOT=${LOG_ROOT:-$UOCR_ROOT/logs}
SERVICE_ROOT=${SERVICE_ROOT:-$UOCR_ROOT/service}
MODEL_PATH=${MODEL_PATH:-}
ASSET_ROOT=${ASSET_ROOT:-$POSTTRAIN_ROOT/ms_swift_title_mask/data_assets}
PAGE_ROOT=${PAGE_ROOT:-}
LENGTH_ROOT=${LENGTH_ROOT:-}
TRAIN_JSONL=${TRAIN_JSONL:-}
VAL_JSONL=${VAL_JSONL:-}
TEST_JSONL=${TEST_JSONL:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
PYTHON_BIN=${PYTHON_BIN:-$UOCR_VENV/bin/python}
SWIFT_BIN=${SWIFT_BIN:-$UOCR_VENV/bin/swift}
TRAINING_RECIPE=${TRAINING_RECIPE:-}
LOSS_MODE=${LOSS_MODE:-}
MAX_LENGTH=${MAX_LENGTH:-32768}
ATTN_IMPL=${ATTN_IMPL:-sdpa}
# Leave this unset unless the caller overrides it. _run_train.sh selects 8 for
# one GPU and 4 for two GPUs so both launchers keep the same global batch size.
GRAD_ACCUM=${GRAD_ACCUM:-}

export POSTTRAIN_ROOT HYX_ROOT UOCR_ROOT UOCR_VENV MS_SWIFT_ROOT UOCR_SHIMS_ROOT
export UOCR_WORK_ROOT WORK_ROOT DATA_ROOT REVIEW_DATA_ROOT DATA_RECIPE_ROOT MODEL_OUTPUT_ROOT
export EVALUATION_ROOT EVAL_ROOT LOG_ROOT SERVICE_ROOT
export MODEL_PATH ASSET_ROOT PAGE_ROOT LENGTH_ROOT
export TRAIN_JSONL VAL_JSONL TEST_JSONL OUTPUT_DIR PYTHON_BIN SWIFT_BIN
export TRAINING_RECIPE LOSS_MODE MAX_LENGTH ATTN_IMPL GRAD_ACCUM
export PATH="$UOCR_VENV/bin:$PATH"
export PYTHONPATH="$UOCR_SHIMS_ROOT:$MS_SWIFT_ROOT:$POSTTRAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[uocr-env] root=$UOCR_ROOT"
echo "[uocr-env] compatibility_work=$UOCR_WORK_ROOT"
echo "[uocr-env] data=$DATA_ROOT review_data=$REVIEW_DATA_ROOT"
echo "[uocr-env] recipes=$DATA_RECIPE_ROOT"
echo "[uocr-env] output=$MODEL_OUTPUT_ROOT"
echo "[uocr-env] evaluation=$EVALUATION_ROOT"
echo "[uocr-env] ms-swift=$MS_SWIFT_ROOT"
echo "[uocr-env] MODEL_PATH/TRAIN_JSONL/VAL_JSONL/OUTPUT_DIR/TRAINING_RECIPE/LOSS_MODE must be set per experiment"
