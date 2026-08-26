#!/usr/bin/env bash
set -euo pipefail

if [[ "${UOCR_ALLOW_LEGACY_20260825:-0}" != "1" ]]; then
  echo "refusing legacy 2026-08-25 title-weighted scheduler; set UOCR_ALLOW_LEGACY_20260825=1 to opt in" >&2
  exit 2
fi

DELAY_SECONDS=${1:-10800}
if [[ ! "$DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "delay must be a non-negative integer number of seconds" >&2
  exit 2
fi

BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POSTTRAIN_ROOT=$(cd "$BUNDLE_ROOT/.." && pwd)
HYX_ROOT=$(cd "$POSTTRAIN_ROOT/.." && pwd)
UOCR_ROOT=${UOCR_ROOT:-$HYX_ROOT/uocr-ms-swift-title-mask}
WORK_ROOT=${UOCR_WORK_ROOT:-$UOCR_ROOT/runs/20260823}
FULL_CE_OUTPUT_FRAGMENT=${FULL_CE_OUTPUT_FRAGMENT:-"$WORK_ROOT/outputs/reviewed-full-ce-dual-32k"}
SCHEDULER_PID_FILE=${SCHEDULER_PID_FILE:-}
if [[ -n "$SCHEDULER_PID_FILE" ]]; then
  printf '%s\n' "$$" > "$SCHEDULER_PID_FILE"
fi

FULL_CE_LOG=${FULL_CE_LOG:-"$WORK_ROOT/full_ce_dual_32k.log"}
WEIGHTED_LOG=${WEIGHTED_LOG:-"$WORK_ROOT/title_weighted_dual_32k.log"}
WEIGHTED_OUTPUT=${WEIGHTED_OUTPUT:-"$WORK_ROOT/outputs/reviewed-title-weighted-dual-32k"}

echo "[uocr-scheduler] queued_at=$(date '+%F %T %Z') delay_seconds=$DELAY_SECONDS"
sleep "$DELAY_SECONDS"
echo "[uocr-scheduler] checking full_ce at $(date '+%F %T %Z')"

if ps -eo args= | grep -F -- "$FULL_CE_OUTPUT_FRAGMENT" | grep -v grep >/dev/null; then
  echo "[uocr-scheduler] full_ce is still running; weighted training was not started" >&2
  exit 1
fi
if [[ ! -f "$FULL_CE_LOG" ]] || ! grep -Eq "global_step/max_steps.*443/443|TrainOutput" "$FULL_CE_LOG"; then
  echo "[uocr-scheduler] full_ce completion marker is missing; weighted training was not started" >&2
  exit 1
fi
if [[ -e "$WEIGHTED_OUTPUT" ]]; then
  echo "[uocr-scheduler] refusing to overwrite existing weighted output: $WEIGHTED_OUTPUT" >&2
  exit 1
fi

source "$BUNDLE_ROOT/scripts/hyx_env.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TRAIN_JSONL=${TRAIN_JSONL:-"$WORK_ROOT/length_32k_v1/fit/train.jsonl"}
export VAL_JSONL=${VAL_JSONL:-"$WORK_ROOT/length_32k_v1/fit/validation.jsonl"}
export OUTPUT_DIR="$WEIGHTED_OUTPUT"
export LOSS_MODE=title_weighted
export MAX_LENGTH=${MAX_LENGTH:-32768}
export ATTN_IMPL=${ATTN_IMPL:-sdpa}
export GRAD_ACCUM=${GRAD_ACCUM:-4}

mkdir -p "$(dirname "$WEIGHTED_LOG")"
echo "[uocr-scheduler] starting title_weighted at $(date '+%F %T %Z')"
echo "[uocr-scheduler] log=$WEIGHTED_LOG output=$WEIGHTED_OUTPUT"
exec bash "$BUNDLE_ROOT/scripts/run_dual_h100.sh" >"$WEIGHTED_LOG" 2>&1
