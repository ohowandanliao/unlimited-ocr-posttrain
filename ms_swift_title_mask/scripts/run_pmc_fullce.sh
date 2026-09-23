#!/usr/bin/env bash
# run_pmc_fullce.sh — PMC v2.2.1 Full-CE 单卡训练入口（READoc-1706 配方复刻）
#
# 用法：
#   run_pmc_fullce.sh <16k|32k> [extra swift arguments...]
#
# 约定：
#   - 默认 GPU 1（GPU 0 常驻其他服务；如需换卡 CUDA_VISIBLE_DEVICES=0 run_pmc_fullce.sh ...）
#   - recipe=pmc_fullce（PMC-only / full_document gate），loss=full_ce，LoRA 超参沿用
#     _run_train.sh 写死的 rank16/alpha32（与 readoc-view-16k-full-ce 参考实验一致）
#   - 数据 = ~/hyx/dataset/pmc-fullce-data/pmc-fullce-<budget>/{train,validation}.jsonl
#   - 输出 = $UOCR_ROOT/output/pmc-fullce-<budget>/v0-<时间戳>（已存在会拒绝，防覆盖）
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POSTTRAIN_ROOT=${POSTTRAIN_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
UOCR_ROOT=${UOCR_ROOT:-/home/jovyan/hyx/uocr-ms-swift-title-mask}
PMC_DATA_ROOT=${PMC_DATA_ROOT:-/home/jovyan/hyx/dataset/pmc-fullce-data}
MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT:-$UOCR_ROOT/output}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <16k|32k> [extra swift arguments...]" >&2
  exit 2
fi
BUDGET=$1
shift
case "$BUDGET" in
  16k) MAX_LENGTH_DEFAULT=16384 ;;
  32k) MAX_LENGTH_DEFAULT=32768 ;;
  *) echo "budget must be 16k or 32k" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export MODEL_PATH=${MODEL_PATH:-/home/jovyan/hyx/models/Unlimited-OCR}
export TRAIN_JSONL=${TRAIN_JSONL:-$PMC_DATA_ROOT/pmc-fullce-$BUDGET/train.jsonl}
export VAL_JSONL=${VAL_JSONL:-$PMC_DATA_ROOT/pmc-fullce-$BUDGET/validation.jsonl}
export UOCR_ROOT PMC_DATA_ROOT MODEL_OUTPUT_ROOT POSTTRAIN_ROOT
export OUTPUT_DIR=${OUTPUT_DIR:-$MODEL_OUTPUT_ROOT/pmc-fullce-$BUDGET/v0-$(date +%Y%m%d-%H%M%S)}
export TRAINING_RECIPE=${TRAINING_RECIPE:-pmc_fullce}
export LOSS_MODE=full_ce
export MAX_LENGTH=${MAX_LENGTH:-$MAX_LENGTH_DEFAULT}
# 2026-09-17 放宽（用户拍板）：PMC val 847 篇单轮 eval ~13.7min，50 步一评吃掉约一半墙钟；
# eval 放宽到 200；save 谨慎起见维持 50（checkpoint 46MB/个，断点损失窗口 ≤50 步 ≈ 14min）。
export EVAL_STEPS=${EVAL_STEPS:-200}
export SAVE_STEPS=${SAVE_STEPS:-50}

echo "[pmc-fullce] budget=$BUDGET max_length=$MAX_LENGTH gpu=$CUDA_VISIBLE_DEVICES eval_steps=$EVAL_STEPS save_steps=$SAVE_STEPS"
echo "[pmc-fullce] train=$TRAIN_JSONL"
echo "[pmc-fullce] val=$VAL_JSONL"
echo "[pmc-fullce] output=$OUTPUT_DIR"
exec bash "$SCRIPT_DIR/_run_train.sh" 1 "$@"
