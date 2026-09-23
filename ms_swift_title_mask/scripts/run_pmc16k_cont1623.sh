#!/usr/bin/env bash
# pmc-fullce-16k 步数对照续训: ckpt-1160 → 1623 步(与 mix@1623 等步数)。
# 目的: "同数据、纯步数"格——463 步是否复现 mix 的多页表格坍塌/复读上升。
# 新变量(登记): lr 调度按 max_steps=1623 重铺(cosine),resume 后 lr 处于 1160/1623 位置,
#               非原 1160-cosine 的延长;数据为第二遍遍历(mix 为单遍)。
# 用法: bash run_pmc16k_cont1623.sh   (GPU1 空闲时)
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1  # torch 2.11 weights_only 默认拒绝老 ckpt 的 rng_state(numpy),本机自训 ckpt 可信
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POSTTRAIN_ROOT=${POSTTRAIN_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
UOCR_ROOT=${UOCR_ROOT:-/home/jovyan/hyx/uocr-ms-swift-title-mask}
MODEL_OUTPUT_ROOT=${MODEL_OUTPUT_ROOT:-$UOCR_ROOT/output}
CKPT=${CKPT:-$MODEL_OUTPUT_ROOT/pmc-fullce-16k/v0-20260917-004031/v0-20260917-004305/checkpoint-1160}
[[ -d $CKPT ]] || { echo "FATAL: ckpt-1160 not found: $CKPT" >&2; exit 1; }
export POSTTRAIN_ROOT UOCR_ROOT MODEL_OUTPUT_ROOT
export OUTPUT_DIR=${OUTPUT_DIR:-$MODEL_OUTPUT_ROOT/pmc-fullce-16k-cont1623/v0-$(date +%Y%m%d-%H%M%S)}
export MAX_STEPS=1623
exec bash "$SCRIPT_DIR/run_pmc_fullce.sh" 16k --resume_from_checkpoint "$CKPT"
