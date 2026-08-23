#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 WORLD_SIZE [extra swift arguments...]" >&2
  exit 2
fi

WORLD_SIZE=$1
shift
BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-$(dirname "$BUNDLE_ROOT")}
PYTHON_BIN=${PYTHON_BIN:-python}

: "${MODEL_PATH:?set MODEL_PATH to the Unlimited-OCR model directory or model ID}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to reviewed train.jsonl}"
: "${VAL_JSONL:?set VAL_JSONL to reviewed validation.jsonl}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new experiment directory}"

LOSS_MODE=${LOSS_MODE:-title_mask}
MAX_LENGTH=${MAX_LENGTH:-32768}
MAX_STEPS=${MAX_STEPS:-}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
LOGGING_STEPS=${LOGGING_STEPS:-5}
EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-50}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
SEED=${SEED:-0}

case "$WORLD_SIZE" in
  1) GRAD_ACCUM=${GRAD_ACCUM:-8} ;;
  2) GRAD_ACCUM=${GRAD_ACCUM:-4} ;;
  *) echo "only WORLD_SIZE=1 or 2 is supported by this reviewed recipe" >&2; exit 2 ;;
esac

case "$LOSS_MODE" in
  title_mask|full_ce) ;;
  *) echo "LOSS_MODE must be title_mask or full_ce" >&2; exit 2 ;;
esac

case "$MAX_LENGTH" in
  4096|8192|16384|24576|32768) ;;
  *) echo "MAX_LENGTH must be one of 4096, 8192, 16384, 24576, 32768" >&2; exit 2 ;;
esac

if [[ ! -d "$MS_SWIFT_ROOT/swift" ]]; then
  echo "MS_SWIFT_ROOT is not an ms-swift checkout: $MS_SWIFT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_JSONL" || ! -f "$VAL_JSONL" ]]; then
  echo "TRAIN_JSONL and VAL_JSONL must both exist" >&2
  exit 2
fi

"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/validate_reviewed_jsonl.py" "$TRAIN_JSONL" "$VAL_JSONL"

export CROP_MODE=false
export IMAGE_SIZE=1024
export BASE_SIZE=1024
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

TARGET_REGEX='^model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)|mlp\.shared_experts\.(gate_proj|up_proj|down_proj))$'
PLUGIN_PATH="$BUNDLE_ROOT/plugin/uocr_title_mask.py"

SWIFT_ARGS=(
  sft
  --model "$MODEL_PATH"
  --external_plugins "$PLUGIN_PATH"
  --callbacks uocr_lora_target_guard
  --dataset "$TRAIN_JSONL"
  --val_dataset "$VAL_JSONL"
  --split_dataset_ratio 0
  --load_from_cache_file false
  --template unlimited_ocr
  --tuner_type lora
  --target_regex "$TARGET_REGEX"
  --lora_rank 16
  --lora_alpha 32
  --lora_dropout 0.05
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --torch_dtype bfloat16
  --attn_impl eager
  --packing false
  --padding_free false
  --truncation_strategy raise
  --max_length "$MAX_LENGTH"
  --sequence_parallel_size 1
  --average_tokens_across_devices false
  --use_liger_kernel false
  --enable_dft_loss false
  --router_aux_loss_coef 0
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 1
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --gradient_checkpointing true
  --gradient_checkpointing_kwargs '{"use_reentrant":false}'
  --learning_rate "$LEARNING_RATE"
  --weight_decay 0.1
  --adam_beta2 0.95
  --lr_scheduler_type cosine
  --warmup_ratio "$WARMUP_RATIO"
  --max_grad_norm 1.0
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --logging_steps "$LOGGING_STEPS"
  --eval_strategy steps
  --eval_steps "$EVAL_STEPS"
  --save_strategy steps
  --save_steps "$SAVE_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --dataloader_num_workers 4
  --seed "$SEED"
  --data_seed "$SEED"
  --report_to tensorboard
  --output_dir "$OUTPUT_DIR"
)

if [[ "$LOSS_MODE" == title_mask ]]; then
  SWIFT_ARGS+=(
    --template unlimited_ocr_title_mask
    --loss_scale default
    --is_binary_loss_scale false
    --loss_type uocr_title_active_mean
  )
fi
if [[ -n "$MAX_STEPS" ]]; then
  SWIFT_ARGS+=(--max_steps "$MAX_STEPS")
fi
if [[ "$WORLD_SIZE" == 2 ]]; then
  SWIFT_ARGS+=(--ddp_find_unused_parameters false)
fi
SWIFT_ARGS+=("$@")

cd "$MS_SWIFT_ROOT"
if [[ "$WORLD_SIZE" == 2 ]]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
  export NPROC_PER_NODE=2
else
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  unset NPROC_PER_NODE || true
fi

echo "[uocr] mode=$LOSS_MODE world_size=$WORLD_SIZE grad_accum=$GRAD_ACCUM max_length=$MAX_LENGTH"
echo "[uocr] ms-swift=$MS_SWIFT_ROOT output=$OUTPUT_DIR"
exec swift "${SWIFT_ARGS[@]}"
