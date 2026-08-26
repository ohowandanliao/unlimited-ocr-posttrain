#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 WORLD_SIZE [extra swift arguments...]" >&2
  exit 2
fi

WORLD_SIZE=$1
shift
BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POSTTRAIN_ROOT=$(cd "$BUNDLE_ROOT/.." && pwd)
HYX_ROOT=$(cd "$POSTTRAIN_ROOT/.." && pwd)
UOCR_ROOT=${UOCR_ROOT:-$HYX_ROOT/uocr-ms-swift-title-mask}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-$UOCR_ROOT/repos/ms-swift-uocr}
UOCR_SHIMS_ROOT=${UOCR_SHIMS_ROOT:-$UOCR_ROOT/runtime/shims}
PYTHON_BIN=${PYTHON_BIN:-$UOCR_ROOT/env/ms-swift-venv/bin/python}
SWIFT_BIN=${SWIFT_BIN:-$UOCR_ROOT/env/ms-swift-venv/bin/swift}

# The venv's console script starts outside the checkout; make both the patched
# ms-swift tree and the optional Apex compatibility shim importable explicitly.
export PYTHONPATH="$UOCR_SHIMS_ROOT:$MS_SWIFT_ROOT:$POSTTRAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL_PATH:?set MODEL_PATH to the Unlimited-OCR model directory or model ID}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to reviewed train.jsonl}"
: "${VAL_JSONL:?set VAL_JSONL to reviewed validation.jsonl}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new experiment directory}"
: "${TRAINING_RECIPE:?set TRAINING_RECIPE to readoc_r0, replay_r1, pmc_s10, trusted_title, or legacy_20260823}"
: "${LOSS_MODE:?set LOSS_MODE explicitly to full_ce, uniform_ce, title_weighted, or title_mask}"

MAX_LENGTH=${MAX_LENGTH:-32768}
MAX_STEPS=${MAX_STEPS:-}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
if [[ "$TRAINING_RECIPE" == "legacy_20260823" ]]; then
  LEARNING_RATE=${LEARNING_RATE:-1e-4}
  WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
  WARMUP_RATIO=${WARMUP_RATIO:-0.05}
else
  LEARNING_RATE=${LEARNING_RATE:-2e-5}
  WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
  WARMUP_RATIO=${WARMUP_RATIO:-0.03}
fi
LOGGING_STEPS=${LOGGING_STEPS:-5}
EVAL_STEPS=${EVAL_STEPS:-50}
SAVE_STEPS=${SAVE_STEPS:-50}
if [[ "$TRAINING_RECIPE" == "legacy_20260823" ]]; then
  SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
else
  SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
fi
SEED=${SEED:-0}
ATTN_IMPL=${ATTN_IMPL:-sdpa}

case "$WORLD_SIZE" in
  1) GRAD_ACCUM=${GRAD_ACCUM:-8} ;;
  2) GRAD_ACCUM=${GRAD_ACCUM:-4} ;;
  *) echo "only WORLD_SIZE=1 or 2 is supported by this reviewed recipe" >&2; exit 2 ;;
esac

case "$TRAINING_RECIPE" in
  readoc_r0|replay_r1|pmc_s10|trusted_title|legacy_20260823) ;;
  *) echo "invalid TRAINING_RECIPE: $TRAINING_RECIPE" >&2; exit 2 ;;
esac

case "$LOSS_MODE" in
  title_mask|title_weighted|uniform_ce|full_ce) ;;
  *) echo "LOSS_MODE must be full_ce, uniform_ce, title_weighted, or title_mask" >&2; exit 2 ;;
esac
case "$TRAINING_RECIPE" in
  readoc_r0|replay_r1|pmc_s10)
    if [[ "$LOSS_MODE" != "full_ce" ]]; then
      echo "$TRAINING_RECIPE only permits LOSS_MODE=full_ce" >&2
      exit 2
    fi
    ;;
  trusted_title)
    if [[ "$LOSS_MODE" == "title_mask" ]]; then
      echo "trusted_title permits full_ce, uniform_ce, or title_weighted; title_mask is historical" >&2
      exit 2
    fi
    ;;
esac
if [[ "$TRAINING_RECIPE" == "legacy_20260823" && "${UOCR_ALLOW_LEGACY_20260823:-0}" != "1" ]]; then
  echo "legacy_20260823 is historical; set UOCR_ALLOW_LEGACY_20260823=1 to reproduce it" >&2
  exit 2
fi

if [[ "$TRAINING_RECIPE" == "legacy_20260823" ]]; then
  TITLE_WEIGHT=${TITLE_WEIGHT:-2.0}
else
  TITLE_WEIGHT=${TITLE_WEIGHT:-1.5}
fi
EOS_WEIGHT=${EOS_WEIGHT:-1.0}
export UOCR_TITLE_WEIGHT=$TITLE_WEIGHT
export UOCR_EOS_WEIGHT=$EOS_WEIGHT

case "$ATTN_IMPL" in
  sdpa|eager) ;;
  *) echo "ATTN_IMPL must be sdpa or eager" >&2; exit 2 ;;
esac

case "$MAX_LENGTH" in
  4096|8192|16384|24576|32768) ;;
  *) echo "MAX_LENGTH must be one of 4096, 8192, 16384, 24576, 32768" >&2; exit 2 ;;
esac

if [[ ! -d "$MS_SWIFT_ROOT/swift" ]]; then
  echo "MS_SWIFT_ROOT is not an ms-swift checkout: $MS_SWIFT_ROOT" >&2
  exit 2
fi
if [[ "$PYTHON_BIN" == */* && ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$PYTHON_BIN" != */* ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "PYTHON_BIN was not found on PATH: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$SWIFT_BIN" == */* && ! -x "$SWIFT_BIN" ]]; then
  echo "SWIFT_BIN is not executable: $SWIFT_BIN" >&2
  exit 2
fi
if [[ "$SWIFT_BIN" != */* ]] && ! command -v "$SWIFT_BIN" >/dev/null 2>&1; then
  echo "SWIFT_BIN was not found on PATH: $SWIFT_BIN" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_JSONL" || ! -f "$VAL_JSONL" ]]; then
  echo "TRAIN_JSONL and VAL_JSONL must both exist" >&2
  exit 2
fi

"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/validate_reviewed_jsonl.py" "$TRAIN_JSONL" "$VAL_JSONL"
RECIPE_REPORT=${RECIPE_REPORT:-$OUTPUT_DIR.recipe_audit.json}
RECIPE_ARGS=(
  --recipe "$TRAINING_RECIPE"
  --report "$RECIPE_REPORT"
)
if [[ -n "${MIN_CJK_CHAR_SHARE:-}" ]]; then
  RECIPE_ARGS+=(--min-cjk-char-share "$MIN_CJK_CHAR_SHARE")
fi
if [[ -n "${MAX_PMC_CHAR_SHARE:-}" ]]; then
  RECIPE_ARGS+=(--max-pmc-char-share "$MAX_PMC_CHAR_SHARE")
fi
"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/validate_training_recipe.py" \
  "${RECIPE_ARGS[@]}" "$TRAIN_JSONL" "$VAL_JSONL"

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
  --attn_impl "$ATTN_IMPL"
  --use_logits_to_keep true
  --packing false
  --padding_free false
  --truncation_strategy None
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
  --weight_decay "$WEIGHT_DECAY"
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
if [[ "$LOSS_MODE" == title_weighted ]]; then
  WEIGHTED_LOSS_TYPE=uocr_title_weighted_token_mean
  if [[ "$TRAINING_RECIPE" == "legacy_20260823" ]]; then
    WEIGHTED_LOSS_TYPE=uocr_title_weighted_mean
  fi
  SWIFT_ARGS+=(
    --template unlimited_ocr_title_weighted
    --loss_scale default
    --is_binary_loss_scale false
    --loss_type "$WEIGHTED_LOSS_TYPE"
  )
fi
if [[ "$LOSS_MODE" == uniform_ce ]]; then
  SWIFT_ARGS+=(
    --template unlimited_ocr_uniform_ce
    --loss_scale default
    --is_binary_loss_scale false
    --loss_type uocr_uniform_token_mean
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

echo "[uocr] recipe=$TRAINING_RECIPE mode=$LOSS_MODE world_size=$WORLD_SIZE grad_accum=$GRAD_ACCUM max_length=$MAX_LENGTH"
echo "[uocr] lr=$LEARNING_RATE weight_decay=$WEIGHT_DECAY warmup_ratio=$WARMUP_RATIO recipe_report=$RECIPE_REPORT"
if [[ "$LOSS_MODE" == "title_weighted" ]]; then
  if [[ "$WEIGHTED_LOSS_TYPE" == "uocr_title_weighted_mean" ]]; then
    WEIGHTED_DENOMINATOR=active_weight_sum_per_micro_batch
  else
    WEIGHTED_DENOMINATOR=native_accumulated_tokens
  fi
  echo "[uocr] title_weight=$TITLE_WEIGHT eos_weight=$EOS_WEIGHT loss_type=$WEIGHTED_LOSS_TYPE denominator=$WEIGHTED_DENOMINATOR"
fi
echo "[uocr] ms-swift=$MS_SWIFT_ROOT output=$OUTPUT_DIR"
exec "$SWIFT_BIN" "${SWIFT_ARGS[@]}"
