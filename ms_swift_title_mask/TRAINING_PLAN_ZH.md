# Unlimited-OCR 下一轮训练执行方案

版本：2026-08-26。当前只允许按 `E0 -> R0 -> R1 -> S10 -> T -> V` 顺序推进。旧 natural-union、
title+EOS-only 和旧 1:2 title-weighted 不得作为默认入口。

## 1. E0 前置条件

启动训练前，先冻结文档级去重的 benchmark，并对 Base 生成完整结果。至少分桶：中/英文、单/多页、扫描、
表格、公式、双栏、重复表头和长文档。每条记录固定 PDF SHA、GT SHA、页数、渲染参数、prompt、生成参数、
模型/checkpoint 与停止原因。

阻断条件：请求完整率不是 100%，中文主指标相对 Base 回退超过 1 个百分点，严重重复或 cap-hit 超过 1%，
纯 Markdown 协议率低于 98%，或任一主要桶退化超过 2 点。当前快速 analyzer 不能替代 NED/TEDS/CDM 正式指标。

## 2. 环境

所有路径显式设置，工作目录不得位于 Git 仓库或原始数据目录内：

```bash
export POSTTRAIN_ROOT=/absolute/path/to/unlimited-ocr-posttrain
export UOCR_ROOT=/absolute/path/to/uocr-posttrain-runtime
export UOCR_WORK_ROOT=/absolute/path/to/uocr-posttrain-runs
export MS_SWIFT_ROOT=/absolute/path/to/ms-swift
export UOCR_SHIMS_ROOT=/absolute/path/to/runtime-shims
export PYTHON_BIN=/absolute/path/to/venv/bin/python
export SWIFT_BIN=/absolute/path/to/venv/bin/swift
export MODEL_PATH=/absolute/path/to/Unlimited-OCR

cd "$POSTTRAIN_ROOT"
source ms_swift_title_mask/scripts/hyx_env.sh
bash ms_swift_title_mask/scripts/run_preflight.sh
```

ms-swift 必须是测试 commit `1a1ba3ee86488af323ef9b64ca3d34edee90ab11`，环境使用
`transformers==4.46.3`。更新框架后必须重新 review compatibility diff，不能直接绕过。

## 3. R0：clean READoc-only

输入 pool 是仓库外的 READoc full split 目录；原始 split 为 `1411/71/70`。其中
`readoc_github_160428551__full` 含两个无法从 PDF 像素推断的仓库相对图片路径，R0 用显式 exclusion 排除并写入
recipe report，不静默改 target。输出 split 为 `1410/71/70`，随后按正式 tokenizer 生成 32K fit/overflow：

```bash
export READOC_POOL=/absolute/path/to/readoc_full
export R0_RECIPE_ROOT="$UOCR_WORK_ROOT/recipes/readoc_r0_v1"
export R0_LENGTH_ROOT="$UOCR_WORK_ROOT/length/readoc_r0_32k_v1"

"$PYTHON_BIN" ms_swift_title_mask/scripts/build_recipe_mix.py \
  --pool "readoc=$READOC_POOL" \
  --exclude-row-id readoc_github_160428551__full \
  --output-dir "$R0_RECIPE_ROOT"

"$PYTHON_BIN" ms_swift_title_mask/scripts/build_length_buckets.py \
  --input-dir "$R0_RECIPE_ROOT" \
  --model "$MODEL_PATH" \
  --output-dir "$R0_LENGTH_ROOT" \
  --max-length 32768
```

`report.json` 必须记录 tokenizer 文件 SHA、input JSONL SHA、各 pool 的 target token 总数和全部 overflow；
overflow 不截断、不 packing、不混入 fit。

先做 1-step smoke：

```bash
export TRAIN_JSONL="$R0_LENGTH_ROOT/smoke/train.jsonl"
export VAL_JSONL="$R0_LENGTH_ROOT/smoke/validation.jsonl"
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/readoc_r0_smoke"
export TRAINING_RECIPE=readoc_r0
export LOSS_MODE=full_ce
export MAX_STEPS=1
bash ms_swift_title_mask/scripts/run_dual_h100.sh
```

正式 R0 使用 fit 数据，LR 只做 `2e-5` 与 `5e-5` 两臂；WD `0.01`、warmup `0.03`、最多 1 epoch。先将
`MAX_STEPS` 设为 `report.json` 中 train fit 对应总 optimizer steps 的约 25%，生成 benchmark 后再决定是否续跑：

```bash
export TRAIN_JSONL="$R0_LENGTH_ROOT/fit/train.jsonl"
export VAL_JSONL="$R0_LENGTH_ROOT/fit/validation.jsonl"
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/readoc_r0_lr2e-5"
export TRAINING_RECIPE=readoc_r0
export LOSS_MODE=full_ce
export LEARNING_RATE=2e-5
export MAX_STEPS=<约四分之一 epoch 的整数>
bash ms_swift_title_mask/scripts/run_dual_h100.sh
```

checkpoint 以 Base retention gate 和任务指标选择，不以 eval loss 单独选择。

## 4. R1：中文/原生能力 replay

仅现有 READoc/PMC 不足以构造 R1：READoc 没有中文文档，PMC 中文极少。新增 replay 必须与最终 129-PDF test
及 8 个失败样本做文档级去重，并覆盖扫描、表格、公式、双栏和中文标准/报告。禁止把 test GT 回灌训练。

初始 mix 用字符数做确定性 25% 上限代理；完成长度分桶后，必须在 `report.json` 中按正式 tokenizer 核对 replay
target token 占比为 20%-30%，不满足则重建 mix：

```bash
"$PYTHON_BIN" ms_swift_title_mask/scripts/build_recipe_mix.py \
  --pool "readoc=$READOC_POOL" \
  --pool "replay=/absolute/path/to/chinese_replay" \
  --cap-target-char-share replay=0.25 \
  --output-dir "$UOCR_WORK_ROOT/recipes/replay_r1_v1"
```

随后重复 R0 的长度分桶、smoke 和阶段训练，改为 `TRAINING_RECIPE=replay_r1`。recipe gate 默认要求 CJK 字符
占比至少 15%，且拒绝任何 PMC。成功标准是中文不低于 Base、READoc 回退不超过 1 点、Markdown 协议率至少 98%。

## 5. S10：PMC 重新准入

PMC pool 必须先移除缺失资源链接、XML image、坏 LaTeX、粘连 metadata、空页和 image-only table，且每行具有
独立 `content_review_status=CONTENT_ACCEPTED`。同一 PMCID 只能选择 full/window/single 一种形态；只有存在真实
续段/表格关系的 window/full 才能标 merge。

S10 以通过 R1 的 recipe 为 base，PMC 使用字符代理 10% 上限；训练前 gate 的硬上限是 12%：

```bash
"$PYTHON_BIN" ms_swift_title_mask/scripts/build_recipe_mix.py \
  --pool "base=$UOCR_WORK_ROOT/recipes/replay_r1_v1" \
  --pool "pmc=/absolute/path/to/content_accepted_pmc" \
  --cap-target-char-share pmc=0.10 \
  --require-content-status pmc=CONTENT_ACCEPTED \
  --output-dir "$UOCR_WORK_ROOT/recipes/pmc_s10_v1"
```

长度分桶后再次按 tokenizer target token 检查实际 PMC 比例。运行时使用 `TRAINING_RECIPE=pmc_s10`、
`LOSS_MODE=full_ce`。跨页桶不改善或任一 retention 桶回退超过 1 点时，不进入 S20。

## 6. T：可信标题消融

该阶段只接受 `title_review_status=HUMAN_ACCEPTED` 的行；规则级 `SILVER_ACCEPTED` 不够。若含 PMC，仍需满足
S10 的正文状态、污染和比例 gate。三臂使用完全相同的数据顺序、初始 adapter、长度、prompt 和超参：

```bash
export TRAINING_RECIPE=trusted_title
export LOSS_MODE=full_ce
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/trusted_title_native"
bash ms_swift_title_mask/scripts/run_dual_h100.sh

export LOSS_MODE=uniform_ce
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/trusted_title_uniform"
bash ms_swift_title_mask/scripts/run_dual_h100.sh

export LOSS_MODE=title_weighted
export TITLE_WEIGHT=1.5
export EOS_WEIGHT=1.0
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/trusted_title_w1.5"
bash ms_swift_title_mask/scripts/run_dual_h100.sh
```

先跑 10-step native `full_ce` 与 `uniform_ce` parity；loss/gradient 不一致时停止。之后依次尝试标题权重
`1.25/1.5/2.0`，EOS 权重另做单变量实验。标题 F1/层级/顺序至少提升 2 点、正文 NED/TEDS 回退小于 0.5 点，
该方案才保留。

## 7. V 与历史复现

视觉侧先做无需训练的单页 `single_base` / `single_gundam` A/B。没有证明小字、扫描、公式和表格桶稳定提升前，
不训练 crop adapter；stock forward 不支持 `multi_gundam`。

旧快照只能显式运行：

```bash
export UOCR_ALLOW_LEGACY_20260823=1
export TRAINING_RECIPE=legacy_20260823
export LOSS_MODE=title_mask  # 或旧 full_ce/title_weighted 复现
```

2026-08-25 的 service/evaluation/scheduler shell 另要求 `UOCR_ALLOW_LEGACY_20260825=1`。这些入口只用于复核旧结果，
不得复用为下一轮 benchmark。
