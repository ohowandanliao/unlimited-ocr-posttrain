# ms-swift Unlimited-OCR 标题训练执行方案

版本：2026-08-23；测试基线：ms-swift `1a1ba3ee86488af323ef9b64ca3d34edee90ab11`。

## 1. 只保留两组实验

| 实验 | template / loss | 用途 |
|---|---|---|
| `reviewed-full-ce` | 原生 `unlimited_ocr` / 原生完整 CE | 必跑 baseline，完整 assistant Markdown 都算 loss |
| `reviewed-title-mask` | `unlimited_ocr_title_mask` / `uocr_title_active_mean` | 完整 Markdown teacher forcing，只让 ATX heading token 和 EOS 算 loss |

两组必须使用同一批 `HUMAN_ACCEPTED` 与可追溯 `SILVER_ACCEPTED` 文档、同一顺序、split、prompt、图片、LoRA、长度和优化器参数；provenance 不混淆。
loss 数值不能横向比较，最终比较生成结果中的标题检测、文本、层级、顺序，以及正文是否回退。

正式数据是三个池的 natural union：READoc full（1,552/15,290 页/21,400 headings）、PMC full
（1,486/26,690 页/48,726 headings）和 paired PMC strict-single（1,486 行/6,011 headings）。
每个 READoc full 一行、每个 PMC full 一行、同 PMC 再一条 strict-single，PMC full/single 必须同 split；
train 预计 1,411+1,331+1,331=4,073 行，strict-single share 32.7%。不混入 `train_short`，不加 MinerU 输入，
不跑旧 S10/S20，不做 full parameter。
title-mask 是监督目标消融，不是长序列省显存方案：正文仍进入 teacher forcing，attention、logits 和 R-SWA
长度都不变。

### 1.1 四种视觉模式与本轮选择

| 模式 | stock Unlimited-OCR / ms-swift | 本轮处理 |
|---|---|---|
| `single_gundam` | 支持单页 dynamic crop | 当前不训；需要时另开 crop 进程和独立实验 |
| `single_base` | 支持单页 no-crop | PMC strict-single 使用 |
| `multi_base` | 支持多页 no-crop | READoc full、PMC full 使用 |
| `multi_gundam` | 不支持逐页 crop forward | 不实现 |

当前只使用一套进程级视觉策略 `base/no-crop 1024`，但数据同时包含单页和多页。训练脚本统一设置
`CROP_MODE=false, IMAGE_SIZE=1024, BASE_SIZE=1024`，所以 `single_base` 与 `multi_base` 可以自然混合。
crop 环境变量不能逐样本切换，因此不把 `single_gundam` 混进本轮 A/B；也不通过修改模型 forward 实现
`multi_gundam`。

## 2. ms-swift LoRA 粒度与推荐

ms-swift 支持三类定位方式：`target_modules` 后缀列表、`target_regex` 模块正则，以及面向裸
`nn.Parameter` 的 `target_parameters`。Unlimited-OCR 是 MoE，默认 `all-linear` 会把大量 routed expert
也纳入，参数多且每 token 只激活少数 expert，不适合当前小规模标题数据。

本方案固定使用 `target_regex`：

```regex
^model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)|mlp\.shared_experts\.(gate_proj|up_proj|down_proj))$
```

在当前 Unlimited-OCR 上应精确命中：

| 模块组 | 数量 |
|---|---:|
| attention projection | 48 |
| dense MLP | 3 |
| shared-expert MLP | 33 |
| routed expert | 0 |
| 合计 | 84 |

推荐参数：rank 16、alpha 32、dropout 0.05、LoRA LR `1e-4`。约 3.98M trainable parameters，既能调整
输出结构和层级，又避免改 64 路 routed experts。ViT、projector、router、embedding、lm_head 和 base weights
保持冻结。插件 callback 会在 step 1 前核验上述 84/48/3/33/0；命名或匹配发生漂移时直接失败。

attention-only LoRA（48 个模块）只用于故障定位，不进入正式 A/B；`all-linear`、routed-expert LoRA、
DoRA、RS-LoRA、LoRA-GA 和全参均不进入当前方案。

## 3. 输入和输出冻结

训练行统一是 ms-swift messages schema：

```json
{"id":"readoc_x","channel":"title_reviewed","messages":[{"role":"user","content":"<image>Multi page merge."},{"role":"assistant","content":"# Title\n\n## Section\n\nBody...\n"}],"images":["/abs/page_0000.png","/abs/page_0001.png"],"meta":{"source":"READoc-arxiv","doc_id":"x","split":"train","title_review_status":"HUMAN_ACCEPTED","title_review_id":"...","title_review_version":"...","title_target_sha256":"...","title_heading_count":2}}
```

约束：

- 一个 user + 一个 assistant；多图 prompt 中只写一个 `<image>`，ms-swift 会按图片数展开。
- 多页 prompt 固定为 `"<image>Multi page merge."`；若文档天然只有一页，则用
  `"<image>document parsing."`。prompt 必须与图片数一致，不能因为来源名称硬套多页 prompt。
- PMC full 使用完整多页 `Multi page merge.`；paired PMC strict-single 使用 `document parsing.`，但 output 仍是完整 Markdown。
- 官方新 README 的 `Multi page parsing.` 不在这次 A/B 中迁移；以后迁移必须另建数据版本。
- output 是完整连续 Markdown，不是标题列表，不插 `<PAGE>`，末尾固定一个 `\n`。
- 图片路径写服务器绝对路径且按页序排列。
- `channel` 必须是 `title_reviewed`；插件不信任会被 dataset preprocessor 丢弃的任意 `meta` 路由字段。

## 4. Provenance 数据闸门

`HUMAN_ACCEPTED` 表示 reviewer 已检查该文档中全部 ATX 标题的文本、层级和顺序。
`SILVER_ACCEPTED` 表示规则冻结、完整可追溯且 source/target SHA 与 PDF 证据齐全。两者都可训练，provenance 不混淆。

```json
{"id":"readoc_x","status":"SILVER_ACCEPTED","split":"train","review_version":"readoc-title-rules-v1","source_sha256":"64位小写sha256","assistant_sha256":"64位小写sha256","title_heading_count":12}
```

| 数据池 | 文档/行 | 页 | headings | 当前准入 |
|---|---:|---|---|---|
| READoc full | 1,552 | 15,290 | 21,400 | `SILVER_ACCEPTED` 可训 |
| PMC full | 1,486 | 26,690 | 48,726 | `SILVER_ACCEPTED` 可训 |
| PMC strict-single | 1,486 paired rows | - | 6,011 | 与 PMC full 同 split，可训 |

随包快照与 SHA-256 见 `data_assets/STATUS_ZH.md`。

### 4.1 三池 natural union

三个池按相同 doc split 生成 `train/validation/test.jsonl` 后，做确定性 natural union：

```bash
export UOCR_WORK_ROOT=/absolute/path/to/uocr-work-20260823
python ms_swift_title_mask/scripts/build_reviewed_mix.py \
  --readoc-dir "$UOCR_WORK_ROOT/readoc_silver_v1/readoc_full" \
  --pmc-full-dir "$UOCR_WORK_ROOT/pmc_silver_v2/pmc_full" \
  --pmc-single-dir "$UOCR_WORK_ROOT/pmc_silver_v2/pmc_single" \
  --output-dir "$UOCR_WORK_ROOT/final_mix_v1"
```

脚本要求 READoc full 和 PMC full 各一行，并为每个 PMC full 配对一条 strict-single；三者按 family/source/doc_id
检查跨 split 泄漏，允许同 PMCID 的 full/single 在同 split 配对。不重采样、不把 `train_short` 混入；实际行数、
图片数和 target 字符数写入 `mix_report.json`。随 ZIP 已有冻结 `final_mix_v1`，通常不需要在目标服务器重建；
只有从原始 READoc/PMC 数据重新生成时才运行本节，完整顺序见 Git 分支的
`docs/ms_swift_from_source_2026-08-23.md`。

### 4.2 strict-single 与 `train_short`

`train_short.jsonl` 不混入正式三池，避免与 READoc full 重复。

strict-single 是 paired PMC 池中的完整单页 Markdown 行，不是整篇多页 Markdown；它使用 `document parsing.` prompt，
与 PMC full 共享 split。`train_short` 仍不混入正式三池。

### 4.3 GT、源 PDF 与页图映射

训练 ZIP 已携带 4,524 份 standalone GT 和两个 source manifest，不携带 PDF 或页图。最终 mix 中 PMC full/single
复用同一源 PDF，所以 4,524 行对应 3,038 份唯一 PDF、41,980 个去重页需求和 43,466 个图片引用。

先选任意有足够空间的绝对工作目录；不要求 AutoDL 或某个挂载点。包内冻结 JSONL 不原地修改，先生成一份
只改变 `images` 的 relocated 副本，再做离线映射检查：

```bash
cd /path/to/ms-swift
ASSET_ROOT=ms_swift_title_mask/data_assets
FROZEN_MIX="$ASSET_ROOT/final_mix_v1"
READOC_MANIFEST="$ASSET_ROOT/readoc_silver_v1/source_audit/document_manifest.jsonl"
PMC_MANIFEST="$ASSET_ROOT/pmc_silver_v2/source_audit/document_manifest.jsonl"
export UOCR_WORK_ROOT=/absolute/path/to/uocr-work-20260823
export PAGE_ROOT="$UOCR_WORK_ROOT/pages"
export MIX_ROOT="$UOCR_WORK_ROOT/final_mix_v1"

python ms_swift_title_mask/scripts/build_bundle_zip.py --check
python ms_swift_title_mask/scripts/relocate_image_paths.py \
  --input-dir "$FROZEN_MIX" \
  --output-dir "$MIX_ROOT" \
  --image-root "$PAGE_ROOT"
python ms_swift_title_mask/scripts/materialize_pdf_pages.py \
  --dry-run \
  --readoc-manifest "$READOC_MANIFEST" \
  --pmc-manifest "$PMC_MANIFEST" \
  --output-root "$PAGE_ROOT" \
  "$MIX_ROOT/train.jsonl" "$MIX_ROOT/validation.jsonl" "$MIX_ROOT/test.jsonl"
```

dry-run 必须得到 `docs=3038`、`pages=41980`、`failures=0`。然后读取服务器源 PDF 做实体验证：

```bash
export READOC_ROOT=/path/to/readoc_source_root
export PMC_ROOT=/path/to/pmc_v26_pdf_gt_20260816

python ms_swift_title_mask/scripts/materialize_pdf_pages.py \
  --verify-sources --workers 8 \
  --readoc-root "$READOC_ROOT" --readoc-manifest "$READOC_MANIFEST" \
  --pmc-root "$PMC_ROOT" --pmc-manifest "$PMC_MANIFEST" \
  --output-root "$PAGE_ROOT" \
  --report-path "$UOCR_WORK_ROOT/source_verify.json" \
  "$MIX_ROOT/train.jsonl" "$MIX_ROOT/validation.jsonl" "$MIX_ROOT/test.jsonl"
```

`READOC_ROOT` 下必须能解析 manifest 的 `source_pdf_archive`，并在 ZIP 内找到 `source_pdf_member`；`PMC_ROOT`
下必须能解析 manifest 的 `source_pdf`。验证必须得到 `verified_source_docs=3038`、
`verified_source_pages=41980`、`failures=0`。SHA、实际 PDF 页数或页索引任一不符都先修源文件/root，不能改
manifest 或按同名文件强行放行。

验证全过后再渲染到 relocated JSONL 已冻结的 `PAGE_ROOT`；目标目录应为空，默认拒绝覆盖已有页图：

```bash
python ms_swift_title_mask/scripts/materialize_pdf_pages.py \
  --workers 8 --dpi 144 \
  --readoc-root "$READOC_ROOT" --readoc-manifest "$READOC_MANIFEST" \
  --pmc-root "$PMC_ROOT" --pmc-manifest "$PMC_MANIFEST" \
  --output-root "$PAGE_ROOT" \
  --report-path "$UOCR_WORK_ROOT/render.json" \
  "$MIX_ROOT/train.jsonl" "$MIX_ROOT/validation.jsonl" "$MIX_ROOT/test.jsonl"
```

渲染报告必须为 `rendered_docs=3038`、`rendered_pages=41980`、`failures=0`。只有这三层检查全过，GT/PDF/页图
才算完成服务器映射。

## 5. 运行前检查

ms-swift 训练环境使用其 Unlimited-OCR 注册要求的 `transformers==4.46.3`。不要把官方仓库推理 README
中的另一套 transformers 环境直接覆盖到这里。

```bash
export MS_SWIFT_ROOT=/path/to/ms-swift
export MODEL_PATH=/path/to/Unlimited-OCR-model
export UOCR_WORK_ROOT=/absolute/path/to/uocr-work-20260823
export TRAIN_JSONL="$UOCR_WORK_ROOT/final_mix_v1/train.jsonl"
export VAL_JSONL="$UOCR_WORK_ROOT/final_mix_v1/validation.jsonl"
export TEST_JSONL="$UOCR_WORK_ROOT/final_mix_v1/test.jsonl"
bash ms_swift_title_mask/scripts/run_preflight.sh
```

preflight 会检查 ms-swift commit 和关键接口、plugin 注册、transformers 版本、fast tokenizer offset、纯逻辑
单测、CPU DDP 数学以及全部数据契约。ms-swift commit 不同会失败；只有人工 review 过代码 diff 后才能临时设
`ALLOW_UNTESTED_MS_SWIFT=1`，不能为赶进度绕过。

## 6. H100 长度策略

共同设置：base/no-crop（`CROP_MODE=false, IMAGE_SIZE=1024, BASE_SIZE=1024`，同进程含 single/multi）、R-SWA、bf16、
`attn_impl=eager`、batch size 1、gradient checkpointing、packing false、sequence parallel 1，超长策略固定
`raise`，绝不静默左/右截 target。

H100 正式默认使用 32K bucket：

1. 使用事先选出的真实、长度不超过 32,768 且尽量接近 32K 的样本，分别以 `MAX_LENGTH=32768 MAX_STEPS=1`
   跑 full CE 与 title-mask smoke，验证链路、mask 日志和 LoRA gradient。
2. 先做长度预扫描；超过档位的样本进入明确的 overflow manifest，不截断、不假装已训练。若 dense eager OOM，记录结果并退到已实测最高档，A/B 必须使用同一档。

长度扫描在固定 ms-swift commit、Unlimited-OCR v1、fast tokenizer、no-crop 1024 配置下是精确口径：每页
273 个视觉 token，加实际 prompt/assistant tokenizer 长度和 BOS/EOS。273 不是其他 image size、crop mode 或
DeepSeekOCR v2 的通用常量；更换这些条件必须重新 review 计数逻辑。

```bash
export LENGTH_ROOT="$UOCR_WORK_ROOT/length_32k_v1"
python ms_swift_title_mask/scripts/build_length_buckets.py \
  --input-dir "$UOCR_WORK_ROOT/final_mix_v1" \
  --model "$MODEL_PATH" \
  --output-dir "$LENGTH_ROOT" \
  --max-length 32768
```

`report.json` 和 `length_manifest.jsonl` 记录每条样本的实际桶位；正式数据只能使用
`$LENGTH_ROOT/fit/{train,validation,test}.jsonl`，overflow 保留在 `$LENGTH_ROOT/overflow/`，不能截断后塞回 fit。

双卡脚本是标准 DDP，只提升吞吐；一条样本仍完整落在单张 H100 上，所以不会扩大单条 32K 样本容量。
title-mask 也不会减少 attention 或 LM-head 显存。

## 7. 单卡 H100

有效 batch 是 `1 x grad_accum 8 = 8`：

```bash
export MS_SWIFT_ROOT=/path/to/ms-swift
export MODEL_PATH=/path/to/Unlimited-OCR-model
export UOCR_WORK_ROOT=/absolute/path/to/uocr-work-20260823
export LENGTH_ROOT="$UOCR_WORK_ROOT/length_32k_v1"
export TRAIN_JSONL="$LENGTH_ROOT/fit/train.jsonl"
export VAL_JSONL="$LENGTH_ROOT/fit/validation.jsonl"
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/reviewed-title-mask-single"
export LOSS_MODE=title_mask
export MAX_LENGTH=32768
bash ms_swift_title_mask/scripts/run_single_h100.sh
```

full CE 对照只改两个实验变量：

```bash
export LOSS_MODE=full_ce
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/reviewed-full-ce-single"
bash ms_swift_title_mask/scripts/run_single_h100.sh
```

1-step smoke 使用 `$LENGTH_ROOT/smoke/{train,validation}.jsonl` 中真实且最接近 32K 的 fit 样本，并设置
`MAX_LENGTH=32768 MAX_STEPS=1`；不能把 smoke 结果当正式实验。

## 8. 双卡 H100

每卡 batch 1、grad accumulation 4，global effective batch 仍为 8；不加 ZeRO/FSDP：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export MS_SWIFT_ROOT=/path/to/ms-swift
export MODEL_PATH=/path/to/Unlimited-OCR-model
export UOCR_WORK_ROOT=/absolute/path/to/uocr-work-20260823
export LENGTH_ROOT="$UOCR_WORK_ROOT/length_32k_v1"
export TRAIN_JSONL="$LENGTH_ROOT/fit/train.jsonl"
export VAL_JSONL="$LENGTH_ROOT/fit/validation.jsonl"
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/reviewed-title-mask-dual"
export LOSS_MODE=title_mask
export MAX_LENGTH=32768
bash ms_swift_title_mask/scripts/run_dual_h100.sh
```

custom loss 已在每个 micro-step all-reduce active-token denominator，并乘 world size 抵消 DDP 的梯度平均。
`average_tokens_across_devices` 因此固定 false。梯度累计的口径是“每个 micro-step 的 active-token mean 再平均”；
单卡和双卡脚本保持相同 effective document batch。

## 9. title mask 的精确定义

- 用 `markdown-it-py` CommonMark AST 只识别 fenced code 外的 ATX `#...######` heading。
- heading span 含 `#`、标题正文和该行换行；setext、HTML heading、粗体句、图注和代码中的 `#` 不算。
- assistant 完整字符串只 tokenize 一次，并用 fast tokenizer `offset_mapping` 对齐。
- `input_ids` 和完整 labels 与原生 `unlimited_ocr` 完全一致；只替换独立 `loss_scale`。
- heading token 与 EOS 权重 1，prompt/image/body/padding 权重 0。
- token 若同时跨标题和正文的非空白字符，直接报错，不猜归属。
- R-SWA prefix 继续从完整 labels 的第一个非 `-100` 推导，因此 full CE 和 title-mask 的 attention mask 不变。

ms-swift Trainer 已先 roll 并乘 `loss_scale`，custom loss 只做 active-weight sum 归一化，不会再 roll 或再乘一次。
日志增加 `title_token_loss`、`title_token_acc`、`active_title_tokens`、`active_eos_tokens` 和
`masked_body_tokens`。无标题、无 provenance、slow tokenizer 和 token 对齐失败都会停止训练。

## 10. 验收与停止条件

启动日志必须出现 LoRA 84/48/3/33/0 校验，R-SWA collator applied，以及 title-mask 组的 active/masked 指标。
先固定一小批样本，断言两组编码长度与 R-SWA prefix 完全一致。

validation/test 必须在完整自回归生成上比较：

- 标题文本 precision/recall/F1。
- L1-L6 层级准确率、父子边和标题顺序。
- 正文 Markdown 相似度、漏字/重复、公式/表格/代码块回退。
- arXiv/GitHub、READoc/PMC/strict-single 和 32K 长度桶分开报告。

出现以下任一情况停止扩大训练：title-mask 改变 input IDs/labels/R-SWA prefix；routed expert 命中非 0；
正文生成明显回退；长文末尾标题变差；不同 loss 使用了不同长度或数据。训练 loss 下降本身不构成通过。
