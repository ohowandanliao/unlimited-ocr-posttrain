# 从 READoc / PMC 源数据重建 ms-swift 标题训练资产

> **历史 2026-08-23 快照重建手册，不是下一轮训练入口。** 其中 PMC full/single natural union 会重现已知的
> 正文准入缺陷，只能用于核对旧实验。新 R0/R1/S10 命令见
> [`../../../ms_swift_title_mask/TRAINING_PLAN_ZH.md`](../../../ms_swift_title_mask/TRAINING_PLAN_ZH.md)。

本文面向一台新的 Linux 服务器：只有 `ms-swift-patch` 分支、外部 ms-swift checkout、Unlimited-OCR
模型和 READoc/PMC 源数据，也能从标题 GT 审计一直走到单卡或双卡 H100 训练。服务器不需要 AutoDL，所有
工作路径都由环境变量指定。

如果已经拿到本项目生成的完整 ZIP，不需要重建 GT；应直接执行 ZIP 内 `TRAINING_PLAN_ZH.md` 第 4.3 节，
先 relocation，再验证 PDF、渲页、分桶和训练。ZIP 已经携带 4,524 份 GT 和 final mix，但仍不携带 3,038 份
源 PDF 或 41,980 张页图。

## 1. 固定版本和源数据布局

测试基线：

- posttrain 分支：`ms-swift-patch`
- ms-swift commit：`1a1ba3ee86488af323ef9b64ca3d34edee90ab11`
- Unlimited-OCR repository：`d49ff64afffc1f47ab563dc1c589bc2f78808fa4`
- ms-swift 环境中的 transformers：`4.46.3`

READoc root 必须是以下布局；Markdown stem 和 ZIP 内 PDF stem 必须一一对应：

```text
READOC_SOURCE/
  ground_truth/
    arxiv/<doc_id>.md
    github/<doc_id>.md
  archives/
    arxiv.zip        # member: arxiv/pdf/<doc_id>.pdf
    github.zip       # member: github/pdf/<doc_id>.pdf
```

PMC root 必须是：

```text
PMC_SOURCE/
  documents/
    <doc_id>/
      middle.json
      document.pdf
```

所有生成目录必须是新的、彼此不嵌套的目录。脚本默认拒绝覆盖，避免把源 GT、PDF 或已有审计结果改坏。

## 2. 环境和路径

先进入已经能运行 ms-swift 的 conda 环境，只补充小依赖，不重新安装 torch：

```bash
export POSTTRAIN_ROOT=/absolute/path/to/unlimited-ocr-posttrain
export MODEL_PATH=/absolute/path/to/Unlimited-OCR-model
export READOC_SOURCE=/absolute/path/to/READoc
export PMC_SOURCE=/absolute/path/to/pmc_v26_pdf_gt_20260816
export UOCR_ROOT=/absolute/path/to/uocr-ms-swift-title-mask
export MS_SWIFT_ROOT="$UOCR_ROOT/repos/ms-swift-uocr"
export UOCR_SHIMS_ROOT="$UOCR_ROOT/runtime/shims"
export PYTHON_BIN="$UOCR_ROOT/env/ms-swift-venv/bin/python"
export UOCR_WORK_ROOT="$UOCR_ROOT/runs/20260823"
export PAGE_ROOT="$UOCR_WORK_ROOT/pages"
export ASSET_ROOT="$POSTTRAIN_ROOT/ms_swift_title_mask/data_assets"

cd "$POSTTRAIN_ROOT"
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r ms_swift_title_mask/requirements.txt
```

新 clone 的 `ASSET_ROOT` 只有 `.gitignore` 和 `STATUS_ZH.md`；下面三个被 Git 忽略的 payload 目录必须尚不
存在：`readoc_silver_v1`、`pmc_silver_v2`、`final_mix_v1`。

## 3. 从源数据生成可追溯 silver GT

### 3.1 READoc

READoc builder 同时读取原始 Markdown 和 ZIP 内 PDF，使用仓库固定的 review policy，只允许标题范围改动，并
记录 source/derived/PDF SHA、PDF 页数和每处修改证据：

```bash
python scripts/data/build_readoc_title_gt.py \
  --input-root "$READOC_SOURCE" \
  --output-root "$UOCR_WORK_ROOT/readoc_title_gt_v1"
```

当前 v1 源数据的硬闸门是 `validation.json.valid=true`、2,233 篇全部完成 PDF/GT 配对且源文件无变异。
规则会将其中 1,552 篇 `SILVER_CANDIDATE` 冻结进入训练；681 篇 `REVIEW_REQUIRED` 不自动准入。

```bash
python ms_swift_title_mask/scripts/build_readoc_silver_jsonl.py \
  --gt-root "$UOCR_WORK_ROOT/readoc_title_gt_v1" \
  --image-root "$PAGE_ROOT" \
  --output-dir "$ASSET_ROOT/readoc_silver_v1" \
  --skip-image-existence
```

预期结果：1,552 documents、15,290 pages、21,400 headings，train/validation/test 为 1,411/71/70。

### 3.2 PMC

PMC builder 逐篇验证 `middle.json`、`document.pdf`、PDF 页数和 SHA，再输出确定性标题候选与审计队列：

```bash
python scripts/data/build_pmc_title_candidates.py \
  --input-root "$PMC_SOURCE" \
  --output-root "$UOCR_WORK_ROOT/pmc_title_audit_v1"
```

当前 v26 源数据预期为 1,486 documents、26,690 pages、48,844 candidates、0 failed documents。随后按
冻结规则生成每篇完整多页 Markdown，并为每篇确定性选择一张有标题的 strict-single 页：

```bash
python ms_swift_title_mask/scripts/build_pmc_silver_jsonl.py \
  --input-root "$PMC_SOURCE" \
  --audit-root "$UOCR_WORK_ROOT/pmc_title_audit_v1" \
  --image-root "$PAGE_ROOT" \
  --output-dir "$ASSET_ROOT/pmc_silver_v2" \
  --skip-image-existence
```

不要加 `--skip-pdf-hash`。预期 PMC full 与 PMC single 各 1,486 行，train/validation/test 均为
1,331/76/79；full 为
48,726 headings，single 为 6,011 headings。

`title_gt_pipeline/markdown-title-repair/` 可以接其他 OpenAI-compatible API 生成行级 review candidate，
但当前冻结版本不依赖该 API。API 结果不能自动升级为 `HUMAN_ACCEPTED`，也不能绕过上述确定性规则、PDF
证据和 SHA 闸门。

## 4. 组装 source audit、GT 和 final mix

silver builders 已写出 standalone GT 和池 JSONL。为了让 ZIP 保留从训练行回到源 PDF/GT 的证据，再复制
最小 source audit 文件；不复制 PDF、原始 Markdown 或 review shard：

```bash
mkdir "$ASSET_ROOT/readoc_silver_v1/source_audit"
cp "$UOCR_WORK_ROOT/readoc_title_gt_v1/change_manifest.jsonl" \
   "$UOCR_WORK_ROOT/readoc_title_gt_v1/document_manifest.jsonl" \
   "$UOCR_WORK_ROOT/readoc_title_gt_v1/review_policy.json" \
   "$UOCR_WORK_ROOT/readoc_title_gt_v1/summary.json" \
   "$UOCR_WORK_ROOT/readoc_title_gt_v1/validation.json" \
   "$ASSET_ROOT/readoc_silver_v1/source_audit/"

mkdir "$ASSET_ROOT/pmc_silver_v2/source_audit"
cp "$UOCR_WORK_ROOT/pmc_title_audit_v1/document_manifest.jsonl" \
   "$UOCR_WORK_ROOT/pmc_title_audit_v1/title_candidates.jsonl" \
   "$ASSET_ROOT/pmc_silver_v2/source_audit/"
```

三池使用固定 seed 做 natural union。PMC full/single 必须来自同一文档并保持同 split：

```bash
python ms_swift_title_mask/scripts/build_reviewed_mix.py \
  --readoc-dir "$ASSET_ROOT/readoc_silver_v1/readoc_full" \
  --pmc-full-dir "$ASSET_ROOT/pmc_silver_v2/pmc_full" \
  --pmc-single-dir "$ASSET_ROOT/pmc_silver_v2/pmc_single" \
  --output-dir "$ASSET_ROOT/final_mix_v1" \
  --skip-image-existence

python ms_swift_title_mask/scripts/build_bundle_zip.py --check
```

打包检查必须得到 READoc/PMC full/PMC single 为 1,552/1,486/1,486，split 为 4,073/223/228，standalone
GT 为 4,524，唯一 PDF 为 3,038。它会逐行核验 assistant target 与 GT、target SHA、PDF SHA/页数、页索引、
canonical image suffix 和 PMC full/single 配对；任一不符都不能继续。

需要给其他服务器上传时，在 Git 仓库外生成 ZIP：

```bash
python ms_swift_title_mask/scripts/build_bundle_zip.py \
  --output /absolute/path/unlimited-ocr-ms-swift-title-mask-20260823.zip
```

## 5. 验证服务器 PDF 并渲染页图

从源重建时，JSONL 已使用本机最终 `PAGE_ROOT`，无需再次 relocation。从其他机器生成的 ZIP 解压过来时，
必须先运行 `relocate_image_paths.py` 生成一份新 JSONL；禁止原地改冻结 JSONL。具体命令见包内
`TRAINING_PLAN_ZH.md` 第 4.3 节。

先只检查训练行、manifest 和目标页映射，不读取 PDF：

```bash
export MIX_ROOT="$ASSET_ROOT/final_mix_v1"
export READOC_MANIFEST="$ASSET_ROOT/readoc_silver_v1/source_audit/document_manifest.jsonl"
export PMC_MANIFEST="$ASSET_ROOT/pmc_silver_v2/source_audit/document_manifest.jsonl"

python ms_swift_title_mask/scripts/materialize_pdf_pages.py \
  --dry-run \
  --readoc-manifest "$READOC_MANIFEST" \
  --pmc-manifest "$PMC_MANIFEST" \
  --output-root "$PAGE_ROOT" \
  "$MIX_ROOT/train.jsonl" "$MIX_ROOT/validation.jsonl" "$MIX_ROOT/test.jsonl"
```

必须为 `docs=3038, pages=41980, failures=0`。再读取本机源 PDF 做 SHA 和真实页数全量验证：

```bash
python ms_swift_title_mask/scripts/materialize_pdf_pages.py \
  --verify-sources --workers 8 \
  --readoc-root "$READOC_SOURCE" --readoc-manifest "$READOC_MANIFEST" \
  --pmc-root "$PMC_SOURCE" --pmc-manifest "$PMC_MANIFEST" \
  --output-root "$PAGE_ROOT" \
  --report-path "$UOCR_WORK_ROOT/source_verify.json" \
  "$MIX_ROOT/train.jsonl" "$MIX_ROOT/validation.jsonl" "$MIX_ROOT/test.jsonl"
```

必须为 `verified_source_docs=3038, verified_source_pages=41980, failures=0`。最后渲染；`PAGE_ROOT` 应为空，
脚本默认拒绝覆盖：

```bash
python ms_swift_title_mask/scripts/materialize_pdf_pages.py \
  --workers 8 --dpi 144 \
  --readoc-root "$READOC_SOURCE" --readoc-manifest "$READOC_MANIFEST" \
  --pmc-root "$PMC_SOURCE" --pmc-manifest "$PMC_MANIFEST" \
  --output-root "$PAGE_ROOT" \
  --report-path "$UOCR_WORK_ROOT/render.json" \
  "$MIX_ROOT/train.jsonl" "$MIX_ROOT/validation.jsonl" "$MIX_ROOT/test.jsonl"
```

必须为 `rendered_docs=3038, rendered_pages=41980, failures=0`。只有这三层全部通过，才说明 PDF、GT、页索引
和训练图片完成闭环；文件名相似不能代替 SHA 验证。

## 6. 长度分桶、preflight 和训练

按训练时相同 tokenizer 和 no-crop 1024 契约扫描，不截断 target：

```bash
export LENGTH_ROOT="$UOCR_WORK_ROOT/length_32k_v1"
python ms_swift_title_mask/scripts/build_length_buckets.py \
  --input-dir "$MIX_ROOT" \
  --model "$MODEL_PATH" \
  --output-dir "$LENGTH_ROOT" \
  --max-length 32768

export TRAIN_JSONL="$LENGTH_ROOT/fit/train.jsonl"
export VAL_JSONL="$LENGTH_ROOT/fit/validation.jsonl"
export TEST_JSONL="$LENGTH_ROOT/fit/test.jsonl"
bash ms_swift_title_mask/scripts/run_preflight.sh
```

先用真实临界长度样本做 1-step smoke，H100 依次确认 16K、24K、32K 中原生 dense eager R-SWA 能稳定
backward 的最高档。full CE 与 title-mask 必须使用同一最高稳定长度；双卡 DDP 只增加吞吐，不扩大单条样本
显存容量。

单卡 title-mask：

```bash
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/reviewed-title-mask-single"
export LOSS_MODE=title_mask
export MAX_LENGTH=32768
bash ms_swift_title_mask/scripts/run_single_h100.sh
```

双卡只替换入口并保持 global effective batch 8：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OUTPUT_DIR="$UOCR_WORK_ROOT/outputs/reviewed-title-mask-dual"
bash ms_swift_title_mask/scripts/run_dual_h100.sh
```

full CE 对照只改 `LOSS_MODE=full_ce` 和新 `OUTPUT_DIR`。所有正式超参、LoRA 84-module guard、title-mask
精确定义、日志指标和停止条件以 `ms_swift_title_mask/TRAINING_PLAN_ZH.md` 为唯一执行口径。
