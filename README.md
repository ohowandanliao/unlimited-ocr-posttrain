# Unlimited-OCR post-training toolkit

这个仓库保存 Unlimited-OCR 的可审计数据构造、ms-swift external plugin、训练 recipe、离线评测工具和实验记录。
PDF、页图、训练 JSONL、模型、adapter 与日志必须留在仓库外。

## 当前状态

截至 2026-09-22，六条路线已按统一汇总口径完成对照。训练路线中表现最好的 `cont1623`
修正 Overall 为 `0.7205`，仍低于未训练 Base 的 `0.7554`。同为 1623 步时，混合路线
`mix1623` 的多页表格 TEDS / `<table>` 发射数为 `0.3908 / 53`，PMC 续训路线为
`0.5089 / 147`；现有证据指向混合配方整体，但还不能单独归因给 READoc 或单页数据。

当前决策、证据等级和下一轮四臂消融设计见
[`docs/reports/analysis_experiments_20260922.md`](docs/reports/analysis_experiments_20260922.md)。
完整逐篇数据见
[`docs/reports/experiments_dump_20260922.md`](docs/reports/experiments_dump_20260922.md)，文档总入口见
[`docs/README.md`](docs/README.md)。

## 当前入口

| 路径 | 用途 |
|---|---|
| [`docs/README.md`](docs/README.md) | 当前结论、操作指南、审计证据与历史归档索引 |
| [`scripts/data/audit_raw_sources.py`](scripts/data/audit_raw_sources.py) | 只读审计 READoc ZIP 和 PMC PDF/`middle.json` |
| [`ms_swift_title_mask/scripts/validate_training_recipe.py`](ms_swift_title_mask/scripts/validate_training_recipe.py) | 启动前检查语言、来源、重复、正文状态和污染模式 |
| [`ms_swift_title_mask/scripts/probe_evaluation_pdfs_parallel.py`](ms_swift_title_mask/scripts/probe_evaluation_pdfs_parallel.py) | 当前正式评测 runner（129-PDF 统一协议；断点续跑、逐文件 prompt manifest） |
| [`ms_swift_title_mask/scripts/probe_train_fit.py`](ms_swift_title_mask/scripts/probe_train_fit.py) | 训练数据 overfit 抽测（Train Fit），配套 `docs/guides/posttrain_completion_sop_2026-08-28.md` |
| [`ms_swift_title_mask/scripts/strip_grounding_shell.py`](ms_swift_title_mask/scripts/strip_grounding_shell.py) | 将 grounding 协议输出转换为评分用 Markdown，并执行残留检查 |
| [`scripts/evaluation/analyze_outputs.py`](scripts/evaluation/analyze_outputs.py) | 对已保存输出做快速退化分析 |

通用训练走 `ms_swift_title_mask/scripts/run_single_h100.sh` 或 `run_dual_h100.sh`；PMC Full-CE
复现实验可使用 `run_pmc_fullce.sh`。调用者必须显式确认 recipe、loss、模型、数据和新输出目录。

## Prerequisites 与训练起步

需要包含 `UnlimitedOCR` 模板的 ms-swift 源码环境；使用公开上游
[`modelscope/ms-swift`](https://github.com/modelscope/ms-swift) commit
`1a1ba3ee86488af323ef9b64ca3d34edee90ab11`，并对应
[`baidu/Unlimited-OCR`](https://github.com/baidu/Unlimited-OCR) commit
`d49ff64afffc1f47ab563dc1c589bc2f78808fa4`。模型、PDF、页图、训练 JSONL 和运行目录均须放在仓库外。
训练入口面向 Linux CUDA 服务器，支持 1 或 2 张 GPU，并要求系统提供 GNU `realpath -m`。

在该现有环境中安装本 bundle 的补充依赖（不重装 torch），并设置外部路径：

```bash
git clone https://github.com/modelscope/ms-swift.git /absolute/path/to/patched-ms-swift
git -C /absolute/path/to/patched-ms-swift checkout 1a1ba3ee86488af323ef9b64ca3d34edee90ab11

cd /path/to/unlimited-ocr-posttrain
/path/to/ms-swift-venv/bin/python -m pip install -r ms_swift_title_mask/requirements.txt
export UOCR_ROOT=/absolute/path/to/uocr-workspace
export MS_SWIFT_ROOT=/absolute/path/to/patched-ms-swift
export UOCR_VENV=/absolute/path/to/ms-swift-venv
export MODEL_PATH=/absolute/path/to/unlimited-ocr-base
export TRAIN_JSONL=/absolute/path/to/train.jsonl
export VAL_JSONL=/absolute/path/to/val.jsonl
source ms_swift_title_mask/scripts/hyx_env.sh
```

`MODEL_PATH` 指向从 [Hugging Face](https://huggingface.co/baidu/Unlimited-OCR) 或
[ModelScope](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR) 获取的 Base 权重目录。
训练 JSONL 必须符合 [`train_row.example.jsonl`](ms_swift_title_mask/examples/train_row.example.jsonl)
的 schema 并通过启动前 validator；历史数据与评测 manifest 不随本仓库分发。

合法 `TRAINING_RECIPE` 为 `readoc_r0`、`replay_r1`、`pmc_s10`、`pmc_fullce`、
`pmc_readoc_mix`、`trusted_title`、`readoc_view_ablation` 和 `legacy_20260823`；后者只用于历史复现。对于
`prepare_readoc_view_jsonl.py` 生成的数据，READoc view 消融可直接运行：

```bash
export TRAINING_RECIPE=readoc_view_ablation LOSS_MODE=full_ce
export OUTPUT_DIR="$UOCR_ROOT/output/readoc-view-full-ce"
bash ms_swift_title_mask/scripts/run_single_h100.sh

export TRAINING_RECIPE=readoc_view_ablation LOSS_MODE=title_weighted
export OUTPUT_DIR="$UOCR_ROOT/output/readoc-view-title-weighted"
bash ms_swift_title_mask/scripts/run_dual_h100.sh
```

## 路径与开源使用

运行脚本从自身位置定位仓库代码，运行产物通过 `UOCR_ROOT`、`DATA_ROOT`、`MODEL_OUTPUT_ROOT`、
`EVALUATION_ROOT` 等环境变量派生；外部模型、原始数据和评测输入通过环境变量或命令行参数传入。
公开文档使用这些变量和相对路径表达产物位置；真实机器路径只保留在仓库外的私有运行记录中。

## 数据边界

原始数据布局与当前本地源一致：

```text
READoc: ground_truth/{arxiv,github}/*.md + archives/{arxiv,github}.zip
PMC:    documents/<doc_id>/{middle.json,document.pdf}
```

仓库只保存脚本、规则、聚合统计和少量明确标注的失败输出。所有构造结果写到独立外部工作目录；READoc ZIP
直接读取 member，不把 PDF 解压进工程。旧 4,524-row/3,936-fit 数据的历史快照复现文档已于 2026-09-03 清理（git 历史可查），
不能作为新训练默认数据；当前数据链路见 `ms_swift_title_mask/scripts/prepare_readoc_view_jsonl.py`。

## 兼容范围

- ms-swift commit：`1a1ba3ee86488af323ef9b64ca3d34edee90ab11`
- `transformers==4.46.3`
- Unlimited-OCR v1、base/no-crop 1024、每页 273 个视觉 token 的长度 contract
- decoder LoRA 84 个模块；视觉编码器、aligner、embedding、lm_head 与 routed experts 冻结
- R-SWA、SDPA、bf16；不做 packing 或静默截断

`multi_gundam`、FlexAttention、fused CE、sequence parallel 和全参训练不属于当前已验证链路。License：MIT。
