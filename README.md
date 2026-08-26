# Unlimited-OCR post-training toolkit

这个仓库保存 Unlimited-OCR 的可审计数据构造、ms-swift external plugin、训练 recipe、离线评测工具和实验记录。
PDF、页图、训练 JSONL、模型、adapter 与日志必须留在仓库外。

## 当前状态

截至 2026-08-26，旧的 `READoc + PMC full + PMC single` natural union 已停止使用，title+EOS-only、
Full CE 和 Title-weighted 的旧 checkpoint 均不继续训练。失败不是一个标题权重问题：旧 train target 按字符计
`82.32%` 来自 PMC，中文仅约 `0.00023%`；PMC 正文 quarantine 没有进入训练准入，同一 PMCID 又以
full/single 两种形式重复。

8 组中文失败样本的离线重算结果：Base 平均相似度 `0.554`、严重失败 `1/8`；旧 Full CE 为
`0.146`、`6/8`；旧 Title-weighted 为 `0.134`、`7/8`。该相似度仅用于同批输出诊断，不是论文指标。

下一轮固定按以下顺序推进，任一阶段未通过 retention/重复/EOS gate 就停止：

```text
E0 固定 benchmark
 -> R0 clean READoc-only Full CE
 -> R1 加 20%-30% 中文/原生能力 replay
 -> S10 正文验收后的 PMC <=10%
 -> T uniform custom CE / trusted-title loss 消融
 -> V single-base / single-gundam 视觉消融
```

问题证据、研究依据和停止条件见
[`docs/training_optimization_2026-08-26.md`](docs/training_optimization_2026-08-26.md)。可执行命令见
[`ms_swift_title_mask/TRAINING_PLAN_ZH.md`](ms_swift_title_mask/TRAINING_PLAN_ZH.md)。

## 当前入口

| 路径 | 用途 |
|---|---|
| [`docs/README.md`](docs/README.md) | 当前文档、历史文档与事实源索引 |
| [`scripts/data/audit_raw_sources.py`](scripts/data/audit_raw_sources.py) | 只读审计 READoc ZIP 和 PMC PDF/`middle.json` |
| [`ms_swift_title_mask/scripts/build_recipe_mix.py`](ms_swift_title_mask/scripts/build_recipe_mix.py) | 构造 R0/R1/S10 的确定性外部数据 mix |
| [`ms_swift_title_mask/scripts/validate_training_recipe.py`](ms_swift_title_mask/scripts/validate_training_recipe.py) | 启动前检查语言、来源、重复、正文状态和污染模式 |
| [`ms_swift_title_mask/scripts/build_length_buckets.py`](ms_swift_title_mask/scripts/build_length_buckets.py) | 按正式 tokenizer/视觉 contract 生成 fit/overflow |
| [`scripts/evaluation/analyze_outputs.py`](scripts/evaluation/analyze_outputs.py) | 对已保存输出做快速退化分析 |

训练统一走 `ms_swift_title_mask/scripts/run_single_h100.sh` 或 `run_dual_h100.sh`。调用者必须显式设置
`TRAINING_RECIPE`、`LOSS_MODE`、模型、数据和新输出目录；脚本不再默认指向某个历史 run。

## 数据边界

原始数据布局与当前本地源一致：

```text
READoc: ground_truth/{arxiv,github}/*.md + archives/{arxiv,github}.zip
PMC:    documents/<doc_id>/{middle.json,document.pdf}
```

仓库只保存脚本、规则、聚合统计和少量明确标注的失败输出。所有构造结果写到独立外部工作目录；READoc ZIP
直接读取 member，不把 PDF 解压进工程。旧 4,524-row/3,936-fit 数据只能按
[`docs/data_reconstruction_pipeline_2026-08-26.md`](docs/data_reconstruction_pipeline_2026-08-26.md) 做历史快照复现，
不能作为新训练默认数据。

## 兼容范围

- ms-swift commit：`1a1ba3ee86488af323ef9b64ca3d34edee90ab11`
- `transformers==4.46.3`
- Unlimited-OCR v1、base/no-crop 1024、每页 273 个视觉 token 的长度 contract
- decoder LoRA 84 个模块；视觉编码器、aligner、embedding、lm_head 与 routed experts 冻结
- R-SWA、SDPA、bf16；不做 packing 或静默截断

`multi_gundam`、FlexAttention、fused CE、sequence parallel 和全参训练不属于当前已验证链路。License：MIT。
