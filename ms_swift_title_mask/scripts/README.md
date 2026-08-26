# 脚本索引

当前训练决策以 [`../../docs/training_optimization_2026-08-26.md`](../../docs/training_optimization_2026-08-26.md)
为准，可执行顺序见 [`../TRAINING_PLAN_ZH.md`](../TRAINING_PLAN_ZH.md)。训练 JSONL、PDF、页图、模型和输出目录
必须位于仓库外。

## 当前入口

| 脚本 | 用途 |
|---|---|
| `hyx_env.sh` | 服务器环境骨架；不默认填模型、数据、recipe 或 loss |
| `build_recipe_mix.py` | 构造 R0/R1/S10/trusted-title 的确定性数据 mix |
| `validate_training_recipe.py` | 检查来源、语言比例、PMC 占比、重复文档、正文状态和已知污染 |
| `validate_reviewed_jsonl.py` | 检查 ms-swift 对话与图像字段契约 |
| `build_length_buckets.py` | 用正式 tokenizer/视觉 contract 做长度分桶，不静默截断 |
| `check_ms_swift_compat.py`、`run_preflight.sh` | 固定 ms-swift 接口和运行环境 |
| `run_single_h100.sh`、`run_dual_h100.sh` | 当前统一训练入口，最终调用 `_run_train.sh` |

仓库级原始数据审计使用 `../../scripts/data/audit_raw_sources.py`；已保存输出的快速退化分析使用
`../../scripts/evaluation/analyze_outputs.py`。

## 历史数据快照工具

`build_readoc_silver_jsonl.py`、`build_pmc_silver_jsonl.py`、`build_reviewed_mix.py`、
`prepare_reviewed_jsonl.py`、`materialize_pdf_pages.py`、`relocate_image_paths.py` 和 `build_bundle_zip.py`
保留用于复现 2026-08-23 快照。它们不定义下一轮数据准入，其中旧 PMC builder 只冻结标题规则，
没有消费正文 quarantine；其输出不得用于 `pmc_s10`。

`schedule_title_weighted.sh` 只用于复现 2026-08-25 的旧 title-weighted 训练调度；
`run_inference_service.sh`、`run_parallel_evaluation.sh`、`serve_unlimited_ocr.py`、
`probe_evaluation_pdfs.py`、`probe_evaluation_pdfs_parallel.py` 和 `jsonl_to_markdown.py` 只用于复核
同一轮旧评测。三个 shell 入口均要求显式设置 `UOCR_ALLOW_LEGACY_20260825=1`。

历史训练 recipe 还要求 `UOCR_ALLOW_LEGACY_20260823=1`。新实验只能使用命名 recipe，并必须通过
`validate_training_recipe.py`；不要绕过 gate 直接把旧 natural union 交给 ms-swift。
