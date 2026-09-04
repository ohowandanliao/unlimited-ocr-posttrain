# 脚本索引

当前训练决策与下一轮方向以 [`../../docs/posttrain_weekly_report_2026-08-30.md`](../../docs/posttrain_weekly_report_2026-08-30.md) §5
与 [`../../docs/analysis_all_regress_attribution_2026-09-02.md`](../../docs/analysis_all_regress_attribution_2026-09-02.md)
为准。合法 `TRAINING_RECIPE` 取值以 `validate_training_recipe.py` 实现为准；已训练方案的 recipe
audit 记录在运行目录 `data/recipes/`。训练 JSONL、PDF、页图、模型和输出目录必须位于仓库外。

运行产物统一按用途写入运行根目录：DATA_ROOT=data/，MODEL_OUTPUT_ROOT=output/，EVALUATION_ROOT=evaluation/，训练与评测日志分别写入 logs/，服务日志写入 service/。训练入口拒绝 output/ 之外的权重目录；同一份数据通过路径或 data/review/ 软链接复用，不复制到每个方案。

## 当前入口

| 脚本 | 用途 |
|---|---|
| `hyx_env.sh` | 服务器环境骨架；不默认填模型、数据、recipe 或 loss |
| `validate_training_recipe.py` | 检查来源、语言比例、PMC 占比、重复文档、正文状态和已知污染 |
| `validate_reviewed_jsonl.py` | 检查 ms-swift 对话与图像字段契约 |
| prepare_readoc_view_jsonl.py | 将旧 READoc view 转为当前可追溯的对话/标题训练 schema |
| `run_single_h100.sh`、`run_dual_h100.sh` | 当前统一训练入口，最终调用 `_run_train.sh` |
| `probe_evaluation_pdfs_parallel.py` | 当前正式评测 runner（129-PDF 统一协议；断点续跑会校验旧记录的 prompt 与生成参数，缺 `gen` 块默认重跑，`--trust-legacy-rows` 可显式接受；支持逐文件 prompt manifest） |
| `run_inference_service.sh`、`serve_unlimited_ocr.py` | 当前多模型推理服务；模型与两个 adapter 路径均须显式设置，服务日志写入运行目录的 `service/` |
| `probe_train_fit.py`、`render_train_fit_artifacts.py` | 当前 Train Fit 抽测与产物渲染，配套 `../../docs/posttrain_completion_sop_2026-08-28.md` |

仓库级原始数据审计使用 `../../scripts/data/audit_raw_sources.py`；已保存输出的快速退化分析使用
`../../scripts/evaluation/analyze_outputs.py`。

## 历史数据快照工具

`build_readoc_silver_jsonl.py`、`build_pmc_silver_jsonl.py`、`build_reviewed_mix.py`、
`prepare_reviewed_jsonl.py`、`materialize_pdf_pages.py`、`relocate_image_paths.py` 和 `build_bundle_zip.py`
保留用于复现 2026-08-23 快照。它们不定义下一轮数据准入，其中旧 PMC builder 只冻结标题规则，
没有消费正文 quarantine；其输出不得作为新训练数据准入。

`jsonl_to_markdown.py` 把已有 JSONL 响应渲染为 Markdown。

## 运行时 shims

`../shims/apex.py` 通过 PYTHONPATH 挂载（`hyx_env.sh` 与 `_run_train.sh` 的 `UOCR_SHIMS_ROOT`，
默认 `$POSTTRAIN_ROOT/ms_swift_title_mask/shims`），用于在 bf16 训练环境屏蔽 Apex 的可选导入。

## 测试

单元测试必须使用补丁版 ms-swift（含 `UnlimitedOCR` 模板与 loss 注册），即训练所用的
`MS_SWIFT_ROOT`（见 `hyx_env.sh`，默认 `$UOCR_ROOT/repos/ms-swift-uocr`）。在该现有环境中先安装
本 bundle 的补充依赖（不重装 torch），再运行当前测试：

```bash
source ms_swift_title_mask/scripts/hyx_env.sh
"$PYTHON_BIN" -m pip install -r ms_swift_title_mask/requirements.txt
PYTHONPATH=$POSTTRAIN_ROOT:$MS_SWIFT_ROOT "$PYTHON_BIN" -m unittest discover -s ms_swift_title_mask/tests -t .
```

## 服务边界

推理服务默认仅监听 `127.0.0.1`，且没有鉴权；仅可在可信本机网络使用。若显式改为非回环地址，调用者
必须在网络层自行限制访问。

历史快照复现的 recipe 需要
`UOCR_ALLOW_LEGACY_20260823=1`；新实验只能使用命名 recipe，并必须通过 `validate_training_recipe.py`；
不要绕过 gate 直接把旧 natural union 交给 ms-swift。
