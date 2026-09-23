# 脚本索引

当前实验决策见
[`../../docs/reports/analysis_experiments_20260922.md`](../../docs/reports/analysis_experiments_20260922.md)，
评测协议见
[`../../docs/guides/EVAL_GUIDE_2026-09-20.md`](../../docs/guides/EVAL_GUIDE_2026-09-20.md)。训练 JSONL、
PDF、页图、模型、adapter 和输出目录必须位于仓库外。

## 现行训练与校验

| 脚本 | 用途 |
|---|---|
| `hyx_env.sh` | 外部运行目录、ms-swift、模型和数据的环境变量骨架 |
| `validate_reviewed_jsonl.py` | 检查 ms-swift 对话、图像和标题字段契约 |
| `validate_training_recipe.py` | 检查 recipe 的来源、样本形态、重复文档和已知污染 |
| `_run_train.sh` | 统一训练实现，由训练入口调用 |
| `run_single_h100.sh` / `run_dual_h100.sh` | 通用单卡 / 双卡训练入口 |
| `run_pmc_fullce.sh` | PMC 16K / 32K Full-CE 复现实验入口 |
| `run_pmc16k_cont1623.sh` | 从已登记的 PMC checkpoint 续训到 1623 步的诊断入口 |

当前 recipe 为 `readoc_r0`、`replay_r1`、`pmc_s10`、`pmc_fullce`、`pmc_readoc_mix`、
`trusted_title`、`readoc_view_ablation` 和 `legacy_20260823`。最后一个仅用于历史复现，并要求
`UOCR_ALLOW_LEGACY_20260823=1`。

## 当前数据构建

| 脚本 | 用途 |
|---|---|
| `prepare_readoc_view_jsonl.py` | 将 READoc view 转为当前训练 schema |
| `build_pmc_singlepage_jsonl.py` | 从 PMC 32K 与 16K 的差集构造单页样本 |
| `build_pmc_readoc_mix.py` | 合并 PMC full、PMC single-page 与 READoc 数据 |

## 评测与诊断

| 脚本 | 用途 |
|---|---|
| `probe_evaluation_pdfs_parallel.py` | 129-PDF 推理 runner，支持断点续跑和逐文件 prompt manifest |
| `run_inference_service.sh` / `serve_unlimited_ocr.py` | 本机回环地址上的多模型推理服务 |
| `strip_grounding_shell.py` | 保守剥离 grounding 协议；评分目录必须通过 `--check` |
| `aggregate_sweep.py` | 聚合逐篇 OmniDocBench sweep 结果 |
| `synthesize_combined_metric.py` | 合成 AgentBuilder 所需的 combined metric 文件 |
| `probe_train_fit.py` / `render_train_fit_artifacts.py` | Train Fit 抽测与结果渲染 |
| `run_pmc_readoc_spage_eval.sh` | `mix1623` 的完整评测快照 |
| `run_pmc_spage_ckpt1450_eval.sh` | `mix1450` 的 checkpoint 诊断评测快照 |
| `run_pmc16k_cont1623_eval.sh` | PMC `cont1623` 的诊断评测快照 |
| `bypass_repeat_suppression.py` | 解码期复读抑制旁路实验；不是默认推理协议 |

三份 `*_eval.sh` 固化了 2026-09-18 至 09-21 的实验名称、端口和验收阈值，用于复现对应路线。
新路线应按评测指南复制后显式登记新的 `RUN_NAME`、sweep 前缀和人工复核白名单。

## 历史快照工具

`build_readoc_silver_jsonl.py`、`build_pmc_silver_jsonl.py`、`build_reviewed_mix.py`、
`prepare_reviewed_jsonl.py`、`materialize_pdf_pages.py`、`relocate_image_paths.py` 和
`build_bundle_zip.py` 仅用于复现 2026-08-23 快照，不定义新训练的数据准入。

`jsonl_to_markdown.py` 用于渲染已有 JSONL 响应。

## 运行与测试

运行产物按用途写入 `$DATA_ROOT`、`$MODEL_OUTPUT_ROOT`、`$EVALUATION_ROOT`、`$LOG_ROOT` 和
`$SERVICE_ROOT`。训练入口拒绝将权重写到 `$MODEL_OUTPUT_ROOT` 之外；数据通过路径或软链接复用。

单元测试需要训练所用的补丁版 ms-swift：

```bash
source ms_swift_title_mask/scripts/hyx_env.sh
"$PYTHON_BIN" -m pip install -r ms_swift_title_mask/requirements.txt
PYTHONPATH=$POSTTRAIN_ROOT:$MS_SWIFT_ROOT "$PYTHON_BIN" -m unittest discover -s ms_swift_title_mask/tests -t .
```

推理服务默认只监听 `127.0.0.1` 且没有鉴权。若显式改为非回环地址，调用者必须在网络层限制访问。
