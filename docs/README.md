# 文档索引与当前事实源

更新时间：2026-09-04（Asia/Shanghai，开源整理）

当前状态：**129-PDF 统一协议评测闭环完成。6 套已完整评测方案全部相对主 Base 回退
（-1.70 ~ -43.48 pp），9 套方案 Train Fit 均为 `partial_learning`，不进入部署。**
case 级归因已完成：带壳模型的回退由推理时标题段触发输出协议翻转造成，先验训练与
先验质量均无责，详见
[`analysis_all_regress_attribution_2026-09-02.md`](analysis_all_regress_attribution_2026-09-02.md)。
当前交接总览见 [`HANDOFF_2026-09-01.md`](HANDOFF_2026-09-01.md)。
2026-09-02 至 09-04 已完成文档与脚本清理：过时预研、旧协议材料、未执行的 recipe 阶梯
工具、smoke 测试产物与无法复现的旧快照文档已删除；PMC 审计与历史训练归档保留在本目录，
正式协议默认值、公开路径变量和现行测试已对齐。

## Base 评测口径

正式主基线是未训练 Base 使用 `<image>Multi page parsing.` 的结果，当前主基线为
OmniDocBench Total `53.3645`、AI Builder Overall `0.8271`。训练模型按照训练契约使用
`<image>Multi page merge.`；不能为了"统一 prompt"而让未训练 Base 使用 merge。

夏桢历史评测中的 Base 也是未训练 Base，但使用了 `Multi page merge.`，并且评测入口、单页
图像处理和 `ngram_window` 也不同，因此得到 OmniDocBench `51.7663`、AI Builder Overall
`0.7442`。这不是 Base 权重或 Overall 算法不同造成的结果：两种 Overall 都是
`(Text accuracy + Table accuracy + Reading-order accuracy + Title accuracy) / 4`。

后续周报、方案登记和正式泛化结论统一以 `parsing` Base 为主；夏桢的 `merge` Base 只作为
历史评测 diff 参考，不能作为训练方案的正式相对 Base。具体入口和参数差异见
[`inference_service_2026-08-24.md`](inference_service_2026-08-24.md)。

## 路径变量

公开记录用变量代替原机器路径：`$UOCR_ROOT` 是仓库外运行根目录，`$MODEL_PATH` 是 Base 模型，
`$EVAL_PDF_ROOT` / `$EVAL_GT_ROOT` / `$OMNIDOCBENCH_ROOT` 是评测输入与工具，
`$MINERU_OUTPUT_ROOT` 是标题先验来源。`$WXZ_ROOT`、`$WXZ_READOC_ROOT`、`$WXZ_OUTPUT_ROOT`、
`$PMC_SOURCE_ROOT`、`$PMC_TITLE_CANDIDATES_ROOT`、`$LEGACY_MIX_ROOT` 和
`$LEGACY_READOC_SHORT_JSONL` 仅表示历史外部资产位置，不由 `hyx_env.sh` 自动设置。

## 当前文档

| 文档 | 用途 |
|---|---|
| [`HANDOFF_2026-09-01.md`](HANDOFF_2026-09-01.md) | 当前状态、已清理内容、与后续任务的交接总览 |
| [`posttrain_weekly_report_2026-08-30.md`](posttrain_weekly_report_2026-08-30.md) | 9 套已训练方案的周报总览：数据、训练参数、checkpoint、Benchmark/Base 对比、Train Fit 和下一步计划 |
| [`analysis_all_regress_attribution_2026-09-02.md`](analysis_all_regress_attribution_2026-09-02.md) | 全量回退归因汇总：统一协议结果 + case 级证据 + 两轴定责（壳/loss） |
| [`evaluation_prompt_matrix_2026-09-01.md`](evaluation_prompt_matrix_2026-09-01.md) | 评测 prompt 唯一标准：训练契约、标准矩阵、manifest 规则与各轮审计 |
| [`evaluation_readoc_view_16k_2026-08-27.md`](evaluation_readoc_view_16k_2026-08-27.md) | 2026-08-27 READoc view 16K 全量评测、AI Builder 细分、title 分、base 差分与复读统计 |
| [`evaluation_readoc_heading_prior_2026-09-01.md`](evaluation_readoc_heading_prior_2026-09-01.md) | 本轮两套 Title Prior 模型的正式评测记录（§8）与 case 级归因原文（§9） |
| [`train_fit_heading_prior_2026-09-01.md`](train_fit_heading_prior_2026-09-01.md) | 2026-09-01 两套 Title Prior 模型的 Train Fit / overfit 验收 |
| [`runs_registry_2026-08-27.md`](runs_registry_2026-08-27.md) | 训练方案、LoRA checkpoint、评测数据/结果和运行目录的对应关系 |
| [`training_scheme_registry_2026-08-29.md`](training_scheme_registry_2026-08-29.md) | 全部已训练方案的统一登记，以及每个方案的 Benchmark / Train Fit 双验收状态 |
| [`posttrain_completion_sop_2026-08-28.md`](posttrain_completion_sop_2026-08-28.md) | 每次训练完成后的固定 benchmark 对比、train 数据 overfit 抽测和结论判定规则 |
| [`inference_service_2026-08-24.md`](inference_service_2026-08-24.md) | 推理入口、多 adapter 部署、prompt 矩阵、Base 主口径与夏桢历史评测 diff |
| [`../ms_swift_title_mask/scripts/README.md`](../ms_swift_title_mask/scripts/README.md) | 当前脚本、历史数据 builder 与旧评测工具分类 |

## 数据实现与审计文档

以下文件用于解释构造代码与数据质量事实，不单独决定训练准入：

- [`pmc_title_silver_pipeline_2026-08-23.md`](pmc_title_silver_pipeline_2026-08-23.md)：PMC 标题规则和正文 quarantine 事实（PMC 线暂停，方案保留）。
- [`pmc_data_quality_audit_casebook_2026-08-26.md`](pmc_data_quality_audit_casebook_2026-08-26.md) + [`.xlsx`](pmc_data_quality_audit_casebook_2026-08-26.xlsx)：9 个 PMC 原始数据质量典型 case 与对账表。
- [`wxz_readoc_title_mask_training_2026-08-29.md`](wxz_readoc_title_mask_training_2026-08-29.md)：夏桢两次 READoc Title-mask 训练、原始数据校验值、权重位置与旧 badcase 评测归档；仅作历史记录，评测数字已按统一协议重测取代。
- [`../ms_swift_title_mask/data_assets/STATUS_ZH.md`](../ms_swift_title_mask/data_assets/STATUS_ZH.md)：旧 ZIP payload 说明；Git clone 不包含 payload。
- 训练数据校验记录：`$UOCR_ROOT/data/recipes/*.recipe_audit.json`（5 份，对应已训练方案）。

## 已清理内容

2026-09-02/03 清理中删除：`docs/archive/legacy/`（11 个 2026-07-14 ~ 08-27 预研/设计文档，
git 历史有底）、旧协议失败 case 分析（`evaluation_failure_cases_2026-08-25/`）、旧"下一轮
方案"（`training_optimization_2026-08-26.md`）、论文检索笔记（`research_ocr_posttraining_2026-08-26.md`）、
重复副本（`inference_service_2026-08-24.multi.md`）与脚本 `.orig` 备份。删除清单详见
HANDOFF 的"已清理"章节。

## 不进入 Git 的内容

PDF、页图、训练 JSONL、模型、adapter 和日志仍放在仓库外。仓库只保存构造/审计脚本、数量与 SHA
契约、聚合报告及少量明确标注的结构化失败输出。新工具默认拒绝把输出写进源数据目录，并且不会解压
READoc PDF 到工程中。
