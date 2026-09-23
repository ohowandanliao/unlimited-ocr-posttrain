# 文档索引

更新时间：2026-09-23（Asia/Shanghai）

本目录只保留四类材料：操作指南、实验报告、审计证据和历史归档。当前决策以
[`reports/analysis_experiments_20260922.md`](reports/analysis_experiments_20260922.md)
为准；其他文档用于复现数字、解释口径或追溯历史，不再各自承担“当前状态”入口。

## 当前结论

- 六路线统一汇总中，`cont1623` 的修正 Overall 为 `0.7205`，高于 `pmc16k` 的
  `0.7092`，但仍低于未训练 Base 的 `0.7554`。
- `mix1623` 的主要新增损伤集中在多页表格：TEDS `0.3908`，多页 `<table>` 发射数
  `53`；同为 1623 步的 PMC 续训路线分别为 `0.5089` 和 `147`。
- 现有证据支持“混合配方整体是主要嫌疑”，但不能把责任单独归给 READoc 或单页数据。
- 下一步是从同一 Base、相同步数和调度启动 PMC / PMC+READoc / PMC+spage / 完整混合
  四臂消融，并把表格覆盖、结构缺失和复读纳入验收。

以上数字来自已保存评测产物。9 月 22 日综合分析没有重新访问服务器权重、训练数据或原始
评测目录，证据边界见当前决策文档。

## 阅读顺序

| 优先级 | 文档 | 用途 |
|---|---|---|
| 1 | [`reports/analysis_experiments_20260922.md`](reports/analysis_experiments_20260922.md) | 当前结论、证据强弱和下一轮实验设计 |
| 2 | [`reports/experiments_dump_20260922.md`](reports/experiments_dump_20260922.md) | 六路线总表、逐篇结果和三组新增实验的完整数据 |
| 3 | [`guides/EVAL_GUIDE_2026-09-20.md`](guides/EVAL_GUIDE_2026-09-20.md) | 129-PDF 评测流程、口径和历史坑位 |
| 4 | [`reports/posttrain_weekly_report_2026-09-20.md`](reports/posttrain_weekly_report_2026-09-20.md) | 前两轮 Full-CE 训练的阶段总结 |

## 目录说明

### `guides/`：现行操作文档

- [`EVAL_GUIDE_2026-09-20.md`](guides/EVAL_GUIDE_2026-09-20.md)：从 checkpoint 到修正 Overall 的完整评测手册。
- [`posttrain_completion_sop_2026-08-28.md`](guides/posttrain_completion_sop_2026-08-28.md)：Benchmark 与 Train Fit 验收流程。

### `reports/`：实验结果与综合分析

- [`analysis_experiments_20260922.md`](reports/analysis_experiments_20260922.md)：当前决策摘要。
- [`experiments_dump_20260922.md`](reports/experiments_dump_20260922.md)：最新完整数字和逐篇明细。
- [`posttrain_weekly_report_2026-09-20.md`](reports/posttrain_weekly_report_2026-09-20.md)：PMC 与混合路线阶段报告。
- [`evaluation_readoc_view_16k_2026-08-27.md`](reports/evaluation_readoc_view_16k_2026-08-27.md)：READoc 基线评测记录。

### `audits/`：独立复核与数据证据

- [`two_rounds_review_2026-09-18.md`](audits/two_rounds_review_2026-09-18.md)：两轮实验复核与评分口径修正；复算脚本在 [`two_rounds_20260918/`](audits/two_rounds_20260918/README.md)。
- [`repeat_audit_2026-09-18.md`](audits/repeat_audit_2026-09-18.md)：复读、训练数据和 R-SWA 机制审计；产物在 [`repeat_20260918/`](audits/repeat_20260918/README.md)。
- [`pmc_fullce_attribution_2026-09-17.md`](audits/pmc_fullce_attribution_2026-09-17.md)：首轮 PMC 的逐篇掉分归因。
- [`pmc_data_quality_audit_casebook_2026-08-26.md`](audits/pmc_data_quality_audit_casebook_2026-08-26.md)：PMC 原始数据典型案例及配套表格。

### `archive/`：已被新结论取代的历史材料

- [`HANDOFF_2026-09-01.md`](archive/HANDOFF_2026-09-01.md)：Title Prior 阶段交接记录。
- [`wxz_readoc_title_mask_training_2026-08-29.md`](archive/wxz_readoc_title_mask_training_2026-08-29.md)：早期 READoc Title-mask 训练归档。

归档文件保留当时语境，其中的“下一步”和已删除文档名不再代表当前仓库状态。

## 代码入口与仓库边界

训练、数据构建和评测脚本见
[`../ms_swift_title_mask/scripts/README.md`](../ms_swift_title_mask/scripts/README.md)。仓库不保存
PDF、页图、训练 JSONL、模型、adapter、评测输出或日志；这些产物必须位于外部运行目录。
