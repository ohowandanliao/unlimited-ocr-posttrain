# 文档索引与当前事实源

更新时间：2026-08-26（Asia/Shanghai）

当前状态是**暂停继续训练，先修数据与评测契约**。2026-08-24/25 的 Full CE 和
Title-weighted 均出现严重生成退化；title+EOS-only 也没有完成可用性评测。任何旧文档中的
“正式训练入口”“唯一事实源”或“当前方案”若与本页冲突，均按历史记录处理。

## 当前文档

| 文档 | 用途 |
|---|---|
| [`training_optimization_2026-08-26.md`](training_optimization_2026-08-26.md) | 问题证据、根因排序、下一轮训练与停止条件；当前训练决策唯一事实源 |
| [`data_reconstruction_pipeline_2026-08-26.md`](data_reconstruction_pipeline_2026-08-26.md) | 精确复现 2026-08-23 **旧数据快照**；不是下一轮数据配方 |
| [`evaluation_failure_cases_2026-08-25/README.md`](evaluation_failure_cases_2026-08-25/README.md) | 8 组、24 条结构化原始失败输出、GT 与按需展开命令 |
| [`evaluation_failure_cases_2026-08-25/analysis_2026-08-26.md`](evaluation_failure_cases_2026-08-25/analysis_2026-08-26.md) | 对失败包重新计算的长度、重复、协议和结构指标 |
| [`research_ocr_posttraining_2026-08-26.md`](research_ocr_posttraining_2026-08-26.md) | 2024-2026 论文检索全量结果、检索失败和方法启示 |
| [`pmc_data_quality_audit_casebook_2026-08-26.md`](pmc_data_quality_audit_casebook_2026-08-26.md) | 9 个 PMC 原始数据问题典型 case；含定界抽样频次、JSONL 行号及 middle 精确回溯 |
| [`pmc_data_quality_audit_casebook_2026-08-26.xlsx`](pmc_data_quality_audit_casebook_2026-08-26.xlsx) | 上述审计 case 的精简对账表；`Cases`、`Sampling` 与 `Snapshot` 三个工作表 |
| [`../ms_swift_title_mask/TRAINING_PLAN_ZH.md`](../ms_swift_title_mask/TRAINING_PLAN_ZH.md) | 下一轮可执行命令；所有路径必须显式设置 |
| [`../ms_swift_title_mask/scripts/README.md`](../ms_swift_title_mask/scripts/README.md) | 当前脚本、历史数据 builder 与旧评测工具分类 |

## 数据实现文档

以下文件仍用于解释构造代码，但不单独决定训练准入：

- [`pmc_title_silver_pipeline_2026-08-23.md`](pmc_title_silver_pipeline_2026-08-23.md)：PMC 标题规则和正文 quarantine 事实。
- [`archive/legacy/ms_swift_from_source_2026-08-23.md`](archive/legacy/ms_swift_from_source_2026-08-23.md)：旧 4,524-row 快照的详细命令记录。
- [`../ms_swift_title_mask/data_assets/STATUS_ZH.md`](../ms_swift_title_mask/data_assets/STATUS_ZH.md)：旧 ZIP payload 说明；Git clone 不包含 payload。

## 历史记录

- [`archive/legacy/README.md`](archive/legacy/README.md)：旧架构、早期 ms-swift 方案、长度研究和历史 run 索引；不可直接照其中命令启动训练。

## 不进入 Git 的内容

PDF、页图、训练 JSONL、模型、adapter 和日志仍放在仓库外。仓库只保存构造/审计脚本、数量与 SHA
契约、聚合报告及少量明确标注的结构化失败输出。新工具默认拒绝把输出写进源数据目录，并且不会解压
READoc PDF 到工程中。
