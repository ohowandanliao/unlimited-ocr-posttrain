# 历史文档索引

本目录只保存已经失效、但仍有复盘价值的设计和实验记录。它们不能作为训练、数据构造或评测入口；
当前决策以 [`../../training_optimization_2026-08-26.md`](../../training_optimization_2026-08-26.md) 和
[`../../../ms_swift_title_mask/TRAINING_PLAN_ZH.md`](../../../ms_swift_title_mask/TRAINING_PLAN_ZH.md) 为准。

| 文档 | 保留原因 | 当前状态 |
|---|---|---|
| [`DESIGN.md`](DESIGN.md) | 早期工程边界和 R-SWA 架构分析 | 设计已被当前 ms-swift 实现取代 |
| [`MODES.md`](MODES.md) | base/gundam、single/multi 能力边界 | 仅作模型能力背景 |
| [`ms_swift_comparison_2026-08-04.md`](ms_swift_comparison_2026-08-04.md) | 早期自研与 ms-swift 对照 | 版本和实验条件已过期 |
| [`ms_swift_data_pipeline_2026-08-18.md`](ms_swift_data_pipeline_2026-08-18.md) | 旧数据混合、长度和 loss 方案推导 | 配方不可直接执行 |
| [`ms_swift_from_source_2026-08-23.md`](ms_swift_from_source_2026-08-23.md) | 4,524-row 旧快照的完整重建命令 | 只用于复现旧实验和对账 |
| [`multipage_encoder_bakeoff_2026-07-14.md`](multipage_encoder_bakeoff_2026-07-14.md) | 多页视觉编码器路线研究 | bake-off 未进入当前训练队列 |
| [`multipage_feasibility_2026-07-16.md`](multipage_feasibility_2026-07-16.md) | 多页探针结果和失败边界 | 已停止放量 |
| [`ocr_vlm_dataset_deep_dive_2026-07-08.md`](ocr_vlm_dataset_deep_dive_2026-07-08.md) | 开源 OCR/VLM 数据集调研 | 不代表已批准数据依赖 |
| [`readoc_96page_training_options_2026-08-16.md`](readoc_96page_training_options_2026-08-16.md) | READoc 长度审计和容量分析 | 训练队列已被取代 |
| [`posttrain_three_schemes_2026-08-25.md`](posttrain_three_schemes_2026-08-25.md) | 三种 loss 的训练事实与三路失败评测 | 历史 run 总览，不是启动入口 |

被清理的逐回答 Markdown、分析 JSON 等派生文件均可由仓库内的结构化原始记录和脚本重新生成；
归档目录不再保存同一事实的逐日状态副本。
