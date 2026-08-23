# ms-swift reviewed-title 训练决策（2026-08-23）

当前执行方案已随可上传 ZIP 的源码一起维护在：

- [`../ms_swift_title_mask/TRAINING_PLAN_ZH.md`](../ms_swift_title_mask/TRAINING_PLAN_ZH.md)
- [`../ms_swift_title_mask/README.md`](../ms_swift_title_mask/README.md)
- [`../ms_swift_title_mask/COMPATIBILITY.md`](../ms_swift_title_mask/COMPATIBILITY.md)

决策是三池 natural union（READoc full、PMC full、paired PMC strict-single）上的两条训练臂：同一数据顺序的
`reviewed-full-ce` 与 `reviewed-title-mask`。`HUMAN_ACCEPTED` 与规则冻结、完整可追溯的 `SILVER_ACCEPTED`
均可训练且 provenance 分开记录。预计 train 为 1,411+1,331+1,331=4,073 行，strict-single share 32.7%。
两组均由 ms-swift `swift sft` 执行，使用 84 模块 decoder-backbone LoRA、原生 Unlimited-OCR R-SWA、
base/no-crop 1024（同一进程混合 single/multi）、H100 单/双卡和完整 Markdown output；默认探测
`MAX_LENGTH=32768`，超长进入 overflow manifest，绝不截 target。

本文件只作为仓库索引，任何参数或数据契约冲突时，以 ZIP 内 `TRAINING_PLAN_ZH.md` 为准。
