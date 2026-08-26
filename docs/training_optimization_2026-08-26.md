# Unlimited-OCR 后训练问题诊断与下一轮方案

更新时间：2026-08-26（Asia/Shanghai）

状态：**当前训练暂停；旧 natural-union 配方、title-only 和 title-weighted 均不继续放大。**
本文是下一轮训练决策的唯一事实源。旧数据快照仍可复现，但不能继续作为默认训练集。

## 1. 结论

现有问题不是“标题权重没有调好”，而是训练数据域、完整 Markdown 质量、实验归因和评测协议同时失控：

1. 训练目标几乎全是英文，而 8 组失败 GT 全是中文；训练后模型相对 Base 出现明显能力遗忘。
2. 旧 train mix 按 assistant 字符计 `82.32%` 来自 PMC；同一 1,331 个 PMC 文档又以 full/single
   两种形式各出现一次。
3. PMC 标题 audit 被错误扩大成完整正文准入。builder 没有消费 `content_quarantine.jsonl`，并把
   所有 full/single 行标成 `SILVER_ACCEPTED`。
4. PMC full 只是逐页字符串用空行拼接，没有跨页续段、重复表头或表格结构合并；“无 `<PAGE>`”不等于
   “已经语义 merge”。
5. Full CE 与 Title-weighted 不只差标题 1:2。custom loss 与 native CE 在 gradient accumulation
   的长短样本归一化口径不同，现有对照混入第二个变量。
6. 旧评测用不同的 Base/trained 多页 prompt，生成上限统一给 32K，且只看字符数和重复行；它不是严格 A/B。
7. 所有输入统一 no-crop 1024 且视觉侧冻结，无法靠 decoder 标题 loss 补回扫描件、小字、公式和复杂表格像素。

下一轮顺序固定为：

```text
E0: 冻结 benchmark 和 Base retention gate
 -> R0: clean READoc-only Full CE
 -> R1: 加中文/原生能力 replay
 -> S10: quarantined PMC <=10% target budget
 -> T: uniform custom CE vs trusted-title weighting
 -> V: single-base vs single-gundam 视觉 bake-off
```

任何一阶段未通过停止条件，不进入下一阶段。

## 2. 已验证证据

### 2.1 旧训练数据组成

对仓库外现存的 `uocr_readoc_pmc_silver_mix_v1/train.jsonl` 用
`validate_training_recipe.py` 重新扫描：

| 指标 | 结果 |
|---|---:|
| train rows | 4,073 |
| READoc target chars | 36,214,339（17.68%） |
| PMC target chars | 168,588,930（82.32%） |
| PMC full/single 重复文档 | 1,331 |
| 中文字符占全部 target | 464 / 204,803,269（约 0.00023%） |
| 相对 Markdown 图片链接 | 7,958；其中 PMC 7,956 |
| XML image tags | 6,873；全部来自 PMC（HTML 注释内标签不计） |
| 已知坏 LaTeX 模式 | 6,151；全部来自 PMC |
| `content_review_status` | 4,073 行全部缺失 |

原始数据也支持这个域判断：READoc 2,233 篇 GT 中没有中文文档；PMC 1,486 份
`middle.json` 只有极少数出现中文字符。当前 8 组失败 GT 则全部为中文业务/科技文档。

### 2.2 原始数据全量只读审计

使用 `scripts/data/audit_raw_sources.py --strict-pdf-probe` 对用户给出的两个原始数据根目录重新审计。输出写在
仓库外 `/Users/guofengjiao/Documents/uocr_raw_audit_20260826_codex/`，没有解压 READoc ZIP，也没有复制 PDF、
Markdown 或训练 JSONL 进 Git。

| 数据 | 审计结果 |
|---|---|
| READoc | 2,233 篇、22,290 PDF 页；arXiv/GitHub 为 1,009/1,224 篇；PDF 与 GT 集合完全对应 |
| READoc 语言/结构 | 0 篇含 CJK；6 篇无 ATX heading；62 篇无文件末尾换行 |
| READoc PDF | `2210.15829` 在 strict parser 下因重复 `/Rotate` 失败，`strict=False` 可正常读取 96 页；应记 warning，不应丢文档 |
| PMC | 1,486 篇、26,690 页；PDF 与 `middle.json` 页数全部一致；22 篇共 35 个空 block 页面 |
| PMC 资源 | 23,024 次图片资源引用；按文档去重后的 23,023 个路径全部缺失于当前 PMC 目录 |

审计产物为 3.0 MB 的聚合/逐文档 manifest：`summary.json`、2,233 行 READoc JSONL 和 1,486 行 PMC
JSONL；SHA-256 分别为 `e63c19c2...a9a7c`、`0fbcad1d...b782c4`、`dc8a2a0d...7e81d`。它们留在仓库外，
文档只记录结论和校验摘要。

原始数据能重建候选规则和 source manifest，但不能独立恢复旧 1,552 READoc 准入、PMC 人工决定及精确
3,936/588 快照，因为 accepted decision manifest 不在原始数据根目录。没有这些 manifest 时只能声称“重新构造”，
不能声称“逐 SHA 复原”。

### 2.3 PMC 准入越界

PMC 原始 audit 已记录：

| 内容风险 | 文档 | 次数/页 |
|---|---:|---:|
| `MISSING_IMAGE_ASSET` | 1,486 | 23,024 |
| `JOINED_METADATA` | 904 | 3,310 |
| `BAD_LATEX` | 355 | 2,552 |
| `EMPTY_PAGE` | 22 | 35 页 |
| `IMAGE_ONLY_TABLE` | 45 | 178 |

但 `build_pmc_silver_jsonl.py` 不读取 quarantine，仍逐篇调用 `render_pmc_document()`，生成一条 full
和一条 single，并统一标注 title `SILVER_ACCEPTED`。这个状态只证明标题规则被确定性冻结，不证明正文、
表格、公式、图片引用或跨页 merge 已验收。

### 2.4 失败输出重算

`scripts/evaluation/analyze_outputs.py` 对同 8 个 PDF 的 24 条结果重新计算：

| 模型 | 严重失败 | 文本相似度均值 | 平均重复行 | 平均 output/GT | 协议 |
|---|---:|---:|---:|---:|---|
| Base | 1/8 | 0.554 | 8.9% | 2.00 | 8/8 grounding |
| Full CE | 6/8 | 0.146 | 51.3% | 2.45 | Markdown/mixed/unclassified |
| Title-weighted | 7/8 | 0.134 | 57.8% | 2.43 | Markdown/grounding/unclassified |

这里的相似度是 NFKC/空白规范化后的 `SequenceMatcher`，不是 OmniDocBench NED；绝对值不能当论文指标，
但在同一输入/GT 上足以确认两条训练臂相对 Base 大幅回退。完整逐条结果见失败包内
`analysis_2026-08-26.md`。

## 3. 根因优先级与可证伪实验

| 优先级 | 根因 | 证伪方法 |
|---|---|---|
| P0 | 英文专训导致中文/原生能力遗忘 | 比较 Base 与 0.25/0.5/1 epoch checkpoint 的中英文分桶指标；加入中文 replay 后回退是否消失 |
| P0 | PMC 正文污染和 82% target 主导 | READoc-only R0 与旧 mix 同超参对照；随后只加 clean PMC S10 |
| P0 | 页拼接冒充跨页 merge | 对 repeated-header、续段和跨页表格单独标 merge relation；不含 relation 的样本不得宣称训练 merge |
| P0 | 旧 loss 归一化混入 A/B | 新 `uniform_ce` 已改用 native accumulation token 分母；仍须做单/双卡、accumulation=4 的 10-step loss/gradient parity |
| P1 | prompt/输出协议不一致 | Base/adapter 均跑 `parsing` 与 `merge` 两种 prompt 的 2x2 矩阵 |
| P1 | EOS/生成上限失控 | 记录 EOS、cap、重复中止原因；按 GT token 预算设置离线安全 cap，cap hit 计失败 |
| P1 | 1024 global 视觉瓶颈 | 单页 Base/Gundam 零训练 A/B；多页不直接实现 multi-gundam |

## 4. E0：先冻结 benchmark

训练前必须建立互不重叠的固定集合：

- 英文：READoc arXiv/GitHub，按长度、公式、代码、表格分桶。
- 中文 retention：扫描件、原生 PDF、表格、公式、双栏、标准/合同；不得从最终 129-PDF test 派生训练行。
- 跨页：续段、重复表头、跨页表格、长文；分别记录是否真的需要 merge relation。
- 视觉：小字/低清扫描与正常电子 PDF 分桶。

每个 checkpoint 至少报告：文本 NED、TEDS、公式 CDM、heading text F1、level/order、纯 Markdown
协议率、output/GT、EOS rate、cap-hit、重复块率。现有 Python analyzer 只提供快速诊断；正式论文口径需接
OmniDocBench/TEDS/CDM 实现。

硬停止条件：

- 请求/文件/页数/prompt 完整率不是 100%；
- 中文主指标相对 Base 回退超过 1 个绝对百分点；
- severe repetition 或 cap-hit 超过 1%；
- 目标为纯 Markdown 时 protocol-valid 低于 98%；
- 任一主桶退化超过 2 个点，即使平均分上涨也停止。

离线 raw generation 和生产 guard 必须分开报告。不要用强重复惩罚掩盖模型退化后再声称训练已修复。

## 5. R0：clean READoc-only Full CE

先从现有 1,552 条 reviewed READoc full 构造 R0；显式排除 1 条含不可学习相对图片资源路径的 GitHub row，
得到 train/val/test `1410/71/70`，不加入 PMC single/full，不做 title weighting。exclusion 必须写进 recipe report，
不允许直接改 target。第一轮保持模型范围不变以隔离数据变量：84-module decoder LoRA、rank 16、alpha 32、
dropout 0.05、R-SWA 128、SDPA、bf16、32K fit、不截断、不 packing。

超参只做小范围 A/B：

| 参数 | 候选 |
|---|---|
| learning rate | `2e-5`、`5e-5` |
| weight decay | `0.01` |
| beta2 | `0.95` |
| warmup ratio | `0.03` |
| epoch | 最多 1；保存 0.25/0.5/0.75/1.0 epoch |

checkpoint 不按 eval loss 单独选择。优先满足 Base retention gate，再看 READoc 主指标。若 0.25 epoch 已发生
明显中文回退，停止并直接进入 replay 数据构造，不继续跑满。

## 6. R1：中文/原生能力 replay

仅靠用户给出的 READoc 和 PMC 无法完成这一步，因为它们几乎没有中文。必须新增与最终 test 文档级去重的
中文完整 Markdown GT；不允许把 129-PDF evaluation 或 8 个失败 GT 回灌训练。

replay 按 assistant target 预算占 `20%-30%`，不按 row 数；覆盖扫描、表格、公式、双栏和中文标准/报告。
先固定 R0 最佳 checkpoint、LR 和顺序，只改变 replay。若 READoc 指标回退超过 1 点或中文仍低于 Base，
则当前数据不足，不能通过继续加 epoch 解决。

## 7. S10：重新准入 PMC

PMC 只能作为辅助数据：

1. 正文必须有独立 `content_review_status=CONTENT_ACCEPTED`，不能复用 title status。
2. 删除/重建缺失 asset URL、坏 LaTeX、粘连 metadata、空页和 image-only table target。
3. 同一 PMCID 每个 split/epoch 只选 full、window 或 single 之一，不能 full+single 全量重复。
4. full/window 必须有真实续段/表格 relation；单纯逐页空行拼接不标 merge。
5. 用 `build_recipe_mix.py` 将 PMC target 字符预算限制到 10%；通过后最多试 20%。

`validate_training_recipe.py --recipe pmc_s10` 会拒绝超过 12%、重复 PMCID、缺少 content acceptance 或仍含
已知序列化 artifact 的数据。若跨页桶不改善或任一 retention 桶回退超过 1 点，不进入 S20。

## 8. T：标题 loss 只做后置消融

先比较：

- `full_ce`：native ms-swift CE；
- `uniform_ce`：custom loss 路径，正文/标题/EOS 全部 1；
- `title_weighted`：只在 trusted heading 上尝试 1.25/1.5/2.0；
- EOS 权重单独做 1/4 A/B，不能和标题权重同时首跑。

新 `uniform_ce` 和 `uocr_title_weighted_token_mean` 都使用 ms-swift 的 accumulation-level
`num_items_in_batch` 分母；历史复现入口仍分别保留 title-only 的 active-token mean 和旧 title-weighted
的逐 micro-batch active-weight mean，避免追溯改写旧实验定义。新实验不得调用这两个历史 loss。
先做 native Full CE / uniform CE 的 10-step parity。只有 heading text/level/order
至少提升 2 点，且正文 NED/TEDS 回退小于 0.5 点、早停不增加超过 1 点，标题权重才进入候选。
title+EOS-only 不再用于完整 Markdown OCR；若任务改为只输出标题树，应另建标题 JSON/Markdown 任务。

## 9. V：视觉路线

先在单页同一模型上零训练比较 `single_base` 与 `single_gundam`，按小字、扫描、公式和表格分桶。如果 Gundam
有稳定收益，再单独训练/评估 crop adapter，并检查遗忘。stock Unlimited-OCR/ms-swift 不支持真正
`multi_gundam`；在单页 Pareto 和 2-4 页窗口均未证明必要前，不改多页 forward。

## 10. 工具与数据边界

```text
scripts/data/audit_raw_sources.py
  READoc ZIP + PMC PDF/middle.json -> 只读 source manifest/summary

ms_swift_title_mask/scripts/build_recipe_mix.py
  外部 split JSONL -> 按 target 字符代理预算构造实验 mix；正式 tokenizer 分桶后复核 token 比例

ms_swift_title_mask/scripts/validate_training_recipe.py
  readoc_r0/replay_r1/pmc_s10/trusted_title/legacy composition gate；trusted_title 只接受 HUMAN_ACCEPTED

scripts/evaluation/analyze_outputs.py
  已保存 responses + GT -> 快速退化报告
```

这些脚本不需要把 PDF、页图或训练 JSONL 放入 Git。旧 4,524-row/3,936-fit 快照的精确复现仍见
`data_reconstruction_pipeline_2026-08-26.md`，但下一轮只从新 recipe 输出启动。

## 11. 明确停用/保留

| 项目 | 决策 |
|---|---|
| 旧 READoc+PMC full+PMC single natural union | 停用；仅 `legacy_20260823` 复现 |
| title+EOS-only 完整 Markdown 训练 | 停用 |
| 旧 title-weighted 1:2 checkpoint | 保留失败证据，不续训 |
| Full CE | 保留为基线，但先换 clean recipe |
| full decoder/full LM | 暂停；数据量和归因都不支持 |
| multi-gundam | 暂停；stock forward 不支持 |
| FlexAttention/fused CE | 32K SDPA 已跑通，不触发 |
| 2026-08-25 服务/并行评测脚本 | 历史复现工具，不是当前入口 |
