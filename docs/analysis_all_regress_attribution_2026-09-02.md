# 全量回退归因分析：统一协议评测结果 + Case 级证据（2026-09-02）

> 状态：正式分析文档，自包含。整合两轮分析：
> ① 2026-09-01/02 统一协议 129-PDF 重测的结果与诊断（上一轮）；
> ② 2026-09-02 的 case 级归因：为什么连 Full-CE / Title-weighted 都掉 40+ 分（这一轮）。
> 关联文档：`evaluation_readoc_heading_prior_2026-09-01.md` §8/§9、
> `posttrain_weekly_report_2026-08-30.md` §3、`evaluation_prompt_matrix_2026-09-01.md`。

## 1. 要回答的问题

现象：6 套已完整评测的训练方案相对主 Base（OmniDocBench `53.3645` / AI Builder Overall
`0.8271`）全部回退。两个层次：

- **Q1**：Title-mask 类方案为什么崩（-31 ~ -38 pp）——正文不受监督，可以理解；
- **Q2**：**Full-CE / Title-weighted + Title Prior 为什么也崩（-43 pp）**——loss 是全量
  监督、数据和最接近 Base 的 readoc-1706 同源，理论上最稳的一套，掉得却最狠。

## 2. 白话结论（30 秒版）

1. **评测没有错**：258 条 prompt 与 manifest 逐条一致，评分管线与 Base/8-27 完全相同，
   Base 基线未动。给带 prior 模型喂标题段是训练契约要求，不喂才是错。
2. **训练数据的内容没有错**：heading-prior 数据集的 target 与 readoc-view-16k 同源同格式；
   MinerU 标题本身质量够好（对 GT 标题 recall 0.860 / precision 0.780）。
3. **错的是训练强度配不上输入**：212 步、LoRA rank16、每个训练样本输入都带标题段——
   模型从没学会"看到标题段该怎么输出"（Train Fit 行召回 0-1%）。
4. **推理时标题段成了纯毒药**：模型一看到标题段就跳出 Base 原生 grounding 输出协议、
   转入退化的 markdown 标题刷屏。129 个文件里 91 个按契约喂了标题段（全崩），38 个
   没喂的与 Base/8-27 一模一样好。掉的分几乎全部来自那 91 个。

## 3. 名词约定

- **标题段（下称"壳"）**：推理时附在 prompt 末尾的一段固定文字，逐字如下
  （标题行插在 `### MinerU headings` 之后）：

  ~~~text
  The following headings are extracted from the MinerU parsing result of the same document.
  Use them only as structural hints.
  Verify and correct the heading text, heading levels, ordering, and document structure according to the document images.
  Do not blindly copy the MinerU headings if they conflict with the document images.

  ### MinerU headings
  ...（MinerU 标题列表）

  ### Final corrected Markdown
  ~~~

  来源：heading-prior 训练集每行输入都带这段（训练契约），评测按同一契约喂回。
- **grounding 协议**：本模型家族的原生输出格式，逐块
  `<|det|>角色 [坐标]<|/det|>文本`（角色如 title/text/header/table）。**Base 的正常、
  健康输出**——Base 100% 走该协议并拿到 0.8271。
- **91/38 分组**：129 个评测文件中，MinerU 能解出标题 → 评测时喂壳（91 个：73 单页 +
  18 多页）；解不出 → 不喂壳（38 个，多为无标题的表格页）。这是一组现成的天然消融。

## 4. 上一轮：统一协议重测结果与诊断

### 4.1 分数（全部同一入口、同一评分管线）

| 方案 | OmniDocBench Total | Text | Table | Read | Title | Overall | vs Base |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base（正式主基线） | 53.3645 | 0.7995 | 0.8109 | 0.9351 | 0.7630 | 0.8271 | - |
| readoc-1706 / Full-CE | 44.1397 | 0.7918 | 0.7780 | 0.9278 | 0.7432 | 0.8102 | -1.70 pp |
| readoc-1706 / Title-weighted | 43.3568 | 0.7804 | 0.7796 | 0.9178 | 0.7292 | 0.8017 | -2.54 pp |
| wxz title-mask（无 prior） | 28.6202 | 0.4206 | 0.6949 | 0.8291 | 0.0592 | 0.5122 | -31.49 pp |
| wxz title-mask + prior | 24.4528 | 0.4048 | 0.5879 | 0.7460 | 0.0906 | 0.4500 | -37.71 pp |
| hyx prior / Full-CE | 22.7157 | 0.2951 | 0.6475 | 0.6099 | 0.0485 | 0.3923 | -43.48 pp |
| hyx prior / Title-weighted | 24.8699 | 0.2820 | 0.5798 | 0.6243 | 0.0536 | 0.3940 | -43.31 pp |

### 4.2 生成诊断（重复行≥10% / ≥40% / grounding 格式 / 空 / cap / 欠生成）

| 模型 | rep≥10% | rep≥40% | grounding | 空 | cap | 欠生成 |
|---|---:|---:|---:|---:|---:|---:|
| wxz 无 prior | 21 | 15 | 129 | 0 | 0 | 2 |
| wxz 带 prior | 65 | 52 | 54 | 0 | 0 | 3 |
| hyx Full-CE | 54 | 48 | 74 | 0 | 0 | 5 |
| hyx Title-weighted | 61 | 56 | 77 | 0 | 0 | 6 |

### 4.3 上轮结论

- 6 套全回退、9 套 `partial_learning`、无一可部署。
- 带 prior 反而比自己无 prior 差（wxz：-6.22 pp），旧 badcase 的 prior 正收益是协议缺陷
  （旧评测从未把标题段喂给带 prior 模型）造成的假象；旧产物已删除。
- 当时遗留问题：Q2 未解释（本文第 5 节回答）。

## 5. 这一轮：Case 级归因（回答 Q2）

### 5.1 方法

剥除 `<|det|>` 标签后按行集合计算 F1 与标题集合匹配（行级 F1 受分块/换行影响绝对值偏低，
**只用于同文件横向对比**）；按"是否喂壳"分组，并在同一批文件上跨模型对比。
脚本：`analysis_why_drop*.py`（本地 `.remote-patch-stage/` 镜像）；中间产物 `/tmp/case_analysis/`。

### 5.2 决定性证据：同文件对照

91 个带壳文件上的行级 F1：

| 模型（推理输入） | F1 |
|---|---:|
| Base（无壳） | 0.362 |
| readoc-1706 Full-CE（无壳） | 0.351 |
| wxz title-mask（无壳） | 0.285 |
| **hyx prior / Full-CE（带壳）** | **0.042** |
| **hyx prior / Title-weighted（带壳）** | **0.063** |
| **wxz title-mask + prior（带壳）** | **0.115** |

38 个无壳文件上：hyx Full-CE **0.311** ≈ readoc-1706 Full-CE **0.309** ≈ Base **0.312**，
输出 38/38 正常 grounding 协议、复读率 0.027。

**同一个模型：给壳就崩、不给壳和 Base 一样好。先验训练本身没有损伤模型解析能力；
掉分几乎全部由推理时喂入的标题段触发。**

### 5.3 机制：标题段把输出拖出 grounding 协议、进入退化 markdown

- 91 个带壳文件中 hyx Full-CE 有 55 个跳出 grounding 协议（42 markdown + 13 other）。
  markdown 模式复读率 0.52（Title-weighted 0.59），标题行占输出字符约 40%。
- 典型退化样例（`1-3单元格内换行1`，一页表格文档）：

  ~~~text
  table
  ### 最终版本
  ### 最终版本
  ### 最终版本        ← 同一行连续重复 8 次
  ...
  # 1.1 项目范围确认
  # 2.2 软件开发
  # 3.1 软件开发      ← 连环假编号刷屏
  ...
  ~~~

- 即使保持 grounding 协议的 36 个带壳文件，F1 也只有 0.079（同批文件 Full-CE 无壳
  0.351）；标题块从 Base 的 5.0 个/文件（命中 GT 标题 2.6 个）降到 2.1 个/文件
  （命中 0.1 个）。标题"被带偏"而非缺失。
- **multi+shell（18 个，输入=训练分布原样：merge + 标题段）反而最差**
  （per-doc Overall 代理 0.1465）——排除"单页 prompt OOD"解释：是没学会，不是没见过。
- 触发媒介：壳内标题列表 + `### Final corrected Markdown` 标记。训练契约是
  shell→markdown target，但 212 步 rank16 LoRA 从未收敛到能生成目标格式
  （Train Fit 行召回 0-1%），于是壳既没被当作提示利用，也没学会对应协议，只剩扰动。

### 5.4 被排除的解释

| 嫌疑 | 判定 | 证据 |
|---|---|---|
| 评测喂错 prompt / 评分口径不一致 | ❌ | 258 条 prompt 与 manifest 逐条一致；管线与 Base/8-27 相同 |
| 不该给带 prior 模型喂壳 | ❌ | 喂壳正是训练契约 |
| 训练 target 内容坏了 | ❌ | 四套数据 target 完全同分布（`#` 行 mean 15.3-15.4、p50 12、94.1% 起始标题行、长度 p50 同为 10176；均无 grounding 标签）；prior16k 与 view16k 唯一差异是输入里的壳 |
| MinerU 标题质量差 | ❌ | 对 GT 标题 recall 0.860 / precision 0.780（p50 均 1.0）；Base 自己的标题与 prior 重合 0.65，盲抄天花板 0.86 |
| 模型盲抄错误标题 | ❌ | hyx 输出标题仅 0.15 来自 prior 列表——是被带偏，不是被抄坏 |
| 单页 prompt 未见过（OOD） | ❌ | multi+shell 与训练分布一致，反而最差 |
| loss 权重（Full-CE vs weighted） | ❌ 无关 | 0.3923 vs 0.3940，退化形态一致 |

### 5.5 掉分构成（hyx Full-CE 的 -43.5 pp）

| 组件 | 贡献 |
|---|---:|
| Title（0.7630 → 0.0485） | **-17.9 pp** |
| Text（0.7995 → 0.2951） | -12.6 pp |
| Reading-order（0.9351 → 0.6099） | -8.1 pp |
| Table（0.8109 → 0.6475） | -4.1 pp |

标题崩与协议翻转同源：模型不再以 title 角色发标题块。wxz_p 更极端：title 角色块
5.0 → 0.4 个/文件（角色近乎消失、混入 image_caption），官方 Title 0.0906 的直接原因
是标题角色/结构坍塌，而非标题文本缺失（集合级标题召回仍有 0.609）。

### 5.6 消融阶梯总表（所有方案各归各位）

| 变量 | Overall | 机制 |
|---|---:|---|
| Base | 0.8271 | grounding 原生协议 |
| +LoRA（readoc-1706，无壳，Full-CE/weighted） | 0.8102 / 0.8017 | LoRA 未收敛但不扰动协议，≈无害（-1.7/-2.5 pp 来自多页 merge 行为） |
| +训练加壳（prior16k，Full-CE/weighted） | 0.3923 / 0.3940 | 壳在推理时翻转协议→崩；训练强度不足以学会用它 |
| +title-mask loss（wxz，无壳） | 0.5122 | body 无监督→多页内容/结构弱 |
| +title-mask 且加壳（wxz） | 0.4500 | 壳的伤害再叠加 |

### 5.7 对早先表述的一处修正

早先文档写"58-60% 输出仍为 grounding 格式"暗示 grounding 是异常——方向反了。
grounding 是 Base 原生健康协议（Base 100% grounding 拿 0.8271），异常是**带壳文件近
60% 跳出该协议**进入退化 markdown。其余结论不变。

## 6. 对下一轮的含义

1. Title Prior 方向**没有在"内容"层面被证伪**（先验质量 86%/78% 足够好），被证伪的是
   "212 步 rank16、全量带壳训练能让模型学会利用提示"。
2. 若继续该方向，按优先级：
   a. **构造**：壳里不出现 markdown 记号（标题列表去掉 `#` 前缀、末尾不用
      `### Final corrected Markdown`），降低"协议切换信号"强度；
   b. **配比**：训练集混入无壳样本，保住协议回退能力；
   c. **强度**：加大步数/rank，先在 Train Fit 上验证 shell 输入可复现 target
      （行召回显著大于 0），再谈 benchmark。
3. 标题分与协议翻转同源：先修输出协议，再谈标题利用。
4. 评测协议本身可信，可直接复用于下一轮；不建议再为"协议怀疑"重跑 Base。
5. 可选一锤定音实验：把 hyx prior checkpoint 按"无 prior 模型"协议（完全不喂壳）重测
   129 文件——按现有证据预测 Overall 回到 ≈0.81、与 8-27 Full-CE 打平；若如此，
   "壳有害"即闭环。需重拉一个推理服务，待确认后再排。

## 7. 三个追问的定责结论（2026-09-02 补充）

### 7.1 "是不是 prior 构造格式的问题？"

是**交互问题**（构造 × 训练没教会），不是格式单独的错。三条证据：

1. 训练与推理的壳逐字一致，不存在 train/eval 格式 mismatch；
2. 壳内容质量足够（对 GT 标题 recall 0.860 / precision 0.780）；
3. multi+shell 与训练分布完全一致（merge + 壳），却是最差组——排除"模型没见过这种格式"。

但构造确有一个客观性质：壳内含 markdown 记号（`#` 标题列表、`### Final corrected
Markdown`），是强**输出协议切换信号**；当训练没有教会模型执行这个切换时（212 步
rank16，行召回 0-1%），该信号在推理时必然触发退化。单独哪边都不致命：同样的弱训练
不给壳就是 8-27 的结果（-1.7 pp）。

### 7.2 "没有 title prior 的呢？"

五种配置的伤害机制各不相同：

| 配置 | 输出协议 | 主要伤害 | Overall |
|---|---|---|---:|
| Base | grounding ✓ | — | 0.8271 |
| 全量监督、无壳（8-27 两套） | grounding ✓（128-129/129 文件） | 多页 merge 不完美 + 公式能力归零（CDM 0.78→0） | 0.8102 / 0.8017 |
| title-mask、无壳（wxz_np） | grounding ✓（129/129，复读中位数 0） | 标题结构崩（title 块 5.0→2.3 个/文件，官方 Title 0.0592）+ 多页内容弱（multi F1 0.148 vs fc827 0.302） | 0.5122 |
| 全量监督、带壳（hyx 两套） | 60% 翻成 markdown | 协议翻转→全文崩 | 0.3923 / 0.3940 |
| title-mask、带壳（wxz_p） | 84% 翻成 markdown | 协议翻转 + 本来就弱 | 0.4500 |

"带不带壳"决定崩的方式（协议翻转型 vs 结构内伤型），"loss 监督什么"决定无壳时崩多少。

### 7.3 "能不能理解成 title mask + 壳有害、weighted / full-CE 好一点？"

不能压成一个维度，两轴定责：

- **轴 1（壳）**：对任何 loss 都必崩（-38 ~ -43.5 pp）。title-mask+壳（0.4500）甚至不是
  最差组合，全量监督+壳更差（0.3923 / 0.3940）——"有害的是壳"，不是某个 loss+壳组合。
- **轴 2（loss）**：只在**无壳**前提下可比——全量监督（-1.7/-2.5 pp）≫ title-mask
  （-31.5 pp）；Full-CE 与 Title-weighted 之间全程 ≤0.9 pp，属噪声级别，**不构成结论**。
- "好一点" = 没被壳毒到，不等于学到东西：8-27 两套 Train Fit 行召回仅 12-17%、
  公式归零，partial_learning，不可部署。
- 跨作者比较要克制：wxz（batch 1、lr 1e-4、1706 步）与 hyx（batch 8、lr 2e-5、212 步）
  训练入口不同，wxz_p 与 hyx_fc 之间的分差不能归因给 loss。

## 8. 事实源

- 正式分数与执行记录：`evaluation_readoc_heading_prior_2026-09-01.md` §8；
  本归因原文：同文档 §9（§9.0 白话版）。
- prompt 唯一标准与审计：`evaluation_prompt_matrix_2026-09-01.md`。
- 分析脚本：`analysis_why_drop.py`（v1-v3）、`analysis_why_drop_v4.py`，镜像于本地
  `.remote-patch-stage/`，服务器 `/tmp/`；中间产物 `/tmp/case_analysis/report*.txt`
  与 `cases/`（10 个跨模型 case 摘录）。
- 评测产物：`evaluation/heading_prior_20260901_129/`、
  `evaluation/wxz-readoc-1706-title-mask_20260901_129/`、
  `evaluation/wxz-readoc-1697-title-prior-mask_20260901_129/` 及对应
  `inference_eval_20260901_*` 推理目录。
