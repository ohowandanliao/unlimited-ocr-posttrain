# 2026-09-01 READoc 129-PDF MinerU Title Prior 正式评测记录

> **当前状态（2026-09-02，评测已完成）**：第 7 节测试已全部执行并验收通过
（`status=complete`、`expected=258`、`successful=258`、`failed=0`，258 条 prompt 与
manifest 逐条一致，四类数量 73/36/18/2 × 2 模型精确匹配）。正式结果见**第 8 节**；
第 4-5 节旧批数字仅作历史审计。prompt 口径唯一标准见
`docs/evaluation_prompt_matrix_2026-09-01.md`（含 `max_length=20480` 变更记录）。

## 1. 范围与结论

本页记录两套加入 MinerU 标题提示（Title Prior）的 READoc 16K checkpoint，以及上一批已删除的
评测尝试：

- `readoc-heading-prior-16k-full-ce`
- `readoc-heading-prior-16k-title-weighted`

固定评测集为 129 个 PDF，使用同一份 PDF/GT 集合。上一批没有重复评测已确认的主 Base，
实际完成两套训练模型各 129 条，共 `258/258` 成功、`0` 条失败；但其逐文件 prompt 覆盖不完整，
因此不满足本轮 Title Prior 对照的输入条件。修正后的正式评测仍然是两套训练模型各 129 条，
不是 `129 x 3 = 387` 条。

旧批次的审计数字是：Full-CE 的 OmniDocBench Total / AI Builder Overall 为
`40.5503 / 0.6749`，Title-weighted 为 `29.6696 / 0.6662`。主 Base 的既有结果为
`53.3645 / 0.8271`，本轮直接复用；旧数字不能替代修正输入后的正式结果。

## 2. 输入标题的构造

评测输入标题来自 91 机器上的 MinerU Markdown 目录：

~~~text
$MINERU_OUTPUT_ROOT/markdown
~~~

构造结果：

| 项目 | 数量 |
|---|---:|
| PDF/Markdown 映射行数 | 129 |
| 含 ATX 标题的 Markdown | 91 |
| 不含标题的 Markdown | 38 |
| 原始标题行 | 663 |
| 过滤后保留标题行 | 622 |
| 按文档去重 | 21 |
| 纯数字标题 | 9 |
| 列表标记标题 | 11 |

过滤规则是 fenced-code-aware 的 ATX 提取、删除列表标记标题、删除纯数字标题、按文档去重。
这批标题是从 MinerU 结果确定性抽取的结构提示，不是人工校正后的最终标题；prompt 已明确要求
模型根据图像核验并修正标题内容、层级、顺序和文档结构。没有标题的 38 个文档仍然参加评测，
其 prompt 不附加标题段。

本轮使用的完整 prompt manifest：

~~~text
$UOCR_ROOT/evaluation/input_manifests/heading_prior_20260901/prompts.all_pages.jsonl
$UOCR_ROOT/evaluation/input_manifests/heading_prior_20260901/summary.all_pages.json
~~~

该 manifest 覆盖 129 个 PDF、372 页：单页 109 个（其中 73 个有标题提示、36 个无标题提示），
多页 20 个（其中 18 个有标题提示、2 个无标题提示）。无标题文件仍然参加评测，但只使用对应
页数的标准 prompt。

训练数据也使用同一 Title Prior 变体：train 1697 行，validation 94 行，test 94 行；train
SHA256 为 `31f58ddc03bb10ac716f776a894a8659d8235201bfded42f6f689b5ce8841b41`，validation
SHA256 为 `f160895c2c87ecb3aeb166f86106f1e2af8a07cfffd774c0d9eacb6707244fef`。

## 3. Base 口径

正式主 Base 是未训练模型，使用 `<image>Multi page parsing.`，主 Base 结果为：

- OmniDocBench Total：`53.3645`
- AI Builder Overall：`0.8271`
- Text accuracy：`0.7995`
- Table accuracy：`0.8109`
- Reading-order accuracy：`0.9351`
- Title accuracy：`0.7630`
- FormulaCDM：`0.7816`

两套训练模型使用训练契约中的 `<image>Multi page merge.`，并在 prompt 中附加本轮构造的
MinerU 标题提示。由于模型状态和输入 prompt 不同，下面的主 Base 行是正式参考基线；本轮不把
Base 重新跑一遍，也不将它伪装成与训练模型完全相同的 prompt。

夏桢历史 Base 的 `51.7663 / 0.7442` 使用 `Multi page merge.`、不同评测入口、图像处理和
解码参数，只能作为历史评测 diff，不能替代主 Base，也不用于本轮正式泛化结论。

## 4. 旧输入批次审计结果（已删除，不作为正式结果）

以下表格只保留已删除旧批次的审计数字，便于解释为什么不能直接沿用。该批虽有 258 条成功响应，
但单页请求没有全部套用逐文件 Title Prior manifest；正式结果必须使用第 2 节的 `all_pages` manifest。

Title accuracy 使用格式归一后的 grounding title 内容匹配分，不使用原始 ATX Markdown 格式
分。标题匹配忽略 heading level，并对标题数量差异进行惩罚。结果保留四位小数，原始完整值
保存在各模型的 `metrics.csv` 和 `metrics.json`。

| 模型 | 新请求 | OmniDocBench Total | Text accuracy | Table accuracy | Reading-order accuracy | Title accuracy | AI Builder Overall | FormulaCDM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 主 Base（既有结果） | 复用 | 53.3645 | 0.7995 | 0.8109 | 0.9351 | 0.7630 | 0.8271 | 0.7816 |
| Full-CE + Title Prior（旧审计） | 129/129 | 40.5503 | 0.4006 | 0.8045 | 0.8376 | 0.6569 | 0.6749 | 0.7761 |
| Title-weighted + Title Prior（旧审计） | 129/129 | 29.6696 | 0.4019 | 0.7567 | 0.8441 | 0.6619 | 0.6662 | 0 |

Overall 严格按
`(Text accuracy + Table accuracy + Reading-order accuracy + Title accuracy) / 4` 计算。

### 4.1 相对正式主 Base

以下 `pp` 使用主 Base 文档中公布的四位小数计算，细微差异受 Base 展示精度影响：

| 指标 | Full-CE + Title Prior - Base | Title-weighted + Title Prior - Base |
|---|---:|---:|
| OmniDocBench Total | -12.8142 | -23.6949 |
| AI Builder Overall | -15.22 pp | -16.09 pp |
| Text accuracy | -39.89 pp | -39.81 pp |
| Table accuracy | -0.65 pp | -5.42 pp |
| Reading-order accuracy | -9.75 pp | -9.10 pp |
| Title accuracy | -10.61 pp | -10.11 pp |
| FormulaCDM | -0.55 pp | -78.16 pp |

Full-CE 的 FormulaCDM 为 `0.7761`，接近主 Base 的 `0.7816`；Title-weighted 的
FormulaCDM 为 `0`，且公式编辑距离为 `1.0`。两套模型的主要回退来自 Text accuracy；
Title-weighted 另外有明显表格结构和公式回退。

## 5. 旧输入批次标题和生成协议诊断（仅审计）

129 个文档的 GT 中有标题的文档数为 88，共 581 个 GT 标题。Full-CE 预测到 69 个文档的
grounding 标题，共 193 个；Title-weighted 预测到 71 个文档，共 253 个。两套均有 1 个
缺失预测。原始 ATX 标题诊断分分别为 `0.0030382` 和 `0.0046393`，不计入 Overall。

| 模型 | Grounding | Markdown | Mixed | Unclassified | Severe 文档 | Generation cap | 平均输出/GT | 平均 heading F1 | 重复行比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-CE + Title Prior | 112 | 10 | 6 | 1 | 13 | 2 | 2.20 | 0.312 | 5.4% |
| Title-weighted + Title Prior | 111 | 10 | 6 | 2 | 15 | 1 | 2.03 | 0.312 | 7.4% |

协议仍以 grounding 为主，但复杂长文档存在过生成、数字循环和重复行。代表性问题包括：

- Full-CE 在 `4-4示意图.pdf`、`5-1双栏表格1.pdf` 命中 generation cap 或过生成信号；
- Title-weighted 在 `1-3单元格内换行1.pdf`、`3-1跨页重复表头.pdf`、`4-1统计图3.pdf` 等文档
  出现较高重复比例；
- Title-weighted 在 `5-2表旁水印.pdf`、`4-1统计图1.pdf`、`4-3扫描1.pdf` 等复杂文档中
  也出现严重退化信号。

这些诊断来自原始 response 的自动分析；response 没有在评分前去重、截断、改协议或人工修正。

## 6. 已删除旧产物与正式保留项

为避免下一 session 误用旧 prompt 的输出，已删除以下评测目录：

- `evaluation/inference_eval_20260831_heading_prior_all129/`
- `evaluation/inference_eval_20260831_heading_prior_fullce_weighted_all129/`
- `evaluation/inference_eval_20260831_heading_prior_fullce_weighted_all129_clean/`
- `evaluation/inference_eval_20260831_heading_prior_fullce_weighted_all129_with_mineru_titles/`
- `evaluation/inference_eval_20260831_heading_prior_fullce_weighted_all129_with_mineru_titles_filtered/`

以下内容保留并构成本页正式结果：正确的 `prompts.all_pages.jsonl` manifest、129 个 PDF/GT、
两个 checkpoint、两套方案的 Train Fit 记录、主 Base 既有结果，以及第 8 节的正式推理和评分目录。

模型 checkpoint：

~~~text
$UOCR_ROOT/output/readoc-heading-prior-16k-full-ce/v0-20260831-173734/checkpoint-212
$UOCR_ROOT/output/readoc-heading-prior-16k-title-weighted/v0-20260831-173734/checkpoint-212
~~~

Base model 为 `$MODEL_PATH`。正式评测使用两路端点，动态队列每个端点串行处理；PDF 按
144 DPI 渲染，最终参数为 `max_length=20480`、`temperature=0`、`no_repeat_ngram_size=35`、
单页 `ngram_window=128`、多页 `ngram_window=1024`。

## 8. 正式评测结果（2026-09-01 推理 / 09-02 评分，本页唯一正式结果）

### 8.1 执行记录

- 推理目录：`evaluation/inference_eval_20260901_heading_prior_fullce_weighted_all129_all_pages/`
  （258/258 complete、0 失败、无重复行、无 error 行；外层 prompt 与 `response.prompt` 均与
  manifest 逐条一致；runner note 记录了 `max_length 32768->20480` 的协议变更）。
- 评分目录（2026-09-02 完成）：`evaluation/heading_prior_20260901_129/`（pred 为原始 `response.text`，
  未去重、未截断、未做协议转换；`run_eval.py` + OmniDocBench md2md + 格式归一标题分 +
  FormulaCDM，`--allow-missing` 与 2026-08-27 轮口径一致）。
- 生成参数：`image_size=1024`、`crop_mode=False`、`no_repeat_ngram_size=35`、单页
  `ngram_window=128`、多页 `ngram_window=1024`、`max_length=20480`、`temperature=0`。
  `max_length` 由 32768 下调至 20480（2026-09-01 数据决策：GT 中位数 1065 字符、最长合法
  输出约 17.7K token，cap 对合法输出零影响，只截退化复读尾巴；详见 prompt 标准文档）。

### 8.2 结果（对照主 Base `53.3645 / 0.8271`）

| 指标 | 主 Base | Full-CE (checkpoint-212) | Title-weighted (checkpoint-212) |
|---|---:|---:|---:|
| OmniDocBench Total | 53.3645 | 22.7157 | 24.8699 |
| Text accuracy | 0.7995 | 0.2951 | 0.2820 |
| Table accuracy | 0.8109 | 0.6475 | 0.5798 |
| Reading-order accuracy | 0.9351 | 0.6099 | 0.6243 |
| Title accuracy（格式归一） | 0.7630 | 0.0485 | 0.0536 |
| FormulaCDM | 0.7816 | 0.0000 | 0.2846 |
| AI Builder Overall | 0.8271 | 0.3923（-43.48 pp） | 0.3940（-43.31 pp） |

### 8.3 生成诊断

| 模型 | 重复行≥10% | 重复行≥40% | grounding 格式输出 | 空输出 | 命中 20480 cap | 欠生成（<0.3×GT） |
|---|---:|---:|---:|---:|---:|---:|
| Full-CE | 54/129 | 48/129 | 74/129 | 0 | 0 | 5 |
| Title-weighted | 61/129 | 56/129 | 77/129 | 0 | 0 | 6 |

### 8.4 结论

1. 两套 Title Prior 模型相对主 Base 大幅回退（Overall 约 -43 pp），与 Train Fit 的
   `partial_learning`（target 行召回 0-1%、标题召回 2.8-4.2%、severe 2/6）一致。
2. 输出协议被标题段扰动：带壳文件近 60% 跳出 Base 原生 grounding 协议（grounding 本身是健康协议，见第 9 节），标题分崩至 5% 左右；复读普遍
   （重复行≥40% 的文档 48-56/129）。
3. `max_length=20480` 在最终数据上零命中（无 cap 截断行），说明回退不是截断造成的。
4. 不部署；本页第 4-5 节的旧审计数字与第 7 节执行计划一并归档为历史记录。

### 8.5 同日完成的相关评测（同一协议）

夏桢两套 Title-mask 用同一入口、同一 prompt 标准重测（推理目录
`evaluation/inference_eval_20260901_wxz_prior_all129/` 与
`evaluation/inference_eval_20260901_wxz_noprior_all129/`，评分目录
`evaluation/wxz-readoc-1697-title-prior-mask_20260901_129/` 与
`evaluation/wxz-readoc-1706-title-mask_20260901_129/`）：

| 模型 | 输入 | OmniDocBench Total | AI Builder Overall | 相对主 Base |
|---|---|---:|---:|---:|
| READoc-1706 / Title-mask | merge（无 prior） | 28.6202 | 0.5122 | -31.49 pp |
| READoc-1697 + Title Prior / Title-mask | merge + 标题段 | 24.4528 | 0.4500 | -37.71 pp |

带 prior 反而比无 prior 低 `6.22 pp`：旧 badcase 中 prior 的正收益是协议缺陷
（带 prior 模型从未收到标题段）造成的假象。旧 badcase 产物已删除。

## 9. Case 级归因：为什么 Full-CE / Title-weighted + Title Prior 掉 40+ 分（2026-09-02）
> 自包含汇总（含上一轮统一协议结果 + 本轮 case 归因）：`analysis_all_regress_attribution_2026-09-02.md`

方法：剥除 `<|det|>` 标签后按行集合计算 F1 与标题集合匹配（行级 F1 受分块/换行影响绝对值
偏低，只用于同文件横向对比）；以 manifest 的 91 个带标题段文件 / 38 个无标题段文件分组，
并与同一批文件上的 Base、readoc-view-16k Full-CE（下称 fc827）直接对比。
分析脚本镜像于 `.remote-patch-stage/analysis_why_drop*.py`。

### 9.0 白话版

- **壳（标题段）是什么**：推理时在 prompt 末尾附加的一段文字——"The following headings
  are extracted from the MinerU parsing result..." + MinerU 标题列表 + "### Final corrected
  Markdown"。因为 heading-prior 训练集每一行的输入都带这段，评测时按训练契约同样喂入。
- **问题是什么**：模型一看到这段输入，输出就从 Base 的正常块状格式（grounding）切换成
  退化的 markdown 标题刷屏/复读，正文内容报废。
- **为什么掉这么多**：129 个文件里 91 个按契约喂了壳（全部崩，行级 F1 0.35→0.04），
  38 个没喂壳的与 Base/8-27 完全一样好（0.311 vs 0.309/0.312）。掉的分几乎全部来自
  那 91 个。
- **是训练数据问题吗**：target 内容没问题（与 view16k 同源同格式，MinerU 标题对 GT
  标题覆盖 86%）；问题在训练没教会"看到壳该怎么输出"（Train Fit 行召回 0-1%），壳在
  推理时成了纯干扰。
- **是评测问题吗**：口径无误——喂壳符合训练-推理一致性，评分管线对所有模型相同，
  Base 基线未动。

### 9.1 同文件对照（决定性证据）

91 个带壳文件上的行级 F1：

| 模型（推理输入） | F1 |
|---|---:|
| Base（无壳） | 0.362 |
| fc827（无壳） | 0.351 |
| wxz_np title-mask（无壳） | 0.285 |
| hyx Full-CE（带壳） | 0.042 |
| hyx Title-weighted（带壳） | 0.063 |
| wxz_p title-mask（带壳） | 0.115 |

同一批文件掉到 1/8~1/3。而 38 个无壳文件上 hyx Full-CE F1=0.311 ≈ fc827 0.309 ≈
Base 0.312，输出 38/38 为正常 grounding 协议、复读率 0.027。**先验训练本身没有损伤
模型解析能力；掉分几乎全部由推理时标题段输入触发。**

### 9.2 机制：标题段把输出从 grounding 协议拖入退化 markdown

- 91 个带壳文件中 hyx Full-CE 有 55 个跳出 grounding 协议（42 markdown + 13 other）。
  markdown 模式复读率 0.52（Title-weighted 0.59），标题行占输出字符约 40%
  （如 `### 最终版本` 连续重复、`# 5.1.1 软件开发` 连环编号刷屏）。
- 即使保持 grounding 协议的 36 个带壳文件，F1 也只有 0.079（同批文件 fc827 0.351）；
  标题块从 Base 的 5.0 个/文件（命中 GT 标题 2.6 个）降到 2.1 个/文件（命中 0.1 个）。
- multi+shell（18 个，输入与训练分布完全一致：merge + 标题段）反而最差
  （per-doc Overall 代理 0.1465）——排除"单页 prompt OOD"解释：是没学会，不是没见过。
- 触发媒介是壳内标题列表与 `### Final corrected Markdown` 标记：训练契约是
  shell→markdown target，但 212 步 rank16 LoRA 从未收敛到能生成目标格式
  （Train Fit 行召回 0-1%），于是壳既没被当作提示利用，也没学会对应协议，只剩扰动。

### 9.3 训练数据取证与两个被排除的解释

- 四套训练数据 target 完全同分布：`#` 行 mean 15.3-15.4、p50 12、94.1% 以标题行开头、
  长度 p50 同为 10176 字符；target 均为 markdown、均无 grounding 标签。prior16k 与
  view16k 唯一差异就是输入里的标题段（1706 vs 1697 行，同源语料）。
- 不是 prior 内容太脏：MinerU 标题对 GT 标题 recall 0.860 / precision 0.780（p50 均为 1.0），
  Base 自己输出的标题与 prior 列表重合 0.65，盲复制天花板有 0.86。
- 也不是"模型盲抄 prior"：hyx 输出标题仅 0.15 来自 prior 列表——它是被带偏，不是被抄坏。
- loss 权重无关：Full-CE 0.3923 vs Title-weighted 0.3940，退化形态一致。

### 9.4 对第 8.4 节结论的修正

8.4 第 2 条"输出协议未收敛（58-60% 输出仍为 grounding 格式）"表述方向反了：grounding
是 Base 原生健康协议（Base 与 8-27 两套均近 100% grounding 且高分），异常的是带壳文件
近 60% 跳出该协议进入退化 markdown。其余结论不变。

### 9.5 对下一轮的含义

- Title Prior 方向没有在"内容"层面被证伪（先验质量 86%/78% 足够好），被证伪的是
  "212 步 rank16、只见 shell 的训练能让模型学会利用提示"。若继续该方向：训练集必须
  混入无壳样本以保持协议回退能力；先在 Train Fit 上验证 shell 输入可复现 target，
  再谈 benchmark。
- 标题分崩（Overall -43.5pp 中 Title 占 -17.9pp）与协议翻转同源：先修输出协议，
  再谈标题利用。
- 附：wxz_p 的 title 角色块从 Base 的 5.0 个/文件降到 0.4 个/文件（title 角色近乎消失、
  出现 image_caption 混用），其官方 Title 0.0906 的直接原因是标题角色/结构坍塌，
  而非标题文本缺失（集合级标题召回仍有 0.609）。
