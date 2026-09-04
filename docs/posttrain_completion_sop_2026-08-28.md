# 训练完成验收 SOP：Base 对比与 Train Fit 抽测

更新时间：2026-08-28（Asia/Shanghai）

## 1. 目的

每次新的 OCR 训练完成后，必须做两类验收：

1. **固定 benchmark 与 Base 对比**：确认训练后模型在未参与训练的固定评测集上是否变好、退化或只改变了输出协议。
2. **训练数据 Train Fit 抽测**：从本次训练实际使用的 train.jsonl 抽样，确认模型是否能从图像恢复训练 target。这是最直接的“模型有没有学会”的检查。

两类结果必须分开记录。benchmark 反映泛化和回归，Train Fit 反映训练集拟合；不能用其中一个替代另一个。

## 2. 验收前先登记 run

训练结束后先登记下面的信息，后续报告和目录都使用同一个 RUN_ID：

| 项目 | 必须记录 |
|---|---|
| checkpoint | 绝对路径、训练 run、step |
| base model | 绝对路径和版本/commit |
| train 数据 | train.jsonl 绝对路径、行数、SHA256 |
| validation/test 数据 | 路径、行数、SHA256 |
| 训练配置 | context、attention、dtype、LoRA、loss、seed、epoch |
| 推理配置 | prompt、max_length、temperature、重复抑制参数 |
| 代码 | posttrain 分支和 commit |

训练日志必须确认任务正常结束、checkpoint 完整、没有 OOM 或未处理异常。不要用旧 checkpoint 冒充本次 run。

当前运行目录约定：

~~~text
uocr-ms-swift-title-mask/
  output/       checkpoint 和 adapter
  evaluation/   benchmark、Train Fit 和诊断结果
  logs/         训练和评测日志
  service/      推理服务日志和 PID
~~~

训练数据、PDF、页图和模型权重不复制进 posttrain 仓库。

## 3. 推理服务检查

如果要启动服务，先确认训练进程已经结束；如果服务占用 GPU，启动训练前先核对 PID 并停止服务。

服务启动后，两个检查都必须通过：

~~~bash
curl http://127.0.0.1:<PORT>/health
curl http://127.0.0.1:<PORT>/v1/models
~~~

必须确认：

- Base、目标 checkpoint 对应的 trained route 都在模型列表中；
- trained route 的 adapter 路径就是本次 checkpoint；
- /infer 可用，不要误请求服务根路径；
- 单页和多页 prompt 与本次训练数据一致；
- 不自行添加 <PAGE> 或改写 assistant target。

## 4. 验收 A：固定 benchmark 与 Base 对比

### 4.1 固定输入

每个 run 使用同一份 benchmark manifest、同一批源 PDF/页图和同一份 groundtruth。当前固定评测集是 129 个 PDF，后续如更换集合，必须新建 manifest 并写明原因。

Base prompt 需要按模型状态固定，不能简单要求 Base 和 trained 使用同一个 prompt：

- 未训练 Base 的正式主基线使用 `<image>Multi page parsing.`。
- 训练模型使用训练数据对应的 prompt；当前 READoc 训练方案使用 `<image>Multi page merge.`。
- `Multi page merge.` 是训练模型的输入契约，不得用于未训练 Base 的正式主基线。
- 夏桢历史 badcase 的 `merge` Base 只能作为评测 diff 记录，不能替代 `parsing` Base。

同一份输入至少跑：

- unlimited-ocr-base
- 本次训练的 checkpoint

如果本次有多个 loss 或多个 checkpoint，则每个都跑，但不能把不同 run 的结果混在一张结果表里。

在同一个评测入口内，Base 和 trained 必须保持以下可比参数一致；prompt 按上面的模型状态和
训练契约固定：

- 图像渲染和 DPI；
- 单页/多页拆分；
- prompt 及其对应的模型状态必须登记，不能将 Base 的 `parsing` 改成 trained 的 `merge`；
- max_length；
- temperature；
- no_repeat_ngram_size 和 ngram_window；
- timeout、重试次数和错误计数规则。

响应原文不得去重、截断、改协议或手工修正后再评分。评分用的 wrapper 必须和原始响应分开保存。

### 4.2 必须产出的结果

至少保存：

~~~text
evaluation/<RUN_ID>/benchmark/
  manifest
  responses.jsonl
  <file>__<model>.md
  summary.md
  metrics.json
  metrics.csv
  repeat_scan.json
  repeat_scan.md
~~~

报告至少包含：

- 请求总数、成功数、错误数、空输出数；
- Base 和 trained 的总分及每个主要子指标；
- trained - base 的绝对差值，单位注明；
- 每个文档的得分差异，标出最大提升和最大回退；
- 输出协议（Markdown、grounding、mixed）；
- 输出长度、达到 generation cap 的数量；
- 重复行、数字循环、整页重复等退化信号；
- 公式、表格、标题、阅读顺序等重点能力的单独对比。

不要只报平均分。若平均分变化很小，但某一类表格、公式、扫描件或长文档严重回退，必须单独列出。

### 4.3 Base 对比的判定

| 观察结果 | 结论 |
|---|---|
| trained 总体和关键子项提升，且没有明显新增退化 | 通过，可继续扩大评测 |
| trained 与 Base 接近，但协议/重复/长度出现明显问题 | 功能未通过，先查生成配置和协议 |
| trained 总体下降，或复杂版式/公式/表格明显回退 | 训练回归，不进入部署 |
| 请求错误、空输出或 generation cap 未清零 | 评测不完整，不能下模型结论 |

分数提升不能抵消严重生成退化。对于 OCR，输出更长、标题格式变了或 route 生效，都不等于效果变好。

## 5. 验收 B：Train Fit 抽测

### 5.1 抽样原则

必须从本次实际训练使用的 train.jsonl 抽样，不能从 validation、test 或别的历史数据代替。

每次至少抽 6 条；推荐使用确定性分层抽样，覆盖：

- 短 target；
- 中等 target；
- 接近 context 上限的长 target；
- 少页数和多页数；
- 至少一个包含表格、公式或复杂 Markdown 结构的样本（数据中存在时）。

保存抽样清单：

~~~text
evaluation/<RUN_ID>/train_fit_probe/
  samples.jsonl
~~~

每条清单记录原始 JSONL 行号、样本 ID、图片路径、页数、target 字符数和 tokenizer token 数。先检查所有图片存在、target 未被截断，并确认样本 prompt 与训练 prompt 一致。

当前 READoc view 16K 数据的训练 prompt 是：

~~~text
<image>Multi page merge.
~~~

不要在抽测时自行加 page、<PAGE> 或新的 assistant 前缀。

### 5.2 请求方式

同一批样本至少请求：

- Base；
- 本次训练 checkpoint。

如果本次训练有 Full CE 和 Title-weighted 等多个方案，所有方案都请求同一批图片，并使用同一套解码参数。temperature 固定为 0；max_length 至少覆盖训练 context，但要记录实际值。

保存完整请求和响应：

~~~text
evaluation/<RUN_ID>/train_fit_probe/
  samples.jsonl
  responses.jsonl
  summary.json
  report.md
  markdown/
~~~

responses.jsonl 中保留 target 和 output 原文，方便复核。错误行必须明确记录，不能把错误请求当作“不学习”。

### 5.3 Train Fit 需要看的指标

对每个样本、每个模型记录：

- 请求是否成功、是否空输出；
- output 字符数 / target 字符数；
- normalized text similarity；
- exact line overlap、target line recall；
- heading text precision/recall/F1；
- 首个标题是否命中；
- 输出协议；
- generation cap；
- 重复行比例、最大重复次数和数字循环；
- output 是否在中途停止或异常延长。

至少人工查看三类原文：

1. 一个短样本的完整 output；
2. 一个中等样本的前后段；
3. 一个长样本的标题、正文中段和结尾。

### 5.4 “学会”的判定

Train Fit 通过必须同时满足：

- trained 请求全部成功且非空；
- 多数抽样样本的内容指标明显优于 Base，而不是只有输出长度变长；
- 训练 target 的首标题、主要 section 和若干正文行能在 trained output 中对齐；
- 没有大面积重复、数字循环、过早停止或 generation cap；
- 输出协议与本次训练目标一致，或协议差异已经明确解释。

下面的情况不能称为“学会”：

- 只出现了 Markdown 标题，但正文与 target 不重合；
- output 比 Base 长很多，但主要是重复；
- trained route 能返回结果，但内容仍是 Base 风格或 grounding 噪声；
- 只有一个短样本看起来相似，长样本全部失败；
- 通过调大 max_length 隐藏 generation cap。

Train Fit 通过也只说明模型具备在训练样本上的拟合能力，不能证明泛化；最终是否可用仍由 benchmark 与 Base 对比决定。

### 5.5 解码退化时的二次诊断

如果标准推理参数下出现明显复读或过生成，可以对同一批样本再做一次诊断请求，例如关闭重复抑制：

~~~text
no_repeat_ngram_size=0
~~~

该结果只能用于区分“模型没有学会”和“模型学到内容但解码提前进入循环”，不能替代标准配置的验收，也不能用于直接宣称线上效果。

## 6. 每次 run 的最终报告

建议在 evaluation/<RUN_ID>/report.md 汇总：

~~~markdown
# <RUN_ID> 训练完成验收

## Run
- checkpoint:
- train data:
- code commit:
- inference config:

## Train Fit
- sample count:
- Base:
- trained:
- 是否学会:
- 主要失败样本:

## Benchmark vs Base
- 请求完成率:
- Base score:
- trained score:
- trained - base:
- 主要提升:
- 主要回退:

## Decision
- [ ] Train Fit 通过
- [ ] Benchmark 完整
- [ ] Benchmark 相对 Base 通过
- [ ] 无阻断性重复/协议问题
- 最终结论:
~~~

只有两类验收都完成，并且阻断性问题已解释，才能把 checkpoint 标为可继续评测或可部署。所有报告都要写清楚“本次用的 checkpoint、数据、服务端口和解码参数”，避免下一次交接时把历史结果当成当前结果。
