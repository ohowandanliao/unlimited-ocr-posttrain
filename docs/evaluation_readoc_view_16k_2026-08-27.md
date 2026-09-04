# 2026-08-27 READoc 评测记录

## 评测范围与可复现性

这是 READoc-1706 Full-CE 和 Title-weighted 方案的最终评测记录，评测在隔离的 GPU 服务器上进行。
本分析使用全量运行聚合、按文档统计的得分差异和自动重复模式检测；没有逐一人工检查每个输出文件。

输入：

~~~text
源 PDF: $EVAL_PDF_ROOT
真值（groundtruth）: $EVAL_GT_ROOT
OmniDocBench: $OMNIDOCBENCH_ROOT
~~~

运行产物：

~~~text
$UOCR_ROOT/evaluation/readoc-view-16k_all129/
~~~

本次运行包含 129 个 PDF 和 387 条模型请求：Base、Full CE checkpoint-213、Title-weighted checkpoint-213。
387 条请求全部成功，没有请求错误或空响应。保存的模型响应没有做修正、去重、截断，也没有在不同输出协议之间
进行转换。OmniDocBench 使用原始预测结果；AI Builder 只在评分时执行其要求的 grounding wrapper 解析，
不会修正响应内容。

远端评测环境已重新确认：Python 3.13.13，`mmeval==0.2.1`。

~~~text
$UOCR_ROOT/evaluation/tooling/omni-eval-venv
~~~

## Base 口径

本报告的 Base 是未训练模型在统一 READoc 评测入口下的正式主基线，多页使用
`<image>Multi page parsing.`，结果为 OmniDocBench Total `53.3645`、AI Builder Overall
`0.8271`。本报告中的 Full-CE/Title-weighted 是训练模型，多页使用训练契约中的
`<image>Multi page merge.`；不能把未训练 Base 改成 merge 来“统一 prompt”。

夏桢 Title-mask 的历史 badcase 评测另有一个 `merge` Base：它使用不同的直接推理入口、
单页图像处理和多页 `ngram_window`，结果为 `51.7663 / 0.7442`。该结果用于解释评测 diff，
不属于本报告的主基线，也不能替换下面表格中的 `base`。两边 AI Builder Overall 算法相同，
都是 `(Text accuracy + Table accuracy + Reading-order accuracy + Title accuracy) / 4`。
完整差异见 `wxz_readoc_title_mask_training_2026-08-29.md` 和
`inference_service_2026-08-24.md`。

第一次直接调用 OmniDocBench 失败，原因是环境中缺少 `mmeval`。随后已在上述环境中成功重新运行三套评分。
成功重跑的日志文件位于结果目录中，分别为：`omnidocbench_base_gtpdf_remote_rerun.log`、
`omnidocbench_full-ce_gtpdf_remote_rerun.log`、`omnidocbench_title-weighted_gtpdf_remote_rerun.log`。

## OmniDocBench 结果

| 模型 | Total |
|---|---:|
| base | 53.3645 |
| READoc-1706 / Full-CE | 44.1397 |
| READoc-1706 / Title-weighted | 43.3568 |

## AI Builder 详细指标

| 指标 | base | full-ce | title-weighted |
|---|---:|---:|---:|
| 文本准确率（Text accuracy） | 0.7995 | 0.7918 | 0.7804 |
| 表格准确率（Table accuracy） | 0.8109 | 0.7780 | 0.7796 |
| 阅读顺序准确率（Reading-order accuracy） | 0.9351 | 0.9278 | 0.9178 |
| 标题准确率（格式归一/grounding 内容匹配） | 0.7630 | 0.7432 | 0.7292 |
| 标题准确率（原始 ATX Markdown，仅协议诊断） | 0.000018 | 0.000018 | 0.002649 |
| TextEdit（越低越好） | 0.2005 | 0.2082 | 0.2196 |
| 表格 TEDS（Table TEDS） | 0.5547 | 0.5248 | 0.5197 |
| 表格 TEDS-S（Table TEDS-S） | 0.5744 | 0.5479 | 0.5468 |
| Read OrderEdit（越低越好） | 0.0649 | 0.0722 | 0.0822 |
| 公式编辑距离（Formula Edit distance，越低越好） | 0.7533 | 0.9924 | 0.9994 |
| 公式编辑准确率（Formula Edit accuracy） | 0.2467 | 0.0076 | 0.0006 |
| FormulaCDM | 0.7816 | 0 | 0 |
| AI Builder Overall（四项算术平均） | 0.8271 | 0.8102 | 0.8017 |

标题指标需要分开解读：原始响应主要使用 `<|det|>title ...` grounding 标签，而不是 ATX Markdown 的 `#` 标题。因此 Overall 使用格式归一后的 grounding 标题内容匹配分；该匹配忽略 heading level，并对标题数量差异进行惩罚。

| 标题诊断指标 | base | full-ce | title-weighted | 是否计入 Overall |
|---|---:|---:|---:|---|
| grounding 标题内容匹配（格式归一） | 0.7630 | 0.7432 | 0.7292 | 是 |
| AI Builder ATX `Title accuracy`（原始协议） | 0.000018 | 0.000018 | 0.002649 | 否 |

第一行按照实际使用的 grounding 协议评估标题内容，并计入本报告的 Overall。第二行只用于说明原始响应的协议形态，
不计入 Overall。

## 与 Base 对比

下面是相对于 Base 的 AI Builder 四项均值及分项绝对得分差异，单位为百分点（`pp`）：

| 指标 | full-ce - base | title-weighted - base |
|---|---:|---:|
| AI Builder Overall（四项算术平均） | -1.70 pp | -2.54 pp |
| 文本准确率（Text accuracy） | -0.77 pp | -1.91 pp |
| 表格准确率（Table accuracy） | -3.30 pp | -3.13 pp |
| 标题准确率（格式归一/grounding 内容匹配） | -1.98 pp | -3.39 pp |
| 表格 TEDS（Table TEDS） | -2.99 pp | -3.50 pp |
| 表格 TEDS-S（Table TEDS-S） | -2.65 pp | -2.76 pp |
| 阅读顺序准确率（Reading-order accuracy） | -0.73 pp | -1.73 pp |
| 公式编辑准确率（Formula Edit accuracy） | -23.91 pp | -24.61 pp |
| FormulaCDM | -78.16 pp | -78.16 pp |

最大回退来自公式能力，其次是表格结构。对 25 个具名 benchmark case 的分项诊断显示，回退主要集中在以下类别：

| 案例组 | 数量 | 主要回退 |
|---|---:|---|
| 1 表格/单元格 | 4 | Text -11.22/-13.48 pp |
| 2 表格结构/公式 | 5 | Table -8.26/-8.26 pp；TEDS -9.10/-9.10 pp |
| 3 跨页表格 | 4 | Table -9.28/-14.07 pp；TEDS -9.60/-13.68 pp |
| 4 图表/扫描/示意图 | 7 | Title-weighted Text -20.81 pp；Reading -16.39 pp |
| 5 版式/水印 | 5 | Table -28.86/-23.10 pp；Formula -72.49 pp |
| 其他业务文档 | 104 | Table -0.78/-1.79 pp |

按分项和原始输出检查，典型退化案例为：

- Full-CE：`5-2表旁水印`、`5-1双栏表格3`、`2-3多层嵌套表头2`。
- Title-weighted：`5-2表旁水印`、`4-1统计图1`、`4-3扫描1`、`2-3多层嵌套表头2`。

其他 104 个业务文档没有显示出普遍性回退；下降主要集中在复杂版式/水印、图表/扫描、公式和表格结构案例。

## 重复与生成退化

重复扫描只对原始响应做诊断性归一化：比较可见内容时忽略 grounding 坐标/标签 wrapper，但原始响应保持不变。
归一化后的重复行比例达到至少 10% 的结果包括：Base 为 9/129、Full CE 为 7/129、Title-weighted 为 10/129。
这个阈值只是诊断信号，不表示每一个重复表头都是生成失败。

明确的信号包括：

- `蓝德_办公电脑配置标准_2.pdf`：三个模型都重复生成同一个 3 行语义块两次；重复语义内容约占 94.3%。
- `5-2表旁水印.pdf`：Full CE 和 Title-weighted 的原始响应字节级完全一致；生成了 1337 个日期，日期中的日号达到 907，
  这是明显的数字循环。
- `4-1统计图1.pdf`：Title-weighted 只输出一个页标记，却生成了 1088 个年份项，年份达到 5447。
- `3-2跨页分页切断.pdf`：Title-weighted 的归一化重复行比例为 10.2%，其中一个块最多出现 12 次。
- `4-2段落中带「图x-x」.pdf`：Full CE 和 Title-weighted 的归一化重复行比例分别为 21.4% 和 19.0%。
- `4-1统计图3.pdf`：Base/Full CE/Title-weighted 的归一化重复行比例分别为 19.7%、16.7% 和 20.8%。
- `4-3扫描2.pdf`：Title-weighted 的归一化重复行比例为 16.5%；Full CE 主要表现为数字/年份过度生成。

本轮没有复现此前在 `3-4长表跨三页以上` 和 `4-3扫描2` 中观察到的 92% 至 98% 整页重复，但局部重复块和数字循环
仍然存在。

## 产物索引

每个模型的 `metrics.csv`、`metrics.json`、`per_doc_scores.json`、`report.md`、`title_grounding_result.json` 和
`formula_cdm_result.json` 均位于 `agentbuilder/{base,full-ce,title-weighted}/` 下。原始请求记录位于 `responses.jsonl`；
自动诊断结果位于 `repeat_scan.json` 和 `repeat_scan.md`。结果目录中原有的 `overall` 字段按旧加权实现生成，
本报告不引用该字段，统一采用上述四项算术平均和格式归一后的标题分。
