# 2026-09-01 MinerU Title Prior Train Fit 验收

## 1. 结论

本次对两套带 MinerU Title Prior 的 READoc 16K checkpoint 做训练数据拟合抽测。抽样来自本次
实际训练使用的 `train.jsonl`，固定抽取 6 条样本；Base、Full-CE 和 Title-weighted 三路共
18 条请求，`18/18` 成功，`0` 条错误或空响应。每个 trained 方案各完成 `6/6` 请求。

两套方案都只能判定为 `partial_learning`，不能判定为 `full_overfit_pass`：少数样本有标题或
局部内容对齐迹象，但总体正文行召回接近 0，长/中样本仍出现过生成，且输出协议从训练 target
的 Markdown/grounding 混合形态发生变化。两套方案均未通过稳定 overfit 验收，也不能因为
129-PDF 评测请求全部成功而进入部署。

## 2. Run 与数据

| 项目 | 内容 |
|---|---|
| Base model | `$MODEL_PATH` |
| Full-CE checkpoint | `$UOCR_ROOT/output/readoc-heading-prior-16k-full-ce/v0-20260831-173734/checkpoint-212` |
| Title-weighted checkpoint | `$UOCR_ROOT/output/readoc-heading-prior-16k-title-weighted/v0-20260831-173734/checkpoint-212` |
| train data | `$UOCR_ROOT/data/readoc-heading-prior-16k/train.jsonl` |
| train rows / SHA256 | 1697 / `31f58ddc03bb10ac716f776a894a8659d8235201bfded42f6f689b5ce8841b41` |
| validation rows / SHA256 | 94 / `f160895c2c87ecb3aeb166f86106f1e2af8a07cfffd774c0d9eacb6707244fef` |
| probe runner | `ms_swift_title_mask/scripts/probe_train_fit.py`，进程内直接推理，无 HTTP 服务端口 |
| code revision | `a1a9aea4327b5fa830d008fda1658d3485ff9a63`；工作树含本轮之前的用户未提交修改 |

每条样本使用 JSONL 中原有的 prompt 和图片，prompt 为带 `<image>Multi page merge.` 及
MinerU 标题提示的 Title Prior 输入，没有在探针中额外添加 `<PAGE>` 或 assistant 前缀。

## 3. 抽样清单

| JSONL 行号 | 样本 ID | 页数 | target 字符数 | target token 数 |
|---:|---|---:|---:|---:|
| 702 | `readoc_github_38921393` | 2 | 1015 | 259 |
| 276 | `readoc_github_222982416` | 3 | 3828 | 939 |
| 184 | `readoc_github_255792625` | 5 | 7885 | 1987 |
| 1077 | `readoc_arxiv_physics9901006` | 8 | 13903 | 3948 |
| 1026 | `readoc_arxiv_1910.11437` | 8 | 25499 | 5498 |
| 693 | `readoc_arxiv_1510.03284` | 14 | 62339 | 11839 |

确定性抽样清单和完整 target/output 原文见：

~~~text
$UOCR_ROOT/evaluation/train_fit/readoc-heading-prior-16k-full-ce-vs-title-weighted-20260831/samples.jsonl
$UOCR_ROOT/evaluation/train_fit/readoc-heading-prior-16k-full-ce-vs-title-weighted-20260831/responses.jsonl
$UOCR_ROOT/evaluation/train_fit/readoc-heading-prior-16k-full-ce-vs-title-weighted-20260831/markdown/
~~~

## 4. 聚合结果

指标均为 6 条样本的算术平均；`severe` 是至少命中一个过生成或复读诊断的样本数。

| 模型 | 请求 | 文本相似度 | target 行召回 | target 标题召回 | 重复行比例 | 协议 | severe |
|---|---:|---:|---:|---:|---:|---|---:|
| Base | 6/6 | 0.4584 | 2.19% | 0.00% | 29.7% | grounding 4、mixed 2 | 3 |
| Full-CE + Title Prior | 6/6 | 0.1010 | 0.00% | 2.78% | 14.8% | markdown 4、mixed 2 | 2 |
| Title-weighted + Title Prior | 6/6 | 0.1245 | 1.03% | 4.17% | 7.9% | markdown 4、mixed 2 | 2 |

探针配置：`max_length=16384`、`temperature=0`、`no_repeat_ngram_size=35`、
`ngram_window=1024`、`attn_implementation=eager`；单页参数为 `image_size=640`、
`crop_mode=False`，多页由原生 `infer_multi` 使用 `image_size=1024`。

## 5. 样本级现象

| 行号 | Full-CE | Title-weighted |
|---:|---|---|
| 702（2 页） | 文本相似度 `0.006`，输出/target `35.69x`，Markdown，过生成 | 文本相似度 `0.012`，输出/target `18.20x`，Markdown，过生成 |
| 276（3 页） | 文本相似度 `0`，输出/target `6.73x`，Markdown，过生成 | 文本相似度 `0`，输出/target `6.73x`，Markdown，过生成 |
| 184（5 页） | 文本相似度 `0.088`，mixed，未命中严重阈值 | 文本相似度 `0.001`，输出/target `2.99x`，Markdown |
| 1077（8 页） | 文本相似度 `0.508`，标题召回 `16.7%`，mixed | 文本相似度 `0.196`，行召回 `5.2%`、标题召回 `16.7%`，mixed |
| 1026（8 页） | 文本相似度 `0.002`，Markdown | 文本相似度 `0.002`，标题召回 `8.3%`，Markdown |
| 693（14 页） | 文本相似度 `0.002`，Markdown | 文本相似度 `0.536`，行召回 `1.0%`，mixed |

长样本 `693` 在 Title-weighted 路线中出现较高字符串相似度，但 target 行召回仍只有 `1.0%`，
不能把单个样本的相似片段当作完整拟合。Full-CE 在 `1077` 上有局部标题对齐，但其它样本
没有形成稳定的正文学习证据。两套方案在短样本 `702`、`276` 上都显著过生成，因此不满足
“多数样本对齐 target 且无阻断性退化”的 overfit 条件。

## 6. 最终判定

| 方案 | Train Fit | Benchmark | 最终结论 |
|---|---|---|---|
| `readoc-heading-prior-16k-full-ce` | `partial_learning` | 129-PDF 正式结果 `22.7157 / 0.3923`，低于主 Base | 未通过 overfit 和泛化验收 |
| `readoc-heading-prior-16k-title-weighted` | `partial_learning` | 129-PDF 正式结果 `24.8699 / 0.3940`，低于主 Base | 未通过 overfit 和泛化验收 |

129-PDF 的完整 benchmark 分数、Title Prior 标题构造和生成诊断见：

~~~text
docs/evaluation_readoc_heading_prior_2026-09-01.md
$UOCR_ROOT/evaluation/heading_prior_20260901_129/
~~~
