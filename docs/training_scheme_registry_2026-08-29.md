# Unlimited-OCR 训练方案与双验收登记

更新时间：2026-09-02（Asia/Shanghai，统一协议 benchmark 完成后更新）

> **当前状态**（2026-09-01 晚）：9 套方案 Train Fit 均为 `partial_learning`。统一协议
> 129-PDF benchmark 已完成 6 套：READoc-1706 Full-CE/Title-weighted（2026-08-27）、夏桢两套
> Title-mask 与 hyx 两套 Title Prior（2026-09-01，统一 prompt 口径与 `max_length=20480`）。
> 6 套全部相对主 Base（`53.3645 / 0.8271`）回退，均不可部署。旧夏桢 badcase 与旧 Title Prior
> 审计数字已废弃，相关产物已删除。Prompt 唯一标准见 `docs/evaluation_prompt_matrix_2026-09-01.md`。

## 1. 登记口径

本表只登记已经实际训练并产生 LoRA 权重的方案，共 9 种。

- Base 是所有方案共用的未训练对照，不算训练方案。
- 除本表 9 种方案外，早期链路验证与未执行的计划阶段工具（recipe 阶梯构造器、
  smoke 测试、preflight/compat 检查）已于 2026-09-03 清理，不计入本表。
- 同一个 loss 在不同数据、prompt 或训练入口下是不同方案，不能只按 loss 名称合并。

每种方案必须维护两类互相独立的结果：

1. Benchmark：固定评测集上与 Base 的同输入对比，判断泛化和回归。
2. Train Fit：从该方案实际使用的 train 数据抽样推理，判断是否学会训练样本内容。

详细执行规则见：

~~~text
docs/posttrain_completion_sop_2026-08-28.md
~~~

## 2. 九种已训练方案

| scheme_id | 方案 | 数据/入口 | checkpoint | benchmark | Train Fit | 结论 |
|---|---|---|---|---|---|---|
| pmc-readoc-3551-title-mask | PMC+READoc-3551 / Title-mask（PMC 2184 + READoc 1367）；title+EOS-only | data/reviewed-title-32k/length_32k_v1/fit/train.jsonl | output/reviewed-title-mask-dual-32k/v0-20260823-201829/checkpoint-{400,443} | 未完成正式评分；未进入 75/387 部分结果 | `evaluation/train_fit/pmc-readoc-3551-20260829/report.md`；`partial_learning`；16/16 成功 | 有标题痕迹，但长样本复读/过生成；未通过 overfit |
| pmc-readoc-3551-full-ce | PMC+READoc-3551 / Full-CE（PMC 2184 + READoc 1367） | 同上 | output/reviewed-full-ce-dual-32k/v0-20260824-233410/checkpoint-{400,443} | 部分结果 75/387；无正式 OmniDocBench 总分 | `evaluation/train_fit/pmc-readoc-3551-20260829/report.md`；`partial_learning`；16/16 成功 | 有局部正文学习，但长样本重复；未通过 overfit |
| pmc-readoc-3551-title-weighted | PMC+READoc-3551 / Title-weighted（PMC 2184 + READoc 1367）；body=1/title=2/EOS=1 | 同上 | output/reviewed-title-weighted-dual-32k/v0-20260825-031657/checkpoint-{400,443} | 部分结果 75/387；无正式 OmniDocBench 总分 | `evaluation/train_fit/pmc-readoc-3551-20260829/report.md`；`partial_learning`；16/16 成功 | 短样本有对齐，但长样本复读；未通过 overfit |
| readoc-1706-full-ce | READoc-1706 / Full-CE | data/readoc-view-16k/train.jsonl | output/readoc-view-16k-full-ce/v0-20260826-232140/checkpoint-213 | evaluation/readoc-view-16k_all129；387/387；Omni Total 44.1397；AI Builder Overall 0.8102 | evaluation/overfit_train_sample_20260828；`partial_learning` | Benchmark 低于 Base；未通过稳定 overfit |
| readoc-1706-title-weighted | READoc-1706 / Title-weighted，body=1/title=1.5/EOS=1 | data/readoc-view-16k/train.jsonl | output/readoc-view-16k-title-weighted/v0-20260826-232140/checkpoint-213 | evaluation/readoc-view-16k_all129；387/387；Omni Total 43.3568；AI Builder Overall 0.8017 | evaluation/overfit_train_sample_20260828；`partial_learning` | Benchmark 低于 Base；未通过稳定 overfit |
| readoc-1706-title-mask | READoc-1706 / Title-mask，无 Title Prior | `$WXZ_READOC_ROOT/readoc_uocr/r0/views/train_le_16k_with_title.jsonl` | `$WXZ_OUTPUT_ROOT/readoc_title_mask_16k/` | evaluation/wxz-readoc-1706-title-mask_20260901_129；OmniDocBench Total 28.6202；AI Builder Overall 0.5122（-31.49 pp vs 主 Base）；2026-09-01/02 统一协议重测 | `evaluation/train_fit/readoc-1706-title-mask-20260829/report.md`；`partial_learning`；8/8 成功 | 3/4 样本严重复读/过生成；未通过 overfit |
| readoc-1697-title-prior-mask | READoc-1697 + Title Prior / Title-mask | `$WXZ_READOC_ROOT/readoc_uocr_heading_prior/views/train_le_16k_with_title.jsonl` | `$WXZ_OUTPUT_ROOT/readoc_heading_prior_title_mask_16k/` | evaluation/wxz-readoc-1697-title-prior-mask_20260901_129；OmniDocBench Total 24.4528；AI Builder Overall 0.4500（-37.71 pp vs 主 Base）；2026-09-01/02 统一协议重测，评测输入含 MinerU 标题段 | `evaluation/train_fit/readoc-1697-title-prior-mask-20260829/report.md`；`partial_learning`；8/8 成功 | 相对无 Title Prior 更差（Overall 0.4500 < 0.5122；归因见 docs/analysis_all_regress_attribution_2026-09-02.md §7）；未通过 overfit |
| readoc-heading-prior-16k-full-ce | READoc-1697 + MinerU Title Prior / Full-CE | `$UOCR_ROOT/data/readoc-heading-prior-16k/train.jsonl`；1697 行；SHA256 `31f58ddc03bb10ac716f776a894a8659d8235201bfded42f6f689b5ce8841b41` | `$UOCR_ROOT/output/readoc-heading-prior-16k-full-ce/v0-20260831-173734/checkpoint-212` | evaluation/heading_prior_20260901_129；OmniDocBench Total 22.7157；AI Builder Overall 0.3923（-43.48 pp vs 主 Base）；2026-09-01 正式重测（正确 manifest） | `evaluation/train_fit/readoc-heading-prior-16k-full-ce-vs-title-weighted-20260831`；总计 18/18，Full-CE 6/6；平均相似度 `0.1010`、target 行召回 `0.0%`、标题召回 `2.8%`、重复行比例 `14.8%`、severe `2/6`；详见 `docs/train_fit_heading_prior_2026-09-01.md` | 129-PDF 相对主 Base 回退；`partial_learning`，未通过 overfit |
| readoc-heading-prior-16k-title-weighted | READoc-1697 + MinerU Title Prior / Title-weighted | `$UOCR_ROOT/data/readoc-heading-prior-16k/train.jsonl`；1697 行；SHA256 `31f58ddc03bb10ac716f776a894a8659d8235201bfded42f6f689b5ce8841b41` | `$UOCR_ROOT/output/readoc-heading-prior-16k-title-weighted/v0-20260831-173734/checkpoint-212` | evaluation/heading_prior_20260901_129；OmniDocBench Total 24.8699；AI Builder Overall 0.3940（-43.31 pp vs 主 Base）；2026-09-01 正式重测（正确 manifest） | `evaluation/train_fit/readoc-heading-prior-16k-full-ce-vs-title-weighted-20260831`；总计 18/18，Title-weighted 6/6；平均相似度 `0.1245`、target 行召回 `1.0%`、标题召回 `4.2%`、重复行比例 `7.9%`、severe `2/6`；详见 `docs/train_fit_heading_prior_2026-09-01.md` | 129-PDF 相对主 Base 回退；`partial_learning`，未通过 overfit |

READoc-1706 的三种 loss 方案使用同一批 1706 个 READoc 样本；Full-CE/Title-weighted 使用派生
recipe，Title-mask 使用历史训练入口。三者后来使用同一 129-PDF 协议评测，但训练入口和超参数不同，
不能把跨入口分差单独归因给 loss。READoc-1697 + Title Prior 是加入 Title Prior 输入后的 1697 条变体。

本轮新增的两套 `readoc-heading-prior-16k-*` 使用 READoc-1697 + MinerU Title Prior 数据，
checkpoint 为 `checkpoint-212`。129-PDF 评测从 `$MINERU_OUTPUT_ROOT/markdown`
确定性提取标题提示：129 行中 91 行有标题、38 行无标题，原始 663 行过滤后保留 622 行。
2026-09-01 已用 `prompts.all_pages.jsonl` 完成两套 trained model 各 129 条、共 `258/258` 的
正式评测（统一 prompt 口径、`max_length=20480`），验收通过；主 Base 沿用已有 `53.3645 / 0.8271`，
未重复跑 Base。结果与诊断见 `docs/evaluation_readoc_heading_prior_2026-09-01.md` 第 8 节。
Train Fit 的抽样、原始响应和最终判定见 `docs/train_fit_heading_prior_2026-09-01.md`。

夏桢两套已于 2026-09-01 用统一协议（统一服务入口、带 prior 模型输入真实标题段、
`max_length=20480`）重测：无 Title Prior Overall `0.5122`（-31.49 pp），带 Title Prior
输入 Overall `0.4500`（-37.71 pp），带 prior 反而比无 prior 低 `6.22 pp`。旧 badcase 分数
（`0.6635`、`0.6801`）因协议缺陷（带 prior 模型未收到标题段、入口与解码参数不同）已废弃，
产物已删除，不再作为任何结论的依据。

### Base 口径与评测 diff

正式主基线是未训练 Base 使用 `<image>Multi page parsing.` 的统一 READoc 入口，主 Base
结果为 OmniDocBench `53.3645`、AI Builder Overall `0.8271`。训练模型使用训练契约中的
`<image>Multi page merge.`；`merge` 不得用于未训练 Base 的正式主基线。

夏桢历史 Base 同样是未训练 Base、没有 LoRA，但使用 `Multi page merge.`，并通过
`eval_badcase.py` 直接推理；其单页处理为 `base_size=1024, image_size=640, crop_mode=True`，
多页 `ngram_window=128`。主 Base 使用统一推理服务、单页 `image_size=1024, crop_mode=False`，
多页 `ngram_window=1024`。因此夏桢 Base 的 `51.7663 / 0.7442` 是历史评测 diff，不能替代
主 Base，也不能用于给四个方案的正式泛化结果重新计算相对差值。两边 AI Builder Overall
算法相同，均为四项准确度的算术平均；详细参数见 `inference_service_2026-08-24.md`。该 badcase 入口已于 2026-09-01 废弃，相关评测产物删除。

## 3. 方案与结果的目录关系

运行根目录：

~~~text
$UOCR_ROOT/
  output/       所有 checkpoint 和 adapter
  evaluation/   所有 benchmark、Train Fit、评分和诊断
  logs/         训练和评测日志
  service/      推理服务日志和 PID
~~~

新 run 推荐使用以下结构：

~~~text
evaluation/<scheme_id>/
  benchmark/
    manifest
    responses.jsonl
    metrics.json
    report.md
  train_fit/
    samples.jsonl
    responses.jsonl
    summary.json
    report.md
~~~

历史目录不强行搬迁；在本表中登记其实际路径即可。不要把不同 scheme 的 response 合并到同一个 JSONL，也不要用方案名相同但数据或 checkpoint 不同的旧目录覆盖新结果。

## 4. Benchmark 登记要求

每个方案的 benchmark 行必须回答：

- 输入 manifest 是否与 Base 完全相同；
- 请求是否全部成功、是否有空输出或 generation cap；
- trained 与 Base 的总体分数和主要子项差值；
- 复杂表格、公式、扫描件、长文档是否回退；
- Markdown、grounding、mixed 协议和重复/过生成信号；
- 评分使用的 checkpoint、代码 commit、服务端口和解码参数。

READoc 固定 benchmark 的详细事实源：

~~~text
docs/evaluation_readoc_view_16k_2026-08-27.md
evaluation/readoc-view-16k_all129/
~~~

READoc Title-mask 两种输入的详细事实源：

~~~text
evaluation/wxz-readoc-1706-title-mask_20260901_129/
evaluation/wxz-readoc-1697-title-prior-mask_20260901_129/
~~~

## 5. Train Fit 登记要求

每个方案的 Train Fit 必须使用该方案自己的 train JSONL、图片、prompt 和 checkpoint。

至少保存：

- 确定性抽样清单；
- Base 和 trained 的完整原始 response；
- target/output 字符数和 tokenizer token 数；
- 文本相似度、target 行召回、标题召回；
- 输出协议、generation cap、重复行和过生成；
- 至少一个短样本、一个中样本和一个长样本的人工原文检查。

Train Fit 的结论只能取以下三类：

| 结论 | 含义 |
|---|---|
| full_overfit_pass | 多数样本能对齐 target，输出稳定，没有阻断性退化 |
| partial_learning | 部分样本能对齐，但长短样本或协议/重复问题仍明显 |
| no_evidence_of_learning | trained 没有比 Base 提供可靠的 target 内容证据 |
| blocked_incomplete | 请求、数据、checkpoint 或评测不完整，暂时不能判定 |

partial_learning 不能写成“学会”，no_evidence_of_learning 也不能仅凭输出协议变化判定。

已完成的 Train Fit 结果：

~~~text
evaluation/overfit_train_sample_20260828/
~~~

对应报告：

~~~text
evaluation/overfit_train_sample_20260828/report.md
~~~

该报告对应 `readoc-1706-full-ce` 和 `readoc-1706-title-weighted` 两种方案，不能代表其他方案。其他方案的补测报告分别登记在各自的 `evaluation/train_fit/<run>/report.md`。

## 6. 每次训练结束后的登记顺序

1. 登记 scheme_id、run、checkpoint、train JSONL 和数据 SHA256。
2. 确认训练日志成功结束；服务使用的 adapter 必须指向本次 checkpoint。
3. 先跑该方案 train-fit 抽测，确认是否存在明显的内容学习证据和生成退化。
4. 再跑固定 benchmark，并与同一输入的 Base 对比。
5. 将两类结果写入本表对应行，并在结果目录保留 raw response。
6. 只有 benchmark 完整、Train Fit 已判定、阻断性问题已记录，才能给 checkpoint 标记可部署或继续实验。

## 7. 登记缺口

截至 2026-09-01，9 种方案均已有 Train Fit 抽测和明确结论：

- `partial_learning`：9/9；`full_overfit_pass`：0/9。
- 2 种 READoc-1706 方案同时有固定 benchmark 和 Train Fit，但两者均未通过稳定 overfit/泛化验收。
- 2 种 READoc Title-mask 输入有 badcase benchmark 和 Train Fit；两套 Train Fit 都存在明显复读/过生成。
- 3 种 PMC+READoc-3551 方案已有统一 Train Fit，但仍缺正式统一 OmniDocBench 总分；部分生成材料只能作为参照。
- 2 种带 MinerU Title Prior 的方案已有 6/6 Train Fit，均为 `partial_learning`；正确 manifest 的 258 条正式评测已完成（2026-09-02），两套均大幅回退，见 `evaluation/heading_prior_20260901_129` 与 `docs/evaluation_readoc_heading_prior_2026-09-01.md` §8。

以后每完成一个新 checkpoint，只新增或更新对应 scheme 的 Benchmark 与 Train Fit 结果链接，不再创建含义不清的总目录。
