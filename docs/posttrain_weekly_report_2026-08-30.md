# Unlimited-OCR Posttrain 周报：9 套方案对比、评测结果与结论

日期：2026-08-30（Asia/Shanghai）；更新：2026-09-02（Asia/Shanghai；统一协议评测 2026-09-01 晚启动、09-02 完成）
范围：截至 2026-09-01 已完成训练并产生 LoRA 权重的方案。本轮已用修正后的 `all_pages`
manifest 完成两套 Title Prior 模型的 258 条正式评测，并用同一协议补测了夏桢两套
Title-mask；旧夏桢 badcase 评测因协议缺陷（带 prior 模型从未收到 prior 输入）已删除，
其历史数字不再引用。Prompt 口径唯一标准见 `docs/evaluation_prompt_matrix_2026-09-01.md`。

## 一句话结论

本轮共登记 9 套训练方案。Train Fit 判定不变：`partial_learning=9/9`，`full_overfit_pass=0/9`。
2026-09-01 统一协议下的 129-PDF 正式 benchmark 显示：**6 套已完整评测的训练模型全部相对
主 Base（OmniDocBench `53.3645` / AI Builder Overall `0.8271`）大幅回退，无一套通过泛化
验收，均不可部署**。其中两套 READoc-1706 Full-CE/Title-weighted 仅小幅回退
（`-1.70 / -2.54 pp`）；两套 Title-mask（含 MinerU Title Prior 输入）与两套
READoc-1697+Prior 回退 `-31.49 ~ -43.48 pp`，伴随严重复读与标题协议未收敛。

新增加权结论：**在统一协议下，夏桢带 Title Prior 模型（输入真正收到 prior 段）反而比自己
的无 prior 版本差 `6.22 pp`**——历史 badcase 中观察到的 prior 正收益是评测协议缺陷造成的
假象，Title Prior 输入在当前训练方式下没有形成正贡献。

## 1. 方案总览

Base 是未训练模型，只作对照，不计入 9 套方案。其余链路验证与未执行的计划阶段
产物（工具与测试输出）已于 2026-09-03 清理，不在本报告范围。

| # | scheme_id | 方案 | 数据与输入 | loss / 关键差异 | 训练步数 | Benchmark 状态 | Train Fit 结论 |
|---:|---|---|---|---|---:|---|---|
| 1 | `pmc-readoc-3551-title-mask` | PMC+READoc-3551 / Title-mask | PMC 2184 + READoc 1367，共 3551 条 | 只监督标题和 EOS，正文权重为 0 | 443 | 不完整：75/387 抽测，无正式总分 | `partial_learning` |
| 2 | `pmc-readoc-3551-full-ce` | PMC+READoc-3551 / Full-CE | 同上 | 正文、标题、EOS 等权 | 443 | 不完整：75/387 抽测，无正式总分 | `partial_learning` |
| 3 | `pmc-readoc-3551-title-weighted` | PMC+READoc-3551 / Title-weighted | 同上 | body=1、title=2、EOS=1 | 443 | 不完整：75/387 抽测，无正式总分 | `partial_learning` |
| 4 | `readoc-1706-full-ce` | READoc-1706 / Full-CE | READoc 1706 条 | 正文、标题、EOS 等权 | 213 | 完整：129-PDF（2026-08-27），见第 3 节 | `partial_learning` |
| 5 | `readoc-1706-title-weighted` | READoc-1706 / Title-weighted | READoc 1706 条 | body=1、title=1.5、EOS=1 | 213 | 完整：129-PDF（2026-08-27），见第 3 节 | `partial_learning` |
| 6 | `readoc-1706-title-mask` | READoc-1706 / Title-mask（无 Title Prior） | READoc 1706 条 | Title-mask；无 Title Prior 输入 | 1706 | 完整：129-PDF 统一协议重测（2026-09-01/02），见第 3 节 | `partial_learning` |
| 7 | `readoc-1697-title-prior-mask` | READoc-1697 + Title Prior / Title-mask | READoc 1697 条 | Title-mask；增加 Title Prior 输入 | 1697 | 完整：129-PDF 统一协议重测（2026-09-01/02，含 prior 输入），见第 3 节 | `partial_learning` |
| 8 | `readoc-heading-prior-16k-full-ce` | READoc-1697 + MinerU Title Prior / Full-CE | READoc 1697 条；129-PDF 输入附 MinerU 标题提示 | 正文、标题、EOS 等权；Title Prior | 212 | 完整：129-PDF 正式重测（2026-09-01），见第 3 节 | `partial_learning`；Train Fit 6/6，详见 `train_fit_heading_prior_2026-09-01.md` |
| 9 | `readoc-heading-prior-16k-title-weighted` | READoc-1697 + MinerU Title Prior / Title-weighted | READoc 1697 条；129-PDF 输入附 MinerU 标题提示 | body=1、title=1.5、EOS=1；Title Prior | 212 | 完整：129-PDF 正式重测（2026-09-01），见第 3 节 | `partial_learning`；Train Fit 6/6，详见 `train_fit_heading_prior_2026-09-01.md` |

## 2. 训练配置摘要

共同配置：decoder-backbone LoRA，视觉编码器和 aligner 冻结；84 个 decoder modules；
LoRA rank=16、alpha=32、dropout=0.05；R-SWA window=128；gradient checkpointing 开启；
`seed=0`、`data_seed=0`。

| 方案组 | 上下文长度 | 有效 batch | 学习率 | 其他关键参数 |
|---|---:|---:|---:|---|
| PMC+READoc-3551 三套 | 32768 | 8 | `1e-4` | 2×H100，`grad_accum=4`，weight decay=0.1，cosine，warmup ratio=5%，bf16，SDPA |
| READoc-1706 Full-CE / Title-weighted | 16384 | 8 | `2e-5` | `grad_accum=8`，weight decay=0.01，cosine，warmup ratio=3%，bf16，SDPA；训练 prompt 为 `<image>Multi page merge.` |
| READoc-1697 + MinerU Title Prior 两套 | 16384 | 8 | `2e-5` | `grad_accum=8`，weight decay=0.01，cosine，warmup ratio=3%，bf16，SDPA；训练 prompt 为 `<image>Multi page merge.` |
| 夏桢两套 Title-mask | 16384 | 1 | `1e-4` | `batch_size=1`，`grad_accum=1`，warmup=85，max grad norm=1.0，eager attention，`multi_base`；训练 prompt 为 `<image>Multi page merge.`（带 prior 套加标题段，已逐行核实） |

PMC+READoc-3551 和 READoc-1706 Full-CE/Title-weighted 都约训练 1 epoch。夏桢两套的
更新步数分别等于训练数据行数，也各完整遍历训练集 1 次。详细 checkpoint、数据校验值和
训练日志位置由方案登记文档维护，本周报只保留可读的方案对比信息。

注：四套训练数据的训练 prompt 逐行核实过，全部为 `<image>Multi page merge.`
（带 prior 的两套额外加 MinerU 标题段）；`document parsing.` 只出现在评测的单页文件，
训练中从未出现。prompt 契约的逐字原文见 `evaluation_prompt_matrix_2026-09-01.md` §1。

## 3. 129-PDF Benchmark 统一对比

### 3.1 口径

- **正式主基线**：未训练 Base，单页 `<image>document parsing.`、多页
  `<image>Multi page parsing.`，统一推理服务入口（`image_size=1024`、`crop_mode=False`、
  单页 `ngram_window=128`、多页 `ngram_window=1024`、`max_length=20480`（2026-09-01 起，
  原 32768，变更依据见 prompt 标准文档）、`temperature=0`）。OmniDocBench `53.3645`、
  AI Builder Overall `0.8271`。
- 2026-08-27 与 2026-09-01 的全部 trained-model 请求走同一服务与同一 runner；区别仅在
  多页 prompt（训练契约 `Multi page merge.`）和 Title Prior 模型的逐文件标题段（来自唯一
  正确 manifest `input_manifests/heading_prior_20260901/prompts.all_pages.jsonl`）。
- 旧夏桢 badcase 入口（`eval_badcase.py`：`base_size=1024`、`image_size=640`、
  `crop_mode=True`、多页 `ngram_window=128`、merge Base `51.7663/0.7442`）已于
  2026-09-01 废弃：其带 prior 模型从未收到标题段，相关评测产物已删除，历史数字
  （`25.9391/0.6635`、`33.3329/0.6801`）不再作为任何方案的正式结果。

### 3.2 结果（全部为统一 READoc 129-PDF 入口）

| 模型/方案 | 输入说明 | OmniDocBench Total | Text | Table | Reading-order | Title（归一） | FormulaCDM | Overall | 相对主 Base |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base（正式主基线） | parsing / Multi page parsing | 53.3645 | 0.7995 | 0.8109 | 0.9351 | 0.7630 | 0.7816 | 0.8271 | - |
| READoc-1706 / Full-CE | merge | 44.1397 | 0.7918 | 0.7780 | 0.9278 | 0.7432 | 0.0000 | 0.8102 | -1.70 pp |
| READoc-1706 / Title-weighted | merge | 43.3568 | 0.7804 | 0.7796 | 0.9178 | 0.7292 | 0.0000 | 0.8017 | -2.54 pp |
| READoc-1706 / Title-mask（夏桢，无 prior） | merge | 28.6202 | 0.4206 | 0.6949 | 0.8291 | 0.0592 | 0.0000 | 0.5122 | -31.49 pp |
| READoc-1697 + Title Prior / Title-mask（夏桢，带 prior 输入） | merge + 标题段 | 24.4528 | 0.4048 | 0.5879 | 0.7460 | 0.0906 | 0.0000 | 0.4500 | -37.71 pp |
| READoc-1697 + MinerU Title Prior / Full-CE（hyx） | merge + 标题段 | 22.7157 | 0.2951 | 0.6475 | 0.6099 | 0.0485 | 0.0000 | 0.3923 | -43.48 pp |
| READoc-1697 + MinerU Title Prior / Title-weighted（hyx） | merge + 标题段 | 24.8699 | 0.2820 | 0.5798 | 0.6243 | 0.0536 | 0.2846 | 0.3940 | -43.31 pp |

AI Builder Overall 严格定义为
`(Text accuracy + Table accuracy + Reading-order accuracy + Title accuracy) / 4`；
Title 使用格式归一后的 grounding title 内容匹配分。表中数值保留 4 位小数。

### 3.3 生成诊断（2026-09-01 四套统一口径重测）

| 模型 | 重复行≥10% | 重复行≥40% | 输出为 grounding 格式 | 空输出 | 命中 20480 cap | 欠生成（<0.3×GT） |
|---|---:|---:|---:|---:|---:|---:|
| READoc-1706 / Title-mask（无 prior） | 21/129 | 15/129 | 129/129 | 0 | 0 | 2 |
| READoc-1697 + Title Prior / Title-mask | 65/129 | 52/129 | 54/129 | 0 | 0 | 3 |
| READoc-1697 + MinerU Prior / Full-CE | 54/129 | 48/129 | 74/129 | 0 | 0 | 5 |
| READoc-1697 + MinerU Prior / Title-weighted | 61/129 | 56/129 | 77/129 | 0 | 0 | 6 |

（2026-08-27 的两套 Full-CE/Title-weighted 诊断见 `evaluation_readoc_view_16k_2026-08-27.md`。）

### 3.4 结果解读

- 回退集中在四套 Title-mask / Title Prior 方案：Text 掉到 0.28-0.42（Base 0.80）、
  Reading-order 掉到 0.61-0.83、标题几乎全部丢失。共同特征是**输出协议未收敛**：
  带壳文件近 60% 跳出 Base 原生 grounding 协议、转入退化的 markdown 标题刷写模式
  （grounding 本身是 Base 原生健康协议；case 归因见评测文档第 9 节），该模式复读显著（重复行≥10% 的文档 21-65/129，≥40% 的 15-56/129）。
- **Title Prior 输入没有带来收益**：同一作者、同一入口的夏桢两套中，带 prior 的
  Overall 比无 prior 低 `6.22 pp`、Total 低 `4.17`；hyx 两套之间 Full-CE 与
  Title-weighted 基本持平（`0.3923` vs `0.3940`）。四套里相对最好的是夏桢无 prior
  Title-mask（`0.5122`），但其输出 100% 为 grounding 格式，标题分同样崩塌。
- **两轴定责（case 级归因后）**：壳对任何 loss 都必崩（-38 ~ -43.5 pp），全量监督+壳
  （`0.3923/0.3940`）比 title-mask+壳（`0.4500`）更差；loss 维度只在无壳时可比——全量
  监督 ≫ title-mask，Full-CE 与 Title-weighted 差 ≤0.9 pp 属噪声。8-27 两套的"好一点"
  是未被壳毒到，不代表学到东西（Train Fit 行召回仅 12-17%、公式归零）。详见
  `analysis_all_regress_attribution_2026-09-02.md` 第 7 节。
- 两套 READoc-1706 Full-CE/Title-weighted（`-1.70 / -2.54 pp`）仍是唯一接近 Base 的方案，
  但也无可证明的泛化收益，且公式能力全损（FormulaCDM 0）。
- PMC+READoc-3551 三套仍只有 75/387 抽测，不参与本表排名。

## 4. Train Fit / overfit

判定标准：`full_overfit_pass` 要求多数抽样样本能对齐 target，输出协议稳定，且没有
阻断性过生成或复读；部分样本命中但长短样本不稳定，统一判为 `partial_learning`。
本轮 benchmark 重测未改变任何 Train Fit 输入或探针代码，既有结论全部维持。

| 方案 | 抽样请求 | 平均文本相似度 | 平均 target 行召回 | 平均标题召回 | 平均重复行比例 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| PMC+READoc-3551 / Title-mask | 16/16 | 0.280 | 2.2% | 42.0% | 23.4% | `partial_learning` |
| PMC+READoc-3551 / Full-CE | 16/16 | 0.317 | 18.9% | 49.1% | 27.2% | `partial_learning` |
| PMC+READoc-3551 / Title-weighted | 16/16 | 0.347 | 18.2% | 46.6% | 30.4% | `partial_learning` |
| READoc-1706 / Full-CE | 15/15 | 0.495 | 12.6% | 9.2% | 9.7% | `partial_learning` |
| READoc-1706 / Title-weighted | 15/15 | 0.375 | 17.2% | 27.1% | 7.3% | `partial_learning` |
| READoc-1706 / Title-mask（无 Title Prior） | 8/8 | 0.067 | 16.3% | 49.1% | 73.5% | `partial_learning` |
| READoc-1697 + Title Prior / Title-mask | 8/8 | 0.209 | 27.8% | 48.2% | 72.2% | `partial_learning` |
| READoc-1697 + MinerU Title Prior / Full-CE | 6/6 | 0.101 | 0.0% | 2.8% | 14.8% | `partial_learning` |
| READoc-1697 + MinerU Title Prior / Title-weighted | 6/6 | 0.125 | 1.0% | 4.2% | 7.9% | `partial_learning` |

因此，本轮“能不能学会”的结论是：**现有 9 套方案都不能判定为稳定 overfit 通过或可部署；
`partial_learning=9/9`。**

## 5. 下一步计划

1. **不做任何部署**。6 套已完整评测方案全部相对主 Base 回退，其中 4 套回退幅度
   `>30 pp`，没有部署候选。
2. 先解决输出协议：统一训练 target 格式、推理模板、服务输出与评测 parser，消除
   Markdown / grounding / mixed 混杂；这是四套 Title-mask 方案标题分崩塌的直接原因。
3. 复读控制：重复行≥40% 的文档占比过高（15-56/129），需要从数据长度分布、loss 配置和
   解码约束三方面分别归因，再决定下一轮训练变量。
4. Title Prior 路线在统一协议下为负收益，且已完成 case 级归因：掉分由推理时标题段触发
   协议翻转造成，先验训练与先验质量（对 GT 标题覆盖 86%）均无责。若继续该方向，训练集
   须混入无壳样本保持协议回退，并在 Train Fit 上先验证 shell 输入可复现 target。
5. 后续实验一次只改变一个变量，分别记录数据组成、loss、输入格式和训练入口；每轮
   benchmark 与 Train Fit 双验收后再进周报。

## 6. 详细事实源

本周报只保留结论和可比较数字，不重复展示机器路径。详细路径变量、checkpoint、原始
metric、预测文件和命令由以下文档维护：

- `training_scheme_registry_2026-08-29.md`
- `runs_registry_2026-08-27.md`
- `evaluation_prompt_matrix_2026-09-01.md`（prompt 唯一标准）
- `evaluation_readoc_view_16k_2026-08-27.md`
- `evaluation_readoc_heading_prior_2026-09-01.md`
- `analysis_all_regress_attribution_2026-09-02.md`（全量回退归因汇总：统一协议结果 + case 级证据）
- `train_fit_heading_prior_2026-09-01.md`
- `posttrain_completion_sop_2026-08-28.md`

评测产物目录：

~~~text
evaluation/readoc-view-16k_all129/                                   2026-08-27 两套 Full-CE/Title-weighted + Base
evaluation/heading_prior_20260901_129/                               2026-09-01 hyx 两套 Title Prior 正式评测
evaluation/wxz-readoc-1706-title-mask_20260901_129/                  2026-09-01 夏桢无 prior 统一协议重测
evaluation/wxz-readoc-1697-title-prior-mask_20260901_129/            2026-09-01 夏桢带 prior 统一协议重测
evaluation/inference_eval_20260901_wxz_{prior,noprior}_all129/       上述两轮的原始推理记录
~~~
