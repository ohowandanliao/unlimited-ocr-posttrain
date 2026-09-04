# 训练、权重与评测登记

更新时间：2026-09-04（Asia/Shanghai，统一协议结果与开源路径整理）

这份文档是训练产物和评测产物的对应关系登记。运行文件统一放在
`$UOCR_ROOT/`，本文件只记录方案、路径、数据和结果，
不把不同数据、入口或评测口径的结果混为同一方案。

## 1. 目录约定

当前根目录按产物用途分开：

~~~text
uocr-ms-swift-title-mask/
  data/          训练数据实体；data/review/ 下是软链接
  output/        模型权重和 checkpoint
  evaluation/    推理与评测结果
  logs/          训练、评测日志
  service/       服务日志
  env/ repos/ runtime/
~~~

训练数据实体：

- data/readoc-view-16k/
- data/reviewed-title-32k/

review 入口只做指向，不复制数据：

- data/review/readoc-view-16k -> ../readoc-view-16k
- data/review/reviewed-title-32k -> ../reviewed-title-32k

recipe、长度分桶和审计报告放在 data/recipes/ 或 data/length/；模型权重只放在 output/，评测只放在 evaluation/。
未登记的 runs、train 和 outputs 不属于本次方案登记。

当前实际目录：

- 训练数据：data/
- 模型权重：output/
- 本轮全量评测：evaluation/readoc-view-16k_all129/
- 评测环境：evaluation/tooling/omni-eval-venv/
- 日志：logs/
- 服务：service/

## 2. 已训练方案与权重

### 2.1 READoc-1706 双 loss 方案

共同训练数据：

| 项目 | 内容 |
|---|---|
| 原始 view train | `$WXZ_READOC_ROOT/readoc_uocr/r0/views/train_le_16k_with_title.jsonl` |
| 原始 view validation | `$WXZ_READOC_ROOT/readoc_uocr/r0/views/val_le_16k.jsonl` |
| 原始 view test | `$WXZ_READOC_ROOT/readoc_uocr/r0/views/test_le_16k.jsonl` |
| 派生 recipe | `data/readoc-view-16k/` |
| 派生划分 | train/validation/test = 1706/95/95 |
| recipe 总行数/图片/标题 | 1896 / 13288 / 27446 |
| prompt | 全部为 `<image>Multi page merge.` |
| 最大长度 | 16384；tokenizer audit overflow = 0 |
| 模型 | `$MODEL_PATH` |
| 训练 | 1 epoch，learning rate 2e-5，seed 0，bf16，SDPA |
| LoRA | 84 个 decoder modules，rank 16，alpha 32，dropout 0.05 |
| 有效 batch | 8（GPU0/GPU1 各一个 WORLD_SIZE=1 任务，grad_accum=8） |

训练权重和评测对应关系：

| 方案 | loss 定义 | 保留 checkpoint | 本轮评测 checkpoint | 评测结果目录/日志 |
|---|---|---|---|---|
| READoc-1706 / Full-CE | native ms-swift token CE，正文/标题/EOS 等权 | `output/readoc-view-16k-full-ce/v0-20260826-232140/checkpoint-{50,100,150,200,213}` | checkpoint-213 | AI Builder：`.../agentbuilder/full-ce/`；Omni：`.../omnidocbench_full-ce_gtpdf_remote_rerun.log` |
| READoc-1706 / Title-weighted | body=1.0，title=1.5，EOS=1.0；accumulation-level token denominator | `output/readoc-view-16k-title-weighted/v0-20260826-232140/checkpoint-{50,100,150,200,213}` | checkpoint-213 | AI Builder：`.../agentbuilder/title-weighted/`；Omni：`.../omnidocbench_title-weighted_gtpdf_remote_rerun.log` |
| Base | 不训练，无 LoRA | 无 | base model | AI Builder：`.../agentbuilder/base/`；Omni：`.../omnidocbench_base_gtpdf_remote_rerun.log` |

本轮 129-PDF 评测使用的 source、GT 和评测工具：

| 项目 | 路径/说明 |
|---|---|
| source PDF | `$EVAL_PDF_ROOT/source` |
| groundtruth | `$EVAL_GT_ROOT` |
| OmniDocBench | `$OMNIDOCBENCH_ROOT` |
| AI Builder 环境 | `evaluation/tooling/omni-eval-venv`，Python 3.13.13，mmeval 0.2.1 |
| 规模 | 129 PDF × 3 模型 = 387 请求，全部成功 |
| 后处理 | 未做去重、截断、修复或协议转换；OmniDocBench 使用 raw prediction，AI Builder 只做必要的 wrapper 解析 |

### 2.2 PMC+READoc-3551 训练权重

这是 PMC+READoc-3551 的三种 loss 方案，与 READoc-1706 结果分开登记：

| 方案 | loss | checkpoint | 训练数据 |
|---|---|---|---|
| PMC+READoc-3551 / Full-CE | native Full CE | `output/reviewed-full-ce-dual-32k/v0-20260824-233410/checkpoint-{400,443}` | `data/reviewed-title-32k/length_32k_v1/fit/train.jsonl`；PMC 2184 + READoc 1367 |
| PMC+READoc-3551 / Title-mask | title active-token mean，title+EOS-only | `output/reviewed-title-mask-dual-32k/v0-20260823-201829/checkpoint-{400,443}` | 同上 |
| PMC+READoc-3551 / Title-weighted | body=1，title=2，EOS=1 | `output/reviewed-title-weighted-dual-32k/v0-20260825-031657/checkpoint-{400,443}` | 同上 |

它们对应的历史推理目录是
`evaluation/inference_eval_20260825_all129_parallel_v3/`（旧 75/387 抽测的实际目录；
该抽测实际只跑了 Base、PMC+READoc-3551 / Full-CE 和 Title-weighted，Title-mask 没有进入这次三路结果；
129-PDF OmniDocBench/AI Builder 也没有使用这些 checkpoint）。这些权重目前保留，但没有部署。

## 3. 2026-08-27 READoc 评测结果

### 3.1 OmniDocBench

| 模型/权重 | Total |
|---|---:|
| Base | 53.3645 |
| READoc-1706 / Full-CE checkpoint-213 | 44.1397 |
| READoc-1706 / Title-weighted checkpoint-213 | 43.3568 |

### 3.2 AI Builder 细分

| 指标 | Base | Full CE checkpoint-213 | Title-weighted checkpoint-213 |
|---|---:|---:|---:|
| Text accuracy | 0.7995 | 0.7918 | 0.7804 |
| Table accuracy | 0.8109 | 0.7780 | 0.7796 |
| Reading-order accuracy | 0.9351 | 0.9278 | 0.9178 |
| Title accuracy（格式归一/grounding 内容匹配） | 0.7630 | 0.7432 | 0.7292 |
| Title accuracy（原始 ATX Markdown，仅诊断） | 0.000018 | 0.000018 | 0.002649 |
| TextEdit | 0.2005 | 0.2082 | 0.2196 |
| Table TEDS | 0.5547 | 0.5248 | 0.5197 |
| Table TEDS-S | 0.5744 | 0.5479 | 0.5468 |
| Read OrderEdit | 0.0649 | 0.0722 | 0.0822 |
| Formula Edit accuracy | 0.2467 | 0.0076 | 0.0006 |
| FormulaCDM | 0.7816 | 0 | 0 |
| AI Builder Overall（四项算术平均） | 0.8271 | 0.8102 | 0.8017 |

本报告的 AI Builder Overall 定义为：`(Text accuracy + Table accuracy + Reading-order accuracy + Title accuracy) / 4`。
Title accuracy 使用格式归一后的 grounding 标题内容匹配分，忽略 heading level；原始 ATX title 分只作协议诊断，
不计入 Overall。原始 `metrics.json/csv` 中的 `overall` 按评测工具旧实现生成，不能与本报告的四项均值混用。

### 3.3 夏桢 Title-mask 正式重测

统一协议重测已完成；两套分别使用无 prior 默认 prompt 与带 prior 的逐文件 manifest，主 Base
直接复用 `53.3645 / 0.8271`。旧 badcase 产物已删除，历史 `25.9391 / 0.6635`、
`33.3329 / 0.6801` 及其相对收益均为废弃历史审计，不代表方案结果。

| 方案 | OmniDocBench Total | AI Builder Overall | 相对主 Base |
|---|---:|---:|---:|
| READoc-1706 / Title-mask（无 Title Prior） | 28.6202 | 0.5122 | -31.49 pp |
| READoc-1697 + Title Prior / Title-mask | 24.4528 | 0.4500 | -37.71 pp |

### Base 口径与评测 diff

本登记的正式主 Base 是未训练模型在统一 READoc 入口下使用
`<image>Multi page parsing.` 的结果：OmniDocBench `53.3645`，AI Builder Overall
`0.8271`。READoc-1706 的训练模型使用 `<image>Multi page merge.`；这个 merge prompt
属于训练模型输入契约，不能反过来用于未训练 Base。

夏桢历史 badcase 的 Base 使用 merge prompt、`eval_badcase.py` 和不同图像/解码参数；其
`51.7663 / 0.7442` 随已删除产物仅作废弃历史审计，不能替代 parsing Base 或重算正式相对差值。

结果报告：

- 详细分析：`docs/evaluation_readoc_view_16k_2026-08-27.md`
- 原始评测输出：`evaluation/readoc-view-16k_all129/`
- 其中 `responses.jsonl` 是 387 条原始响应，`repeat_scan.json/.md` 是自动复读诊断。
- OmniDocBench 成功重跑凭据是结果根目录下的三份 `omnidocbench_*_gtpdf_remote_rerun.log`；
  `omnidocbench/` 目录只保留了评测工具创建的预留目录，不能当作指标文件目录。

复读诊断仍显示局部循环/过生成：`蓝德_办公电脑配置标准_2.pdf` 三路都重复同一语义块，
`5-2表旁水印.pdf` Full-CE 和 Title-weighted 完全一致且生成 1337 个日期，Title-weighted 在
`4-1统计图1.pdf` 生成 1088 个年份项。此前 `3-4长表`、`4-3扫描2` 的 92%-98% 整页复读
本轮没有复现。

## 4. Smoke 权重清理记录

已删除以下旧 smoke 训练输出目录：

- `output/smoke-title-mask-single-32k-sdpa`
- `output/smoke-title-mask-single-4k`
- `output/smoke-title-mask-single-8k`
- `output/smoke-title-mask-single-v4`
- `output/smoke-title-mask-single-v5`
- `output/smoke-title-mask-single-v6`
- `output/smoke-title-mask-single-v7`
- `output/smoke-title-mask-single-v8`

删除前共 144,635,192 bytes；其中前三个含 checkpoint-1 adapter，其他目录是未完成的
训练元数据。对应评测与训练 smoke 数据随后也在 2026-09-03 的清理中删除。

## 5. 服务状态

本文件不记录瞬时服务、PID、端口或 GPU 状态；全部历史评测服务均已停止。现行启动契约见
`inference_service_2026-08-24.md`，其中要求显式设置 `MODEL_PATH`、`FULL_CE_ADAPTER_PATH`
和 `TITLE_WEIGHTED_ADAPTER_PATH`。

## 夏桢 READoc Title-mask 训练（2026-08-24）

完整记录见 [`夏桢_readoc_title_mask_training_2026-08-29.md`](wxz_readoc_title_mask_training_2026-08-29.md)。
这两次训练来自 `$WXZ_ROOT/unlimited-ocr-finetune`，对应 READoc-1706 / Title-mask
和 READoc-1697 + Title Prior / Title-mask。

| 方案 | 数据 train 行数 | grad_accum | 更新步数 | 最终 LoRA |
|---|---:|---:|---:|---|
| READoc-1706 / Title-mask（无 Title Prior） | 1706 | 1 | 1706 | `$WXZ_OUTPUT_ROOT/readoc_title_mask_16k/` |
| READoc-1697 + Title Prior / Title-mask | 1697 | 1 | 1697 | `$WXZ_OUTPUT_ROOT/readoc_heading_prior_title_mask_16k/` |

两次均为 1 epoch、`batch_size=1`、`loss_mode=title_mask`、R-SWA window 128。
旧 badcase 评测产物已删除；正式结果见本文件第 3.3 节与周报。
