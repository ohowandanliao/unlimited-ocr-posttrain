# Unlimited-OCR 历史训练与三路评测记录

> **历史实验事实。** 本文保留 title+EOS-only、Full CE、Title-weighted 的训练事实，以及
> Base/Full CE/Title-weighted 三路评测记录，不再决定下一轮训练；当前根因、recipe 和停止条件见
> [`training_optimization_2026-08-26.md`](../../training_optimization_2026-08-26.md)。

更新时间：2026-08-25（Asia/Shanghai）

本文是本轮 posttrain 分析资料的唯一对照总览。posttrain 在这里仅作为代码、配置和分析材料的归档位置，不是训练、推理或评测的启动入口。本轮训练、服务和全量评测均已结束或停止，本文只记录已经发生的事实；“可能原因”单独标为待确认。

本轮评测对照的三路是：未训练 Base、完整 token CE 的 Full CE、标题加权的 Title-weighted。Base 不是
训练方案，而是同一批推理输入上的未训练基线。此前另有一条 title+EOS-only 训练，但没有进入这次三路
生成评测，不能从它的低 loss 推断完整 Markdown 可用性。

## 1. 共同输入

### 1.1 训练数据

两种训练方案使用同一份 32K fit 数据，输入图片、prompt、完整 assistant target 和数据划分完全一致：

| 项目 | 内容 |
| --- | --- |
| 数据目录 | `/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/length_32k_v1/fit/` |
| train | 3,551 |
| validation | 188 |
| test | 197 |
| overflow | 588；超过 32K，不截断、不训练 |
| 可训练合计 | 3,936 |
| 图片引用 | 28,604 |
| Markdown headings | 54,179 |
| 训练目标 | 完整、连续的 Markdown |

训练消息的输入和输出形态如下：

- 单页 prompt：`<image>document parsing.`
- 多页 prompt：`<image>Multi page merge.`
- assistant target 是完整 Markdown，不是标题列表。
- target 没有加入 `<PAGE>`，也没有加入 `<|det|>` grounding 标签。
- 训练使用完整 `input_ids` 和 `labels`；两种训练方案的差别只在 loss 计算方式。

### 1.2 共同训练配置

- context：32,768
- GPU：2 x H100
- 每卡 batch：1；gradient accumulation：4
- dtype：bf16
- attention：SDPA
- R-SWA：启用，`ring_window=128`
- gradient checkpointing：启用
- LoRA：rank 16、alpha 32、dropout 0.05
- decoder LoRA target：84 个；视觉编码器、aligner/projector、embedding、lm_head 和 routed experts 冻结

### 1.3 推理输入

用于观察现象的 evaluation 输入为 129 个 source PDF，共 372 页：

- 平均 2.88 页/文件，109 个单页 PDF。
- 超过 10 页的只有 8 个，最长 38 页。
- 计划请求为 `129 x 3 = 387` 条，每个 PDF 分别请求 Base、Full CE、Title-weighted。
- 单页三路使用 `<image>document parsing.`。
- 多页 Base 使用 `<image>Multi page parsing.`；Full CE 和 Title-weighted 使用 `<image>Multi page merge.`。

推理输出没有在服务端改写 assistant 文本，原始响应和每条回答 Markdown 均保留在运行目录中。

## 2. 三种方案的输入、loss、输出

### 2.1 Base：未训练基线

| 项目 | 记录 |
| --- | --- |
| 模型输入 | 上述 evaluation PDF 渲染的页图；单页或多页 prompt 见 1.3 |
| 训练 | 不训练；推理时使用 base 模型并禁用 adapter |
| 模型 | `/home/jovyan/hyx/models/Unlimited-OCR` |
| checkpoint | 无 |
| 输出协议 | 更常见的是 Unlimited-OCR 原生 grounding 格式，但长文档和扫描件也会出现重复或过生成 |

已观察到的 Base 例子：`4-3扫描2.pdf` 输出 66,075 字符、29,485 token，重复比例约 39%，其中大量重复 `(No text)`。这说明未训练基线本身也存在长文档/扫描件生成异常，不能把所有异常直接归因于训练。

### 2.2 Full CE：完整 assistant target 的普通 token CE

| 项目 | 记录 |
| --- | --- |
| 模型输入 | 与 Title-weighted 完全相同的图片、prompt、完整 labels |
| loss | assistant target 上的普通 token CE；active response token 等权 |
| 训练结果 | 443/443 steps；无 OOM |
| train loss | `0.23046987` |
| final eval loss | `0.16648881` |
| 训练耗时 | `4669.0563s`，约 77 分 49 秒 |
| checkpoint | `/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/outputs/reviewed-full-ce-dual-32k/v0-20260824-233410/checkpoint-443` |
| 日志 | `/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/full_ce_dual_32k.log` |
| 输出根目录 | `/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/outputs/reviewed-full-ce-dual-32k/` |

Full CE 的输出不是稳定的纯 Markdown：75 条已落盘结果中，既有 Markdown heading，也有原生 grounding 标签；长表、扫描件和跨页文档出现明显重复、过短或过长输出。

### 2.3 Title-weighted：正文 1、标题 2、EOS 1

| 项目 | 记录 |
| --- | --- |
| 模型输入 | 与 Full CE 完全相同的图片、prompt、完整 labels |
| loss | 正文 token 权重 1.0；Markdown heading token 权重 2.0；EOS 权重 1.0；prompt/image/padding 为 0 |
| 训练改动 | 保留完整 labels、R-SWA 前缀推导和 response 起点，只改变逐 token loss scale，并按 active weight sum 归一化 |
| 训练结果 | 443/443 steps；无 OOM |
| train loss | `0.23844284` |
| final eval loss | `0.17209093` |
| 训练耗时 | `4485.6926s`，约 74 分 46 秒 |
| checkpoint | `/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/outputs/reviewed-title-weighted-dual-32k/v0-20260825-031657/checkpoint-443` |
| 日志 | `/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/title_weighted_dual_32k.log` |
| 输出根目录 | `/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/outputs/reviewed-title-weighted-dual-32k/` |

这里的 1:2 是相对权重。若把正文和标题都设置为 0.5，标题相对正文并不会被强调；active-weight mean 只负责稳定 loss 数值尺度。Full CE 和 Title-weighted 的 loss 数值不能直接当作质量分数比较，因为目标函数不同。

### 2.4 Title+EOS-only：未进入三路生成评测的历史 run

| 项目 | 记录 |
| --- | --- |
| loss 范围 | 只监督 Markdown heading 与 EOS；正文权重为 0 |
| 训练结果 | 443/443 steps |
| 最佳记录 | checkpoint-400 eval loss `0.0044472` |
| 生成评测 | 未完成可用性评测，也未进入本文件的三路 evaluation |
| 当前决策 | 不再用于“输出完整 Markdown”的 OCR 任务 |

这个 loss 与 Full CE、Title-weighted 的有效 token 范围不同，数值不能横向比较。若任务以后改成只输出标题树，
应另建标题专用 output contract，而不是复用完整 Markdown response。

## 3. 推理结果的完整性

全量任务在发现生成质量明显退化后停止，保留目录如下：

`/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/inference_eval_20260825_all129_parallel_v3/`

这是 75/387 条，不是完整 129 个 PDF 的准确率评测。保留结果的结构检查如下：

| 检查项 | 结果 |
| --- | ---: |
| 唯一 `(file, model)` 记录 | 75/75 |
| HTTP/请求错误 | 0 |
| 空输出 | 0 |
| 返回模型与请求模型一致 | 75/75 |
| 返回页数与输入页数一致 | 75/75 |
| 返回 prompt 与请求 prompt 一致 | 75/75 |
| 独立回答 Markdown | 75 |

因此，已记录的问题主要发生在生成内容，不是本轮记录中的端口路由、模型串错、页数错配或文件传输错误。

## 4. 已观察到的输出现象

### 4.1 总体统计

- 51/75 条仍包含 `<|det|>...</|det|>` grounding 标签；24/75 条出现 Markdown heading，输出协议混杂。
- 15/75 条被重复行检测器标记，其中 8/75 条的重复比例达到或超过 70%。
- 7/75 条输出超过 30,000 token，接近 32,768 上限；没有超过 32,000 token 的记录。
- Full CE 与 Title-weighted 在以下三个文件上文本完全相同：`1-2无线表2.pdf`（1,971 字符）、`3-1跨页重复表头.pdf`（576 字符）、`5-3页眉近表.pdf`（1,579 字符）。
- 这批结果没有做正式准确率评分；重复比例、字符数和 token 数是输出现象信号，不等于准确率。

### 4.2 代表性案例

| 输入 PDF | 页数 | 模型 | 输出与现象 |
| --- | ---: | --- | --- |
| `3-4长表跨三页以上.pdf` | 26 | Full CE | 46,127 字符 / 25,742 token，重复比例 98% |
| `4-3扫描2.pdf` | 12 | Full CE / Title-weighted | 两路均 2,034 字符 / 1,005 token，重复比例 92%；反复输出同一公司名 |
| `4-3扫描2.pdf` | 12 | Base | 66,075 字符 / 29,485 token，重复比例 39%；大量重复 `(No text)` |
| `5-1双栏表格1.pdf` | 2 | Title-weighted | 10,035 字符 / 5,870 token，重复比例 90% |
| `3-1跨页重复表头.pdf` | 25 | Full CE / Title-weighted | 两路均仅 576 字符 / 266 token，文本完全相同，输出严重缩短 |
| `4-1统计图3.pdf` | 31 | Full CE / Title-weighted | 重复比例分别为 56% / 73% |
| `4-2段落中带「图x-x」.pdf` | 38 | Title-weighted | 38,139 字符 / 22,386 token，重复比例 73%，页级内容反复生成 |
| `4-3扫描1.pdf` | 5 | Full CE / Title-weighted | GT 3,613 字符；输出 51,137 / 53,183 字符，约为 GT 的 14.2 / 14.7 倍 |
| `2-5表内公式.pdf` | 1 | Full CE / Title-weighted | 重复比例分别为 71% / 79% |
| `5-1双栏表格3.pdf` | 未在此表列出 | Full CE | 135,063 字符 / 30,841 token，过生成明显但未被重复检测器标记 |

### 4.3 训练方案之间的直接对照

- Title-weighted 没有消除长表、扫描件、跨页表头和双栏文档上的重复问题。
- 在 `4-3扫描2.pdf` 上，Full CE 和 Title-weighted 的输出完全同样的重复模式；Base 也有异常，但表现为大量 `(No text)` 的长输出。
- 在 `3-1跨页重复表头.pdf` 上，两种训练 checkpoint 输出完全相同且严重缩短。
- 在 `4-1统计图3.pdf` 和 `2-5表内公式.pdf` 上，Title-weighted 的重复信号高于 Full CE。
- 现有 75 条不足以证明哪一种方案整体更好；目前能确认的是，标题加权没有解决已暴露的主要退化现象。

## 5. 用于进一步分析的原始材料

posttrain 中保留一份最小可复算案例包：

`/home/jovyan/hyx/unlimited-ocr-posttrain/docs/evaluation_failure_cases_2026-08-25/`

- `gt__*.md`：8 个对应 groundtruth。
- `cases.jsonl`：24 条结构化原始记录，完整保留 response 文本、prompt、页数、模型和现象信号。
- `analysis_2026-08-26.md`：从 JSONL 与 GT 生成的可读指标表。
- `README.md`：案例索引、原始路径和按需展开回答的命令。

原来的 24 个 `case_*.md` 与 `cases.jsonl` 逐条生成结果 24/24 字节一致，属于可再生产物，已不再重复
提交。需要逐回答 Markdown 时，按案例包 README 的命令写到仓库外临时目录。

原始评测目录和数据路径：

- 75 条完整响应：`/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/inference_eval_20260825_all129_parallel_v3/`
- source PDF：`/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/source/`
- groundtruth：`/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth/`

训练实现相关代码位置：

- `ms_swift_title_mask/core.py`：逐 token loss scale 和 active-weight mean。
- `ms_swift_title_mask/plugin/uocr_title_mask.py`：Full/Title-weighted 的 template 与 loss 注册。
- `ms_swift_title_mask/scripts/_run_train.sh`：训练 mode 选择。

## 6. 可能原因（待确认）

下面只是根据已落盘现象整理的排查方向，不是已验证的模型归因：

1. 三路结果没有 HTTP 错误、空输出或请求字段错配，因此当前证据不支持把主要问题归因于评测脚本或端口分流。
2. 训练 target 是 Markdown，而 Base 常见输出是 grounding；训练后仍混有两种协议，可能与训练 target、推理 prompt/template、原生 Unlimited-OCR 输出协议之间的不一致有关，具体层次尚未隔离。
3. 重复、严重缩短和超长过生成同时存在，可能涉及 EOS 学习、label/loss mask、跨页生成状态、视觉输入质量或解码停止条件；现有结果没有单独实验区分这些因素。
4. Full CE 和 Title-weighted 在部分文件完全相同，可能是这些输入上的 LoRA 差异不足，也可能是两路都受到同一个停止/重复机制影响；不能据此断言加权实现没有生效。
5. 7 条结果接近 32K 上限，可能与 EOS 未及时停止或重复内容没有被当前解码抑制参数拦截有关；本轮没有 OOM 证据。
6. Base 在扫描件上也会重复，说明至少有一部分问题可能属于 base 模型或输入视觉条件；训练退化与 base 固有问题需要单独实验拆分。

## 7. 文档边界

本文件是历史训练和三路评测的事实总览。`evaluation_failure_cases_2026-08-25/` 只保留结构化原始输出、
GT 和一份可读派生分析；不再为同一批事实新增逐条展开或摘要文档。posttrain 不承担启动训练、启动服务或
持续监控职责，运行日志、checkpoint 和评测产物仍放在
`/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/` 下。
