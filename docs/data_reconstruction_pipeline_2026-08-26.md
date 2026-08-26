# Unlimited-OCR 数据构造、训练链路与复现预期

更新时间：2026-08-26（Asia/Shanghai）

> **历史快照复现文档，不是下一轮 recipe。** 旧 PMC builder 没有消费正文 quarantine，标题
> `SILVER_ACCEPTED` 不能证明完整 Markdown 合格；旧 natural union 已停止。当前训练决策见
> [`training_optimization_2026-08-26.md`](training_optimization_2026-08-26.md)。现有原始数据中也不包含全部
> accepted decision manifest，因此只能重建候选/等价链路，不能仅凭原始数据宣称逐 SHA 复原旧 3,936/588 快照。

本文单独记录本轮训练数据从 PMC/READoc 源数据到 32K 训练集的构造链路、每一步的输入输出和数量闸门，供另一台服务器和分析 agent 在无法直接看到旧的 fit JSONL、页图或运行目录时重建数据并分析原因。

本文是数据复现说明，不是训练、推理或评测启动指南。所有重建都应写入新的工作目录，不能覆盖源数据、冻结 JSONL 或旧运行产物。

## 1. 复现目标

本轮实际训练使用的是：

```text
READoc full + PMC full + PMC strict-single
-> reviewed mix
-> Unlimited-OCR template/tokenizer 长度审计
-> 32K fit / overflow
-> Full CE 或 Title-weighted 训练
```

要复现的不是任意一个 PMC/READoc 数据集，而是以下具体快照：

- 页面输入：PDF 按原始页序渲染的 RGB PNG。
- 单页输入 prompt：`<image>document parsing.`
- 多页输入 prompt：`<image>Multi page merge.`
- assistant output：与输入页面对应的完整连续 Markdown。
- 不加入 `<PAGE>`，不把标题列表或 `middle.json` block 列表作为模型 output。
- Full CE 和 Title-weighted 使用同一份 input、完整 labels 和同一数据划分；旧 Title-weighted 除逐 token
  loss scale 外，还采用逐 micro-step active-weight 归一化，因此与 native Full CE 不是干净的单变量 A/B。

## 2. 源数据契约

### 2.1 READoc

READoc 源数据根目录应至少包含：

```text
<READOC_ROOT>/
  ground_truth/arxiv/*.md
  ground_truth/github/*.md
  archives/arxiv.zip
  archives/github.zip
```

PDF 位于 archive member 中，不能只按文件名猜 PDF。构建器使用 `source + doc_id` 配对 Markdown 与 archive member，并记录 PDF SHA-256、页数和 member 路径。

### 2.2 PMC

PMC 源数据根目录应至少包含：

```text
<PMC_ROOT>/
  documents/<doc_id>/middle.json
  documents/<doc_id>/document.pdf
```

`middle.json` 只用于离线构造、定位和审计完整 Markdown；它不是模型 input。模型 input 仍然是从对应 PDF 渲染出的页面图像，output 是完整 Markdown。

## 3. 构造链路

### 3.1 固定 posttrain 工作树

先复制当前 posttrain 工作树，不能只 clone 旧 commit。当前数据规则、标题脚本、JSONL 构建器和训练插件存在未提交修改；只使用历史 commit 可能得到不同数据。

至少需要这些代码和规则文件：

```text
scripts/data/build_readoc_title_gt.py
scripts/data/build_pmc_title_candidates.py
scripts/data/readoc_title_rules.py
scripts/data/pmc_title_rules.py
scripts/data/readoc_title_review_policy_v1.json
ms_swift_title_mask/scripts/build_readoc_silver_jsonl.py
ms_swift_title_mask/scripts/build_pmc_silver_jsonl.py
ms_swift_title_mask/scripts/materialize_pdf_pages.py
ms_swift_title_mask/scripts/relocate_image_paths.py
ms_swift_title_mask/scripts/build_reviewed_mix.py
ms_swift_title_mask/scripts/build_length_buckets.py
ms_swift_title_mask/scripts/validate_reviewed_jsonl.py
ms_swift_title_mask/data_contract.py
```

### 3.2 READoc 标题和 source audit

使用：

```text
scripts/data/build_readoc_title_gt.py
```

输入：READoc 的 `ground_truth/` 和 `archives/`。

输出应包含派生 GT、文档 manifest、PDF profile、heading inventory、title candidates、review queue、change manifest 和 validation。关键校验包括：

- PDF 与 GT 集合一致。
- PDF SHA-256 和页数记录完整。
- 标题改动有 PDF 证据和前后 SHA。
- 规则版本和 review policy 固定。
- 不能把无法确认的标题或正文清洗结果静默放入训练集。

### 3.3 PMC 标题候选和内容 audit

使用：

```text
scripts/data/build_pmc_title_candidates.py
```

输入：PMC `documents/<doc_id>/{middle.json,document.pdf}`。

输出应包含：

```text
document_manifest.jsonl
title_candidates.jsonl
review_queue.jsonl
text_review_queue.jsonl
content_quarantine.jsonl
summary.json
```

脚本会校验 `middle.json` 页数与 PDF 页数、记录 source/PDF SHA，并把坏公式、空页、缺失图片资源、粘连元数据、图片表格等内容问题与标题层级问题分开记录。

### 3.4 生成 READoc/PMC 训练池

READoc 使用：

```text
ms_swift_title_mask/scripts/build_readoc_silver_jsonl.py
```

输入：READoc 派生 GT/audit root 和页面图片 root。

输出：按 `train.jsonl`、`validation.jsonl`、`test.jsonl` 划分的 READoc full pool，以及 standalone ground-truth 和 manifest。

PMC 使用：

```text
ms_swift_title_mask/scripts/build_pmc_silver_jsonl.py
```

输入：PMC 原始 root、PMC audit root 和页面图片 root。

输出：

```text
pmc_full/{train,validation,test}.jsonl
pmc_single/{train,validation,test}.jsonl
```

其中 `pmc_single` 是与 PMC full 配对的严格单页样本，不是从多页 target 随意截断出来的标题列表。full/single 必须继承同一个 document-level split。

这里记录的是旧 builder 的实际输出，不是新的准入建议。该 builder 未读取 `content_quarantine.jsonl`，并把
title status 复用于完整样本 provenance；按当前 gate 生成结果会被 `pmc_s10` 拒绝。

如果使用显式人工或规则 review manifest，应先使用：

```text
ms_swift_title_mask/scripts/prepare_reviewed_jsonl.py
```

它会校验 review status、assistant SHA、heading 数量、split 和图片路径；未被接受的 review 记录不能进入训练池。

### 3.5 渲染页面图片

使用：

```text
ms_swift_title_mask/scripts/materialize_pdf_pages.py
```

输入：训练池 JSONL、READoc/PMC source manifest 和对应源 PDF。

默认渲染参数是 144 DPI、RGB、`alpha=False`，并严格校验：

- source PDF SHA 与 manifest 一致；
- PDF 实际页数与 manifest 一致；
- full_document 的 `page_indices` 覆盖完整连续页序；
- 每个 JSONL image 引用都能落到正确的 source/doc/page；
- 不允许缺页、越界或只按文件名猜路径。

建议输出目录：

```text
<IMAGE_ROOT>/
  readoc/<source>/<doc_id>/page_0000.png
  pmc/<doc_id>/page_0000.png
```

如果冻结 JSONL 中保存的是其他机器的绝对路径，使用：

```text
ms_swift_title_mask/scripts/relocate_image_paths.py
```

生成新的 JSONL 和本机绝对图片路径。不要原地修改冻结 JSONL。

### 3.6 合并三类训练池

使用：

```text
ms_swift_title_mask/scripts/build_reviewed_mix.py
```

输入：READoc full、PMC full、PMC strict-single 三个 pool。

合并规则：

- natural union：每个 pool 的 accepted row 都保留；
- 不按文档数重新过采样；
- PMC full 和 PMC single 必须拥有相同 document 集合和 split；
- 同一 document 不能跨 train/validation/test；
- row id 不能重复；
- 使用固定 seed：`uocr-reviewed-mix-20260823-v2`；
- 校验完整 images、target、provenance 和 source SHA。

### 3.7 32K 长度审计和分桶

使用：

```text
ms_swift_title_mask/scripts/build_length_buckets.py
```

输入：reviewed mix 的三份 split JSONL、同版本 Unlimited-OCR tokenizer 和模型配置。

长度不是按字符数或页数估计，而是按固定 contract 计算：

```text
BOS
+ prompt 中 <image> 两侧的文本 token
+ 每张图对应的视觉 token contract
+ assistant target token
+ EOS
```

`--max-length 32768` 后，脚本输出：

```text
fit/{train,validation,test}.jsonl
overflow/{train,validation,test}.jsonl
buckets/*.jsonl
smoke/*.jsonl
length_manifest.jsonl
```

overflow 必须保留在 manifest 中，但不能被静默截断或混入本轮训练。

## 4. 预期数据快照

### 4.1 三个输入池

| 数据池 | 行/文档 | 图片引用 | headings |
| --- | ---: | ---: | ---: |
| READoc full | 1,552 | 15,290 | 21,400 |
| PMC full | 1,486 | 26,690 | 48,726 |
| PMC strict-single | 1,486 | 1,486 | 6,011 |
| final mix | 4,524 | 43,466 | 76,137 |

final mix 的预期 split 是：

| split | rows |
| --- | ---: |
| train | 4,073 |
| validation | 223 |
| test | 228 |

### 4.2 32K fit/overflow

使用与正式训练相同的 tokenizer、模型 revision、视觉 token contract 和 `max_length=32768` 后，预期为：

| split | fit | overflow |
| --- | ---: | ---: |
| train | 3,551 | 522 |
| validation | 188 | 35 |
| test | 197 | 31 |
| 合计 | 3,936 | 588 |

正式 Full CE 和 Title-weighted 训练只使用 3,936 条 fit 数据。588 条 overflow 不截断、不训练。

如果重建数量、split、图片数、heading 数、target SHA 或 overflow 分布不一致，应先报告差异，不要把重建结果直接当作本次训练数据。

## 5. 数据质量闸门

在进入原因分析前，至少检查：

1. 三个 pool 的 row/document 数量和 source 分布。
2. PDF SHA、PDF 页数、`page_indices` 与渲染图片数量。
3. 每个 image 文件存在、可解码、尺寸合理，且没有同一文档内重复引用。
4. assistant target 是完整 Markdown，末尾换行和 SHA 符合 contract。
5. prompt 与 single/multi 样本形态一致。
6. READoc/PMC 的 document-level split 没有泄漏。
7. PMC full/single 的 document 集合和 split 一一对应。
8. overflow 没有进入 fit；没有静默截断、packing 或随机丢样本。
9. title review status、ruleset、review id、source/target SHA 完整。
10. 生成 `length_manifest.jsonl`，记录 tokenizer、模型、视觉 token contract 和输入 JSONL SHA。

## 6. 复现边界

### 6.1 可以重建等价数据

只要另一台服务器能访问同版本的 PMC/READoc 源数据，并拥有当前 posttrain 工作树，就可以重新生成：

- 标题候选和 silver audit；
- READoc/PMC 训练池；
- 144 DPI 页面图片；
- reviewed mix；
- 32K fit/overflow；
- 数据质量报告和 SHA manifest。

原始 PDF/GT 不需要提交进 posttrain Git，但必须通过 SSH、挂载或明确的数据包提供给重建服务器。

### 6.2 复现本次精确快照的额外条件

要声称“与本次训练使用的数据完全相同”，还需要同时固定：

- READoc/PMC 原始数据版本和源文件 SHA；
- 当前 posttrain 工作树，包括未提交的脚本、规则和 review policy；
- 数据构造脚本的 split seed 和 mix seed；
- PyMuPDF/PyPDF 等 PDF 解析和渲染版本；
- Unlimited-OCR 模型/tokenizer revision；
- 视觉 token contract、prompt 和 EOS 口径；
- 所有 accepted review/silver decision manifest。

只 clone 旧的 `ms-swift-patch` commit，或只拿到原始 PDF/GT，不能自动保证得到本次 3,936/588 的完全同一划分。

### 6.3 分析 agent 的停止规则

另一台服务器的 agent 应先在独立目录重建和对账，再查看 8 个失败案例。若出现以下任一情况，应停止归因并先报告：

- 源 PDF/GT SHA 不一致；
- pool 或 split 数量不一致；
- 页数、页序或图片映射不一致；
- target/prompt/heading 数量不一致；
- tokenizer 或视觉 token contract 不一致；
- 发现当前工作树代码不是本次训练实际使用的版本。

只有对账通过后，才能比较“数据问题、训练问题、推理参数问题”三类原因。

## 7. 相关文件

完整的历史处理说明和脚本用法见：

```text
docs/archive/legacy/ms_swift_from_source_2026-08-23.md
docs/archive/legacy/ms_swift_data_pipeline_2026-08-18.md
ms_swift_title_mask/data_assets/STATUS_ZH.md
```

posttrain Git 不提交 PDF、渲染页图、模型权重、adapter、日志或冻结大数据 payload；`data_assets/` 只保留状态说明和 `.gitignore`。因此新服务器必须显式连接源数据或接收由源数据重建出的数据包。
