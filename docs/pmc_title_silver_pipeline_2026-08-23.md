# PMC 标题 Silver 候选处理（2026-08-23）

## 结论

本轮参考 `pdf评估和gt/docs/10_middle_json到标题GT.md` 的可复用原则，完成了 PMC
`middle.json` 的只读规则处理，但没有照搬其中 PaddleX 重算、B1-B14 中文业务规则、Excel
行号或 33 份样本专用冻结脚本。

当前产物是**可追溯的 silver 结构候选和 review 队列，不是 gold，也不是可直接交给
ms-swift 的训练 JSONL**。原始 `middle.json` 和 PDF 均未修改，因此不会因本轮筛选造成
PDF 与 Markdown 错位。

## 固定训练 Input / Output

本方案的训练契约不变：

- **input**：原始 PDF 确定性渲染后的单页图像，或按真实页序排列的连续多页图像；
- **output**：与全部 input 页面严格对应的一份连续完整 Markdown，不插 `<PAGE>`；
- **不作为 input**：`middle.json`、MinerU block、bbox、已有 Markdown 或标题列表；
- `middle.json` 只在离线数据构建阶段用于生成、定位和审计完整 Markdown GT。

`pmc_title_weighted` 或 `pmc_title_only` 只改变完整 assistant response 上逐 token 的
`loss_scale`。input、完整 Markdown output、`input_ids` 和 labels 均不变；title-only
不是“只输出标题”。本文件生成的候选用于判断完整 Markdown 中哪些 heading span 可以
参与标题 loss，不定义另一种模型 I/O。

核心结论：

- 原始 1,486 篇、26,690 页和 48,665 个 `title` block 全部守恒；没有 schema 失败。
- 44,049 个现有标题可保留为规则级结构 silver；4,616 个现有标题需要结构 review。
- 从普通 `text` 中只留下 179 个 promotion 候选，全部仍是 `REVIEW_REQUIRED`。
- 8,221 个候选含多行源文本，单独进入文字视觉复核，不与标题层级问题混在一起。
- 规则没有自动 relevel、promote、demote、split、merge、删正文或修 LaTeX。

## 输入事实

输入目录：

```text
/Users/guofengjiao/Documents/pmc_v26_pdf_gt_20260816/
  documents/<pmcid>/middle.json
  documents/<pmcid>/document.pdf
```

全量 schema 审计结果：

| 项目 | 结果 |
|---|---:|
| 文档 | 1,486 |
| 页面 | 26,690 |
| `page_size` | 全部 `[595, 842]` |
| `page_idx` | 全部从 0 连续递增 |
| `text` block | 262,182 |
| `title` block | 48,665 |
| `image` block | 11,828 |
| `table` block | 9,764 |
| `interline_equation` block | 4,488 |
| `list` block | 3,546 |

所有文件均为 `_backend=vlm`、`_version_name=3.1.14`。标题层级字段统一是
`text_level`，不是 `level`；span 实际统一使用 `content`。处理器仍兼容参考文档描述的
`spans[].text` 和嵌套 `pdf_info.pdf_info`，但不会把 `preproc_blocks` 再读一遍。

标题层级原始分布：

| `text_level` | 数量 |
|---:|---:|
| 1 | 22,991 |
| 2 | 18,387 |
| 3 | 6,329 |
| 4 | 902 |
| 5 | 56 |

## 实现位置

```text
scripts/data/pmc_title_rules.py
scripts/data/build_pmc_title_candidates.py
tests/test_pmc_title_rules.py
```

`.gitignore` 的根数据目录规则已从 `data/` 收紧为 `/data/`，否则 Git 会错误忽略
`scripts/data/` 下的处理代码。

全量命令：

```bash
python scripts/data/build_pmc_title_candidates.py \
  --input-root /Users/guofengjiao/Documents/pmc_v26_pdf_gt_20260816 \
  --output-root /Users/guofengjiao/Documents/pmc_v26_pdf_gt_20260816_processed_title_gt_v1 \
  --review-shard-size 500
```

小样本可重复传 `--doc-id pmc...`，或使用 `--limit N`。输出已存在时必须显式传
`--overwrite`；替换期间会先保留临时备份，成功后才删除旧派生产物。为保证
`--overwrite` 永远不能触碰源 PDF/JSON，input 和 output 必须是互不包含的同级目录。

## 输出格式

最终输出位于：

```text
/Users/guofengjiao/Documents/pmc_v26_pdf_gt_20260816_processed_title_gt_v1/
  document_manifest.jsonl
  title_candidates.jsonl
  review_queue.jsonl
  text_review_queue.jsonl
  content_quarantine.jsonl
  review_shards/shard_00000.jsonl ... shard_00009.jsonl
  summary.json
```

各文件用途：

| 文件 | 行数 | 用途 |
|---|---:|---|
| `document_manifest.jsonl` | 1,486 | 源 JSON/PDF 相对路径、SHA-256、页数、计数和文档 flags |
| `title_candidates.jsonl` | 48,844 | 全部保留标题和少量 promotion 候选 |
| `review_queue.jsonl` | 4,795 | 结构、层级、caption、重复或 promotion 复核 |
| `text_review_queue.jsonl` | 8,221 | 多行文本拼接的独立视觉复核 |
| `content_quarantine.jsonl` | 1,486 | 正文/资源质量 flags，不直接改变标题标签 |

候选记录至少包含：

```json
{
  "schema_version": "pmc-title-candidates-v1",
  "ruleset_version": "pmc-title-rules-2026-08-23-v1",
  "candidate_id": "cand_...",
  "doc_id": "pmc...",
  "source_middle_json": "documents/pmc.../middle.json",
  "source_sha256": "...",
  "page_idx": 0,
  "pdf_page": 1,
  "page_size": [595, 842],
  "block_id": "blk_...",
  "block_path": "para/1",
  "bbox": [0, 0, 100, 20],
  "text": "source-faithful text",
  "line_texts": ["source line 1", "source line 2"],
  "display_text": "review-only line join",
  "input_type": "title",
  "input_level": 2,
  "numbered_level": 2,
  "operation": "retain",
  "proposed_level": 2,
  "label_grade": "silver_rule",
  "review_status": "AUTO_SILVER",
  "text_review_status": "SOURCE_SINGLE_LINE",
  "reason_codes": [],
  "evidence": []
}
```

`block_id` 由 `doc_id + page_idx + block_path + block 内容` 稳定生成，不依赖 Markdown
行号。`page_idx` 始终是 0-based，`pdf_page` 始终是 1-based。`display_text` 只用于 review，
不能直接当作已冻结的训练 target。

## 规则边界

### 现有 `title` block

满足以下条件时只执行 `retain`，标成结构级 `AUTO_SILVER`：

- 非空；
- `text_level` 在 1-6；
- 没有编号层级冲突或向下跨级；
- 不像 figure/table/box/algorithm caption；
- 不含 `Long description`；
- 不属于同文档规范化重复标题；
- 不超过 320 字符或 50 个词。

出现风险时仍保留原 block、原层级和原文字，只进入 review。编号点数只作为证据，绝不
自动覆盖 `text_level`。例如 `pmc13432129` 的 `1. Familiarization with the Data.` 是
三级标题下的步骤，虽然编号深度为 1，不能自动改成 H1。

重复标题也不自动删除。`pmc13294164` 有 22 个 `Proof`，可能是合法重复；稳定 ID 仍按
文档、页、block 路径区分。

### 普通 `text` block

只有两类内容能成为 promotion 候选：

- 精确 canonical heading，例如 `Introduction`、`Author contributions`；
- 高置信显式编号且满足标题式大小写、长度和上下文约束的短文本。

promotion 永远是 `REVIEW_REQUIRED`。以下内容由规则直接排除，不生成 promotion：

- 首页或地址式作者机构；
- References 区域和含正常/破损年份的参考文献；
- 公式、统计表达式、日期；
- figure/table/box/algorithm caption；
- `Long description` 内部的流程标签；
- 后继为 list，或与相邻同级编号直接构成列表的文本；
- 紧邻已有同名 title 的重复 text；
- reviewer/response 区域中的章节词；
- 页面顶部右侧的短运行页眉。

父标题紧跟子标题仍保留，例如 `2. Strengths of the Manuscript` 后接
`2.1 Practical Relevance`；不能把它当同级列表过滤。

### 多行标题文字

原始 `text` 保持 source-faithful 拼接，`line_texts` 保留每行，`display_text` 仅为 review
插入行间空格。8,221 条候选标记 `MULTILINE_SOURCE_TEXT`，其中 449 条还含
`HYPHENATED_LINE_BREAK`。它们进入独立 `text_review_queue.jsonl`，但不会仅因换行问题
否定结构层级。

最终交叉分布：

| 输入类型 | 结构状态 | 文字状态 | 数量 |
|---|---|---|---:|
| title | `AUTO_SILVER` | `SOURCE_SINGLE_LINE` | 35,960 |
| title | `AUTO_SILVER` | `VISUAL_REVIEW_REQUIRED` | 8,089 |
| title | `REVIEW_REQUIRED` | `SOURCE_SINGLE_LINE` | 4,486 |
| title | `REVIEW_REQUIRED` | `VISUAL_REVIEW_REQUIRED` | 130 |
| text promotion | `REVIEW_REQUIRED` | `SOURCE_SINGLE_LINE` | 177 |
| text promotion | `REVIEW_REQUIRED` | `VISUAL_REVIEW_REQUIRED` | 2 |

即使是 35,960 条最严格规则子集，也只是单行 source silver，并未逐条对 PDF 人工验真。

## 最终 review 统计

结构 review 共 4,795 条，其中 4,616 条来自现有 title，179 条是 text promotion。
reason 可重叠，不能直接相加：

| reason | 候选数 |
|---|---:|
| `DUPLICATE_TITLE` | 4,442 |
| `CAPTION_LIKE_TITLE` | 122 |
| `LONG_DESCRIPTION_TITLE` | 108 |
| `SUSPICIOUS_NUMBERING` | 35 |
| `LEVEL_SKIP` | 27 |
| `NUMBER_LEVEL_CONFLICT` | 18 |
| `TITLE_TOO_LONG` | 1 |
| `CANONICAL_HEADING_TEXT` promotion | 86 |
| `NUMBERED_TEXT_BLOCK` promotion | 93 |

Luna 对规则收紧前的 promotion 做了两批 context-only review：

| 初始批次 | likely | obvious non-heading | ambiguous | 最终规则保留 |
|---|---:|---:|---:|---:|
| 编号 text 116 条 | 93 | 15 | 8 | 93 |
| canonical text 175 条 | 74 | 89 | 12 | 86 |

最终规则只吸收了可泛化的 obvious/list 排除项。保留的 179 条仍没有视觉证据，Luna 结果
不能把它们直接冻结成 gold。

## 内容质量问题

内容 flags 与标题 reason 分开保存。下面的“文档数”和“发生次数”不是同一个口径：

| flag | 文档数 | 发生次数 | 一个 case |
|---|---:|---:|---|
| `MISSING_IMAGE_ASSET` | 1,486 | 23,024 | `pmc13425563` 引用 `assets/table_p9_o3_table.png`，归档未带资源 |
| `PREPROC_DUPLICATES_PARA` | 1,486 | 26,690 页 | 所有页的 `preproc_blocks` 与 `para_blocks` 相同，处理器只读后者 |
| `JOINED_METADATA` | 904 | 3,310 | `pmc13424892`：`Systematic ReviewAcademicSubjects/...` |
| `BAD_LATEX` | 355 | 2,552 | `pmc13431538`：`\\lef t`、`\\fra c`、`\\su m` |
| `EMPTY_PAGE` | 22 | 35 页 | `pmc13252203` 的 `page_idx=17-19` 为空 |
| `IMAGE_ONLY_TABLE` | 45 | 178 | `pmc10933458` 第 21 页 `para/2` 只有 `<img>`，没有 HTML `<table>` |

全量缺图片资源是归档内容事实，不代表 1,486 篇标题全部无效。标题层级训练与完整 Markdown
训练必须分开准入：前者可以继续审计标题，后者不能直接把含失效 `image_path`、坏公式和
粘连元数据的 raw `middle.json` 全量序列化。

## 训练准入

这批文件不能直接传给 ms-swift。后续顺序应是：

1. 冻结规则版本、源 SHA-256 和文档 split；
2. 对 4,795 条结构 review 和 8,221 条文字 review 做分层视觉抽检/复核；
3. 另写 review decision 文件，保存 reviewer/model、prompt 版本、原始响应、结构化决定、
   置信度和错误，不覆盖候选文件；
4. 只有无 unresolved 且能映射回稳定 `block_id` 的决定才能进入 frozen title GT；
5. 再由独立 PMC serializer 从 PDF 页面生成图像，并输出 ms-swift 的
   `messages + images + channel + meta`；
6. `AUTO_SILVER` 可用于低比例 PMC 弱监督，不得替代真实 READoc held-out 评估。

在固定的“页面图像 -> 完整 Markdown”契约下，44,049 条结构 silver 只能用于确定完整
target 内可参与 title loss 的 heading span，不能被导出成标题列表 output。涉及标题文字
token 时，首个保守 heading 池只能从 35,960 条单行且结构无规则风险的记录开始，并仍需
PDF 抽检。完整 Markdown target 还必须通过正文、表格、公式和图片资源的独立
serializer/quarantine；即使某个 profile 把正文 `loss_scale` 设为 0，正文仍保留在 output
和 labels 中。

## 验证

规则单测：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_pmc_title_rules.py' -v
```

当前测试覆盖 schema、`text_level`、0/1-based 页码、稳定 ID、`preproc_blocks` 去重、
编号一致/冲突、重复和 caption、坏 LaTeX/空页分离、promotion 误报、reviewer/页眉过滤、
父子标题相邻、输出可复现性、危险输出路径拒绝、PDF 可读性及 PDF/JSON 页数严格对齐。

全量输出还验证了：manifest 全部 `status=OK`、candidate ID 无重复、所有
`pdf_page=page_idx+1`、PDF 页数严格等于 JSON 页数，且 source title 数严格等于 48,665。
