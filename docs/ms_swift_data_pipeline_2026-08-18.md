# ms-swift 数据准备、混合与标题 Loss 方案（更新于 2026-08-23）

> **历史设计文档，已不再是当前执行口径。** 2026-08-23 起只执行
> [`../ms_swift_title_mask/TRAINING_PLAN_ZH.md`](../ms_swift_title_mask/TRAINING_PLAN_ZH.md)：仅
> `HUMAN_ACCEPTED` 与规则冻结、可追溯的 `SILVER_ACCEPTED` 数据，同数据 full CE / title-mask 两组，
> decoder-backbone LoRA。本文未冻结的 silver candidate、S10/S20、title-weighted 和 full parameter 方案均不进入当前训练。

> 用途：交给另一台服务器上的 Codex 直接实施。训练框架固定为最新 `ms-swift`，模型固定为 Unlimited-OCR，目标是“页面图像 -> 一份连续、无 `<PAGE>` 的完整 Markdown”。本文只规定数据与 loss 改造，不重复模型 LoRA、R-SWA 和长序列 kernel 方案。

## 0. 执行结论

1. **READoc 是主数据。** 使用原始 PDF 与整篇 Markdown GT，第一版只构造“整篇页面列表 -> 整篇连续 Markdown”。READoc 没有可靠页级 target，对它按页切 target、按 token 截 target 或随意构造 2-4 页窗口都不成立。
2. **PMC 合成数据是低比例弱监督。** `middle.json` 有页级 block 和 `merge_prev`，可以构造严格过滤后的单页与连续 2-4 页窗口；不能把原始 `middle.json` 直接序列化后全量训练。
3. **先做 READoc-only baseline，再做 10% PMC A/B。** 混合比例按 assistant target token 预算计算，不按文档数或 optimizer step。10% 无回退后才试 20%，第一阶段不超过 20%。
4. **长度只认真实编码后的 `L=R+T`。** 页数和字符数只用于预筛。必须用训练时相同的 Unlimited-OCR template、图像设置、prompt 和 EOS 口径测量 `R/T/L`，生成 4K、8K、16K、24K、32K 和 overflow manifest。H100 上 4K/8K 只做短 smoke，随后直接探测 16K、24K、32K，正式训练使用现有 ms-swift 能稳定 backward 的最高档，不把 8K/16K 写成数据硬上限。
5. **禁止静默截断和 packing。** READoc 超长整篇样本进入下一长度档或 overflow；PMC 只能按完整页边界缩小窗口并重新生成 target。当前 R-SWA 每条序列只有一个 prefix，第一版不启用 packing。
6. **主线仍是完整 target CE。** 只有规则和 PDF 审计都通过的整篇/窗口才能使用“标题 2、正文 1”；未通过的 READoc/PMC 都保持完整 response 权重 1。title-only 保留为技术备选，不进入当前主实验队列。
7. **数据量够做 LoRA 后训练和 PMC 10% 消融，但不够重新训练通用长上下文能力。** READoc 的 2,233 篇整篇 GT 是主监督；PMC 的标题数量足够做低比例辅助。`>16K` READoc 只有 333 篇，`>32K` 只有 66 篇，因此不能仅靠本数据宣称模型已经学会稳定的 64K/96 页泛化，也不支持当前阶段做大范围全参训练。

`train_short.jsonl` 只保留作旧格式 smoke 示例；标题统计和 PDF 抽检均以本机重新下载的
完整 2,233 篇 READoc 为准。

## 1. 已知数据事实

### 1.1 READoc

目录形态：

```text
READoc/
  arxiv.zip                    # arxiv/pdf/<doc_id>.pdf
  github.zip                   # github/pdf/<doc_id>.pdf
  zenodo.zip                   # 只有 PDF，无 GT；不进入训练
  arxiv_ground_truth/<doc_id>.md
  github_ground_truth/<doc_id>.md
  metadata/                    # 文件完整性信息，不是训练 split
```

- arXiv：1,009 篇。
- GitHub：1,224 篇。
- 总计：2,233 篇。
- Zenodo：1,343 份 PDF，但没有配对 GT，必须排除，不能用空 target 或伪标签混入 SFT。
- 当前未发现可直接复用的官方 train/val/test split；`metadata/verified.jsonl` 是文件 hash/完整性记录。
- GT 是整篇 Markdown，没有可信的 page-to-Markdown 边界。
- arXiv GT 是 Markdown 外壳加大量 LaTeX：969 篇含行内公式，808 篇含块公式，660 篇保留 LaTeX environment。普通公式、表格、脚注正文不能因为含反斜杠而删除。
- GitHub GT 全部使用 CRLF，806 篇含 fenced code，至少 38 篇含 HTML comment。清洗时先保护 code fence，再把 CRLF 规范为 LF。
- CommonMark parser 在 fenced code 外共解析到 34,861 个 heading：arXiv 14,706 个，GitHub 20,155 个。数量足以支持标题加权消融，但 heading level 是原始作者/转换器的 Markdown 写法，不是规范化后的语义树标签。
- arXiv 有 920/1,009 篇把 Abstract 系统性编码为 H6。排除两端为 Abstract 的转换后，arXiv 仍有 73 篇、GitHub 有 237 篇发生向下跨越超过一级。跨级不等于 GT 一定错误，但说明不能把 READoc 的 `#` 数量直接当作完美标题层级 gold。
- 少量 arXiv GT 含未解析的源态 LaTeX 残留，如 `\ref{}`、`\cite{}`、`\label{}`、`\footnotetext` 和 `@@bibref`。命令本身是合法 LaTeX，但此处与 PDF 的已渲染文字不一致，应打 `SOURCE_RESIDUE` 标记并隔离检查。
- 既有长度审计给出的先验是：8K 覆盖 65.29%，16K 覆盖 85.09%，32K 覆盖 97.04%，arXiv 的 8K 覆盖仅 29.93%。该数字不能直接驱动新数据构建；内网实现必须记录 tokenizer/model revision、ms-swift commit、图像设置和 length manifest，并按第 6 节全量重测。
- `train_short.jsonl` 的 `prompt/target/images` 旧式 schema 能被 ms-swift 自动转换，但它只作输入格式示例，不是 clean artifact：末尾换行、source/meta 枚举和 split 尚未统一。新 builder 必须重新校验并输出本文第 2 节的标准 `messages` schema。

#### 1.1.1 READoc 标题全量规则审计与 PDF 抽检

2026-08-23 从官方 `lazyc/READoc` 下载有 GT 的完整数据到：

```text
/Users/guofengjiao/Documents/datasets/READoc/
  archives/{arxiv.zip,github.zip}
  ground_truth/{arxiv,github}/<doc_id>.md
  manifests/
```

两份 ZIP 的 SHA-256 与官方 LFS oid 一致，`unzip -t` 通过；PDF 与 GT ID 集合分别按
1,009/1,224 完全一一对应。Zenodo 没有 GT，因此没有下载。

使用 CommonMark parser 对全部 2,233 篇完整 target 做 fenced-code-aware 审计：

| 风险 | arXiv 文档 | GitHub 文档 | 一个 case |
|---|---:|---:|---|
| 没有 H1、但有其他 heading | 58 | 93 | `arxiv/0911.4272` 的 PDF 题名明显居中放大，GT 却把题名写成普通文本 |
| 完全没有 Markdown heading | 6 | 0 | `arxiv/1102.0638` 的 thesis title 和章节均为普通/强调文本 |
| 多个 H1 | 2 | 157 | `github/10287083` 的三个 H1 在 PDF 中确实都以相同大号粗体渲染 |
| 空 heading | 2 | 8 | `arxiv/1607.02743` 的 PDF 有完整论文题名，GT 首行却只有 `#` |
| 两端均非 Abstract 的向下跨级 | 73 | 237 | `github/148569835` 的 H2 `1.` 跳到 H4 `1.1`，PDF 也保留了两档字号 |
| H6 Abstract 格式约定 | 920 | 0 | `arxiv/0709.4466` 的 `Abstract` 是小号粗斜体，随后章节是另一种 section 样式 |
| 重复规范化 heading 文本 | 48 | 130 | 同名小节可能合法重复，不能按文字自动删除 |
| 超过 200 字符或 40 词 | 2 | 4 | `github/118358642` 把整段迁移说明写成 H3 |

PDF 对照说明规则风险不能直接等同于错误：

- `arxiv/0709.4466` 的 H6 Abstract 是对小号粗斜体摘要标签的系统性编码约定，不是随机
  OCR 错误；但 H6 也不是可跨数据集泛化的语义层级，不能直接给它当通用 level gold。
- `github/10287083` 的多个 H1，以及 `github/148569835` 的 H2 到 H4 跳级，都忠实反映
  README 渲染样式，不能按“树不连续”自动改层级。
- `arxiv/0911.4272`、`arxiv/1607.02743` 是明确坏 case：PDF 有清晰题名，GT 分别漏掉
  H1 标记或只留下空 H1。
- `github/118358642` 的 PDF 顶部是带 `RealWorld` 文字的 logo，GT 空 H1 丢失了该视觉题名；
  后面的长 H3 则是实际粗体公告，是否作为语义标题仍需任务规范决定。
- CommonMark parser 正确忽略了 8,211 个 fenced code block 内的 1,938 行 `#` 代码；未 fenced
  代码仍可能被识别成 heading，例如 `arxiv/1210.6232` 的 JAGS 注释，因此这类命中仍进 review。

标题派生流水线已经在 `scripts/data/build_readoc_title_gt.py` 实现并完成全量运行，产物为：

```text
/Users/guofengjiao/Documents/datasets/READoc_title_gt_v1/
```

全量 2,233 篇、22,290 页均可解码，PDF/GT 集合完整，原始 ZIP 和 GT 复算 SHA-256 后均未
变化。派生集只落了 922 项带版本决定的标题改动：920 个 arXiv `H6 Abstract -> H2`、
`arxiv/0911.4272` 首行提升为 H1、`arxiv/1607.02743` 空 H1 恢复为 PDF 题名。GitHub 的多
H1、跨级、重复标题、空 H1，以及 23 篇“疑似整棵树少一级”的文档都没有自动改。

派生后有 1,552/2,233 篇结构 silver candidate（arXiv 832、GitHub 720），其余 681 篇为
`REVIEW_REQUIRED`。`review_queue.jsonl` 有 2,429 条候选并分成 10 个 shard；所有候选保留
文档级 PDF 页数、首页字体/文字、书签、上下文和稳定 ID；922 条 accepted change 则逐条
保留源/目标/PDF 哈希、具体 PDF 页码及页码证据类型。这 1,552 篇仍不是 title gold，也不能
仅凭该状态直接进入 title-weighted 训练。

下一步从 1,552 个 silver candidate 和 681 个风险文档中按 source、H1-H6、风险桶和长度桶
分层抽取 PDF 页面，核对标题检测、文字、层级、阅读顺序。只有通过视觉验收阈值的整篇
文档才写入 `readoc_title_trusted`；其余样本仍可在完整 Markdown 质量合格时进入
`readoc_full`，但所有 heading 都保持普通权重 1。

### 1.2 PMC 合成数据

解包后的形态：

```text
documents/<pmcid>/
  document.pdf
  middle.json
```

全量审计事实：

- 1,486 篇、26,690 页；PDF 页数与 `len(pdf_info)` 全部一致。
- 全部 PDF 都由 WeasyPrint 66.0 合成，页面固定为 `595 x 842`，存在明显模板域偏差。
- `para_blocks == preproc_blocks` 覆盖全部 26,690 页，只能读取一份，禁止重复序列化。
- 20,952 页普通文本字符数不少于 500；5,738 页少于 500；35 页 `para_blocks` 为空。
- 约 19.2% 是纯视觉/表格/公式页，不能当普通完整 Markdown GT。
- 23,024 个 PNG/JPG/SVG `image_path` 在包内全部缺失；图像仍显示在 PDF 页面上，但 target 不能引用这些路径。
- 9,764 个 table block 中至少 178 个是 image-only table，没有可恢复的 `<table>`。
- 严格只遍历顶层 `para_blocks[].lines[].spans[]` 时，有 22,407 个 `inline_equation` 和 4,488 个 `interline_equation`，合计 26,895 个。4,488 个行间公式有非空 content，但关联 SVG asset 缺失；asset 缺失本身不等于空 GT。
- 公式 content 中可见大量 `\fra c`、`\lef t`、`\rh o` 等可疑命令。数量必须由带版本的检测器重算并记录匹配规则，不能复用递归嵌套后可能重复计数的旧数字。
- 共 48,665 个 title block，全部 1,486 篇文档都含标题；level 分布为 L1 22,991、L2 18,387、L3 6,329、L4 902、L5 56，深层级明显不足。
- 47,179 次相邻标题层级转换中只有 27 次向下跨越超过一级；在 10,032 个可由编号独立推断层级的标题中，10,027 个与 `text_level` 一致，内部一致率为 99.95%。这个数字只证明 `middle.json` 内部自洽，不是对 PDF 真实标题和层级的人工准确率。
- 584/1,486 篇存在规范化后的重复标题。正文、作者、机构和元信息还存在系统性粘词，因此 PMC 始终按 weak label 使用，不能升级为人工 GT。

因此 PMC 的价值是标题、正文阅读顺序、`merge_prev` 和严格可解析表格的辅助监督，不是完整 Markdown 主真值。

### 1.3 数据量是否够用

| 目标 | 当前原始/候选规模 | 结论 |
|---|---:|---|
| READoc 完整 Markdown LoRA 后训练 | 2,233 篇整篇 GT；32K 内 2,167 篇 | **够做主 baseline 和领域适配**，但必须按文档切分并控制 epoch/token 预算 |
| PMC 低比例辅助 | 1,486 篇、20,952 个普通文本不少于 500 字符的候选页、48,665 个 title block | **足够做 S10，S20 也不缺数量**；继续放大只会增加模板偏差 |
| 标题加权或 title-only 消融 | READoc 34,861 个 heading + PMC 48,665 个 title block，原始合计 83,526 个 | **数量够，标签可信度未闭环**；可以做 A/B，不能先声称层级准确 |
| 16K-32K 长样本适配 | READoc `(16K,32K]` 共 267 篇 | **可用于保持/适配已有 32K 能力，不能从头学出稳健长上下文** |
| `>32K`/96 页能力 | READoc 66 篇 `>32K`，其中 96 页仅 4 篇 | **不够形成可靠训练与评估结论**；保留 overflow，不能过采样后冒充数据多样性 |
| `full_backbone/full_decoder` 大范围更新 | 只有 2,233 篇真实整篇 GT | **不够**；当前坚持 `lora_decoder`，全参方案不进主队列 |

以上是清洗前规模；最终要同时报告准入后的文档数、独立文档数、assistant target token 数和长度分布。长文档 token 多不等于版式多，不能通过重复采样同一 PDF 人为解决多样性不足。

普通 90/5/5 切分后，`(16K,32K]` 的 validation/test 各自预计只有约 13 篇，不能支撑稳定的 32K 质量结论。split builder 必须按 source 与长度联合分层，并在 test 内额外冻结一个不参与训练的 `long_eval` 清单：建议从 16K-24K、24K-32K 各保留 20 篇，再覆盖不同页数/文本密度和 arXiv/GitHub。主 test 总分与 `long_eval` 分桶结果分开报告。若实际准入量不足，就把长长度结论写成探索性结果，不通过重复窗口或把 overflow 混入 test 来扩大数字。

标题监督在正式训练前增加一个真实 PDF gold audit：人工标约 1,000 个标题实例，按 L1-L5、重复标题、异常编号、双栏版式及 arXiv/GitHub/PMC 分层，分别评估标题检测、标题文本、层级和顺序。这个审计同时决定 READoc 和 PMC 哪些整篇/窗口能进入 `title_weighted`，不影响先跑 READoc full CE baseline。

## 2. ms-swift 统一数据契约

### 2.1 单页

```json
{"id":"pmc123_p0003","messages":[{"role":"user","content":"<image>document parsing."},{"role":"assistant","content":"# Title\n\nBody..."}],"images":["images/pmc123/page_0003.png"],"meta":{"source":"pmc_synthetic","doc_id":"pmc123","pages":[3],"split":"train"}}
```

### 2.2 连续多页或 READoc 整篇

```json
{"id":"readoc_arxiv_0709.4466","messages":[{"role":"user","content":"<image>Multi page merge."},{"role":"assistant","content":"# Document title\n\nComplete continuous Markdown..."}],"images":["images/arxiv/0709.4466/page_0000.png","images/arxiv/0709.4466/page_0001.png"],"meta":{"source":"readoc_arxiv","doc_id":"0709.4466","pages":[0,1],"split":"train"}}
```

约束：

- 多图 prompt 只放一个 `<image>`。最新版 ms-swift `UnlimitedOCR._encode` 会按 `len(images)` 展开为 N 个占位符。
- 多页 target 是一份连续 Markdown，不插 `<PAGE>`，不输出 bbox/detection markup。
- `images` 必须按真实页序排列；路径使用绝对路径，或相对 `ROOT_IMAGE_DIR` 的稳定路径。
- `meta` 用于构建、审计和采样；ms-swift 训练预处理可能丢弃它，因此 token 统计和混合在生成最终 JSONL 前完成。
- 若实验需要逐样本选择 loss profile，使用 ms-swift 标准顶层字段 `channel`，不能从可能被丢弃的 `meta.source` 猜。原始 source JSONL 不写 profile；只有最终 mix builder 注入受控枚举。
- `id` 必须唯一，并能反查 `source/doc_id/pages/serializer_version`。
- target 必须非空、UTF-8、换行统一为 `\n`，新 serializer 的输出末尾统一保留一个换行；旧 JSONL 必须经过这一步，不能假定已满足。
- 旧字段必须显式归一化，例如顶层 `READoc-arxiv/READoc-github` 统一为 `readoc_arxiv/readoc_github`，`meta.subset` 映射为统一 `source`；`pages` 从有序图片文件名解析，并断言数量等于旧 `n_pages`；最后补上冻结后的 `split`。禁止同一语义保留两套枚举。

所有混合样本统一走 `multi_base/no-crop` 视觉设置，包括只有一张图的 PMC 单页样本：

```bash
export CROP_MODE=false
export IMAGE_SIZE=1024
export BASE_SIZE=1024
export ROOT_IMAGE_DIR=/path/to/processed-data
```

ms-swift 的图像 mode 是进程级设置，不能在同一个训练进程中让部分样本 crop、部分样本 no-crop。`multi_base/no-crop` 是统一视觉模式，不要求样本至少两页，一张图也允许。若以后要做 `single_gundam`，必须作为独立阶段和独立实验，不能混进本文 baseline。

## 3. 公共切分与防泄漏

切分单位必须是文档。确定性 PDF 渲染和 template 长度审计可以在切分前完成，以便按 source 与长度联合分层；任何页面/窗口样本派生、混合采样和训练 JSONL 生成都必须在文档级 split 冻结后进行。

建议固定：

```text
train 90% / validation 5% / test 5%
```

实现要求：

1. 分别在 `readoc_arxiv`、`readoc_github`、`pmc_synthetic` 内按 source 与长度档联合分层；READoc test 还要满足第 1.3 节 `long_eval` 的固定配额。
2. 使用稳定 hash，例如 `sha256("uocr-v1:" + source + ":" + doc_id)`，按 hash 区间分配 split；不能依赖文件遍历顺序或 Python `hash()`。
3. 同一 `doc_id` 的整篇样本、页面、重叠窗口和所有长度版本必须继承同一 split。
4. 在 READoc 与 PMC 之间对规范化 target 做 exact hash，并对正文做 MinHash/SimHash 近重复检查；跨源重复只保留在一个 split。
5. validation/test 不参加混合过采样，不从 train 文档派生任何页面。
6. READoc test 是训练内保留集；如果还要报告官方 READoc benchmark，必须确认官方评测文档未进入 train。

输出 `split_manifest.jsonl`，至少含：

```text
source, doc_id, split, pdf_path, gt_path, pdf_sha256, raw_target_sha256
```

## 4. READoc 处理

### 4.1 配对和渲染

1. 从 `arxiv.zip/github.zip` 建立 PDF 清单，不以解压顺序为准。
2. 用 `source + doc_id` 与对应 ground-truth Markdown 一对一配对。
3. 拒绝缺 PDF、缺 GT、空 GT、PDF 无页或 PDF 解码失败的条目；写入 reject manifest，不静默跳过。
4. 使用 PyMuPDF 按原始页序渲染 RGB PNG，建议 144 DPI、`alpha=False`。Unlimited-OCR template 仍负责 pad/resize 到 1024。
5. 校验渲染页数等于 PDF 页数，并记录每页宽高、文件大小与 SHA256。

目录建议：

```text
processed/readoc/images/<source>/<doc_id>/page_0000.png
processed/readoc/jsonl/<split>.jsonl
processed/readoc/manifests/documents.jsonl
processed/readoc/manifests/lengths.jsonl
processed/readoc/rejects.jsonl
```

### 4.2 Markdown 最小清洗

READoc GT 优于 PMC，不做重写式“纠错”。先保存 raw GT，再使用 Markdown AST（例如 `markdown-it-py`）产生候选 clean GT；任何清洗都不能改写 raw 文件。只有规则已经通过 PDF 对照审计的 clean GT 才进入 baseline，无法证明的文档进入 quarantine。处理顺序如下：

标题层派生已经可以独立重放：

```bash
python scripts/data/build_readoc_title_gt.py \
  --input-root /Users/guofengjiao/Documents/datasets/READoc \
  --output-root /Users/guofengjiao/Documents/datasets/READoc_title_gt_v1 \
  --overwrite
```

该命令直接从 ZIP member 校验每份 PDF，不解压或改写原始归档。输出包含完整派生 Markdown、
`document_manifest.jsonl`、`pdf_profiles.jsonl`、`heading_inventory.jsonl`、
`title_candidates.jsonl`、`review_queue.jsonl`、`change_manifest.jsonl`、冻结 review policy、
summary 和 validation。校验器要求 2,233 篇全覆盖、原始哈希不变，并逐行证明所有 diff 都
有已接受的标题决定。

这个产物只完成标题层，不执行下列正文最小清洗。因此 manifest 将 `full_ce_eligible` 和
`title_weighted_eligible` 保留为 `null`，只提供 `title_weighted_silver_candidate`。后续正文
cleaner 应以派生 Markdown 为输入并记录 parent hash；不能把 silver candidate 直接改名成
`readoc_title_trusted`。

- 将 CRLF/CR 统一为 LF，去 BOM 和文件末尾 NUL，末尾只保留一个换行。
- 在 fenced/inline code 外检测未解析的 `\ref{}`、`\cite{}`、`\label{}`、`\footnotetext`、`@@bibref`。命中文档进入 `SOURCE_RESIDUE` quarantine，不自动猜测引用号、脚注或参考文献文本。
- 在公式和 LaTeX environment 中检测 `\dfra`、`\fra c`、`\lef t` 等已知可疑命令，写入 `BAD_LATEX` flag。检测器必须避开代码内容并记录命中 span；不自动猜修复。
- fenced/inline code 外的 `<!-- ... -->` HTML 注释在确认未渲染后删除并记录数量；代码内看起来像注释的字符串原样保留。
- 普通链接 `[visible text](url)` 只有在 PDF 对照证明 URL 隐藏时，才保留 visible text 并删除 URL；页面上直接印出的裸 URL 保留。
- Markdown 图片/徽章只有在 PDF 对照证明资源路径隐藏时才删除路径。明确可见的非空 alt/caption 保留；badge/build status 等是否可见由验证规则决定，不按名字直接删除。
- reference-link definitions、anchor、源文件路径和 YAML front matter 都先打 flag。只有确认对应内容未渲染时才从 clean view 删除；否则保留或 quarantine。
- Setext heading 可规范成 ATX heading，标题层级不改变。
- 保留普通 LaTeX、代码块、可见表格、公式、列表、可见脚注正文和强调语义。
- 规范空白和空行，但不全局修复重复词、拼写、数学表达式或作者名。

每次清洗必须同时保留 `raw_target_sha256`、`clean_target_sha256`、逐项 diff、清洗规则版本和修改计数。每类规则先随机抽查 arXiv/GitHub 各至少 50 篇，把原 PDF、raw GT、clean GT 并排检查；规则未验收前不得批量放行。

`documents.jsonl` 每篇至少记录：

```text
source, doc_id, split, pdf_path, gt_path, pdf_n_pages,
pdf_sha256, raw_target_sha256, clean_target_sha256,
gt_bytes, gt_lines, heading_count, code_fence_count,
inline_math_count, display_math_count, table_count,
html_comment_count, source_residue_flags, reject_reason,
clean_rule_counts, clean_diff_path, cleaner_version, renderer_version
```

`SOURCE_RESIDUE` 与结构异常都进入 quarantine，不与“缺文件/PDF 不可读”混在同一个硬错误统计中。人工确认某条 source residue 已真实渲染且 clean target 可精确恢复后，才通过带版本的 allowlist 放回数据池。

### 4.3 样本边界

baseline 只生成整篇样本：

```text
images = 该 PDF 的全部页面
target = 该文档的完整 clean Markdown
prompt = <image>Multi page merge.
```

严禁：

- 给前 N 页配整篇 target。
- 按 target token 比例猜页边界。
- 从中间截一段 Markdown 配若干页。
- 直接右截 target 以满足 `max_length`。

READoc 若要生成单页或 2-4 页窗口，必须另做“页图文本 -> Markdown AST block”的单调对齐：

1. Markdown 解析为标题、段落、列表、代码、表格、公式等 block。
2. 从 PDF 每页提取可见文本，仅用于定位，不作为新 GT。
3. 对规范化文本做文档内单调动态规划对齐。
4. 只保留 block 覆盖完整、置信度达标且窗口边界不切断表格/代码/公式/段落的候选。
5. 对齐阈值必须先经人工抽查确定；第一版不实现、不纳入 baseline。

原生一页 READoc 文档可以作为单页样本，但不能靠切整篇 GT 制造单页 target。

### 4.4 READoc 准入

整篇样本需满足：

- PDF/GT 一一配对且页数大于 0。
- clean target 非空且不含 `<PAGE>`。代码或正文中实际可见的本地路径原样保留；只有 Markdown destination 等经 PDF 对照确认不可见的资源路径才从 clean view 删除。任何未解决的 hidden-resource/HTML-comment flag 进入 quarantine。
- 不含未处理的 source-residue flag；allowlist 必须在 manifest 中可追溯。
- Markdown fence、公式 delimiter 和 HTML table 基本结构通过检查；机械检查只用于发现异常，必须忽略代码内容以及 `\\[2pt]` 一类 LaTeX 行断/间距语法，不能仅按反斜杠计数拒绝。
- 实际 template 编码长度落入当前训练桶。
- 超限样本进入 overflow，不截断。

## 5. PMC 合成数据处理

### 5.1 输入规则

- 只读取 `pdf_info[].para_blocks`，忽略内容完全相同的 `preproc_blocks`。
- `page_idx` 必须连续且与 PDF 页数一致。
- 先按 PMCID 切 split，再生成页面和窗口。
- 页面渲染设置与 READoc 相同。
- `image_path` 只用于质量统计，永远不进入 target。

### 5.2 block 到 Markdown

必须使用结构化解析器，不用正则直接改 JATS/HTML。

| block | 输出规则 | 严格池拒绝条件 |
|---|---|---|
| `title` | `text_level` 限制到 1-6，输出对应 ATX heading；合并 block 内可见 text spans | 空标题、层级缺失且无法推断 |
| `text` | 按 line/span 顺序连接，规范空白；不同段落空一行 | 空内容、明显元信息粘连或异常重复达到审计阈值 |
| `list` | 保留可见编号/项目符号；无可见 marker 时按普通段落处理，不凭 block type 伪造 `-` | 内容为空或阅读顺序不确定 |
| `image` | 只输出 `image_caption` 可见文字，删除 image body/path | 无 caption 时省略；不得输出占位路径 |
| `table` | 用 XML/HTML parser 提取真实 `<table>`，把 JATS `italic/bold/sup/sub` 等映射为合法 HTML；caption 只输出一次 | 没有真实 `<table>`、image-only、解析失败、行列结构非法 |
| `interline_equation` | 非空且验证通过时输出 `$$...$$` | 空公式、坏命令、花括号/环境不平衡 |
| inline equation span | 验证通过后输出 `$...$` | 同上 |

禁止自动把 `\fra c` 改成 `\frac`。这类错误不只是一处空格，无法证明修复后的公式与页面一致；应将公式 block、页面或窗口送入 quarantine。

### 5.3 `merge_prev` 与窗口边界

- 在同一候选样本内部，`merge_prev=true` 的 text block 与前一个兼容文本 block 合并，不插空段。
- 不跨样本边界借用上一页或下一页文字。
- 严格窗口不能从一个 `merge_prev=true` 的跨页续段开始，也不能在已知下一页继续当前段落时结束。
- 表格、代码样式内容和公式不得被窗口边界切开。
- target 由窗口内页面重新序列化；禁止先生成整篇 target 后按字符截取。

### 5.4 严格池与 quarantine

第一版只产生两个可训练池：

```text
pmc_strict_single          # 一页，普通可见文本 >= 500 chars
pmc_strict_window_2to4     # 连续 2-4 页，每页普通可见文本 >= 500 chars
```

进入严格池必须同时满足：

- 窗口内每一页都无空 `para_blocks`、普通可见文本不少于 500 chars，且不能是纯视觉/表格/公式页；不能用其他长文本页掩盖一张无可靠 target 的页面。
- 无 image-only table、无表格解析失败。
- 无空公式、已知坏命令或公式结构失败。
- target 不含 `assets/`、`image_path`、缺失图片链接、`<table-wrap>`、`<jtitle>` 等 JATS 残留。
- Markdown/嵌入 HTML 可解析，target 非空且没有 `<PAGE>`。
- 对 PDF text layer 的规范化 token coverage 达到经抽样校准的阈值。`0.90` 可作为初始告警线，不把它当人工 GT 证明。
- 粘词率、重复率和异常字符率写入 manifest。先看全量分布并人工检查高分位样本，再确定拒收阈值；禁止未经校准的全局正则修词。

其他样本进入 `pmc_quarantine.jsonl`，记录多个 reason code，例如：

```text
EMPTY_PAGE
LOW_TEXT
VISUAL_ONLY
IMAGE_ONLY_TABLE
BAD_TABLE
EMPTY_FORMULA
BAD_LATEX
BOUNDARY_CONTINUATION
JOINED_TEXT_HIGH
TARGET_PARSE_ERROR
```

quarantine 第一版不训练。不要为了提高数据量把缺失表格/公式替换成伪造占位符。

PMC manifest 每条至少记录：

```text
pmcid, split, pages, pdf_sha256, target_sha256,
normal_text_chars_by_page, block_type_counts, title_count,
table_count, formula_count, bad_table_count, bad_formula_count,
text_layer_coverage, joined_text_score, quality_flags,
serializer_version, reject_reasons
```

### 5.5 PMC 去相关

所有 PDF 来自相同 WeasyPrint 模板，不能让大量相邻窗口淹没真实数据：

- 一个 PMCID 在一个 epoch manifest 中最多选 4 个窗口。
- 同一 epoch 优先选不重叠窗口；不同 epoch 可用固定 seed 轮换。
- 不使用完整 PMC 文档作为第一阶段训练样本。
- 含表格/公式的 PMC 风险子集最多占总 assistant token 的 5%，且只来自严格可解析样本。

## 6. 真实长度测量与分桶

### 6.1 必须测 template 后长度

长度脚本必须加载与训练完全相同的：

- 模型 tokenizer/processor。
- `UnlimitedOCR` template。
- `CROP_MODE=false, IMAGE_SIZE=1024, BASE_SIZE=1024`。
- prompt、图片列表、EOS 和 chat template。

在 `template.set_mode('train')` 后编码每条样本，记录：

```text
L = 非 padding input_ids 长度
R = 完整 CE labels 中第一个非 -100 的位置
T = L - R  # 包含训练 template 实际追加的 EOS
```

这里的 `R` 必须从完整 CE labels 计算，不能从 title-only 的 `loss_scale` 推断。建议额外记录：

```text
n_pages, image_tokens, prompt_tokens, target_tokens, source,
doc_id, split, quality_flags, serializer_version
```

### 6.2 非累计桶

```text
<=4K
(4K,8K]
(8K,16K]
(16K,24K]
(24K,32K]
>32K
```

这些是互斥的审计桶；训练视图是累计并集。例如 `max_length=8192` 的训练集使用 `<=4K` 与 `(4K,8K]` 两桶，不是只训练后一个桶。页数只用于预筛和报表，不设“最多 N 页”代替 token 上限；同样页数的视觉 prefix 和 target 长度都可能差很多。

为每个 split 同时输出：

- 文档数。
- assistant target token 数。
- 总 token 数。
- arXiv/GitHub/PMC 占比。
- 页数分位数。
- overflow reason。

### 6.3 训练规则

- 目标硬件按单张 80GB H100 规划；若实际是不同显存规格，必须重新跑同一探测，不能沿用本结论。
- 4K/8K 只做 5-10 step 链路 smoke，验证数据、R-SWA、loss、保存和 reload，不为 8K 单独跑完整正式训练。
- 在未修改 ms-swift 核心的条件下，以 `micro_batch_size=1` 依次做 16K、24K、32K 的 forward/backward 显存与吞吐探测。每档使用接近上界的真实样本，并记录 warmup 后 peak allocated/reserved、step time 和 tokens/s。
- 将能够连续稳定运行、显存保留合理余量且吞吐可接受的最高档记为 `L_base`。R0、S10 与所有主 A/B 使用同一个 `L_base` 和相同长度 manifest；预计 dense eager R-SWA 更可能停在 16K 或 24K，不能因为是 H100 就预设 32K 一定可行。
- 如果原生路径能稳定跑 32K，直接以 32K 为正式上限，不实施 FlexAttention。如果 `L_base < 32K` 且确认需要覆盖 `(L_base,32K]`，才触发等价 R-SWA 的 memory-efficient attention 与 fused linear CE 改造。
- `--truncation_strategy raise`，任何超限都应在构建阶段解决。
- `--packing false`。当前一个 packed sequence 内没有多个独立 `prefix_len` 的 R-SWA 语义。

READoc 超限：保留整篇样本到更长桶或 overflow。

PMC 超限：按完整页减少窗口长度并重新序列化；一页仍超限则 quarantine。禁止 target token 截断。

## 7. 混合方案

### 7.1 最终保留的实验顺序

所有比例指 assistant target token，不是 row 数：

| 实验 | READoc | PMC strict | 目的 |
|---|---:|---:|---|
| `R0-full` | 100% | 0% | READoc 普通完整 CE baseline |
| `S10-full` | 90% | 10% | 只改变数据混合，检查合成数据本身的收益与污染 |
| `S10-trusted-title-weighted` | 90% | 10% | 行与 token 预算不变，只给审计通过的整篇/窗口 heading 设 2 倍权重 |
| `S20-full` | 80% | 20% | 仅在 `S10-full` 无回退后运行，验证合成占比上限 |

不要直接从 20% 开始，也不要根据 PMC 样本更多就提高占比。plugin full-parity 只做数值和
短跑验收，不算第五套训练实验；title-only 暂不进入主实验队列。

### 7.2 内部分配

READoc token 预算：

```text
arXiv 50% / GitHub 50%
```

这是为了避免 1,224 个较短 GitHub 文档在 row 采样中压过更复杂的 arXiv。若某长度桶无法达到 50/50，记录实际比例，不重复单个 arXiv 文档超过 2 次来硬凑。

PMC token 预算：

```text
strict single 80% / strict 2-4 page window 20%
```

READoc 整篇样本已经承担主要多页监督，因此 PMC 优先补单页识别；仍保留 20% 连续窗口用于合并与跨页阅读顺序。若采用课程训练，最初 10% step 可用 `single/window=90/10`，随后固定为 `80/20`，但 R0/S10 A/B 的总 token 预算和其他超参不变。

所有池先按 source、split、长度和质量标记生成独立 manifest，再用固定 seed 按 target token quota 生成最终 epoch JSONL。不要依赖 ms-swift 对多个文件的 row-level shuffle 来实现 token 配比。

数据混合比例和标题可信度是两个独立维度。READoc 的普通完整 CE 池可以包含“正文/完整
Markdown 合格但标题结构待复核”的文档；进入标题加权实验时，这些文档仍保留原 row 和完整
output，只使用 `readoc_full` channel。只有一篇文档的全部 heading 均通过规则与 PDF 审计，
才切换为 `readoc_title_weighted`。PMC 同理，只允许全部 heading 都已冻结的 strict 窗口使用
`pmc_title_weighted`；不能把 source 名直接等同于标题可信。

### 7.3 S10/S20 准入标准

primary eval 必须是未参与训练的真实 READoc，PMC validation 只能作 secondary：

- 完整 Markdown 相似度不得显著下降。
- 标题文本 F1、标题层级准确率和跨页标题顺序应提升或持平。
- 正文漏字/重复、公式、表格和代码块不得明显回退。
- GitHub 与 arXiv 分开报告，不能用易样本平均掩盖 arXiv 回退。
- 生成必须评估到样本末尾，不能只看 teacher-forcing loss。

S10 在真实 READoc 上回退时，停止 S20；先检查 PMC 模板偏差、粘词和窗口重复，而不是继续加数据。

## 8. 标题 loss

### 8.1 三种任务不能混淆

**完整 Markdown 普通 CE（主线）**

所有 assistant token 权重 1。不改 ms-swift，是 R0/S10/S20 的默认设置。

**标题加权、正文仍训练（推荐消融）**

完整 Markdown target 不变，Markdown heading token 权重 2，其他 assistant token 权重 1。这样仍学习正文生成，R-SWA prefix 也不会被 loss mask 改变。

**完整 Markdown teacher forcing，但只有标题参与 loss（高风险消融）**

body token 仍作为输入，却没有生成监督；推理时正文误差会传到后续标题。它不能替代完整 Markdown 主线，也不等价于“输出标题列表”。

如果任务真正只需要标题树，最干净的做法是把 assistant target 改成标题层级 JSON/Markdown 列表。该任务无需改 Trainer，但不再评估完整 Markdown OCR。

### 8.2 标题识别

先把 target 规范为 ATX heading，再用 Markdown parser 获取 heading token 的源行区间。不能用裸正则，因为 fenced code 内可能出现以 `#` 开头的代码。

标题 span 包含：

- `#` 标记。
- 标题正文。
- 该标题行的终止换行。

不默认包含图注、表题、加粗句或 HTML `<h1>`；若清洗阶段把 HTML heading 规范成 ATX，它们自然进入标题 span。

token offset 必须完全落在 heading span 内或外。若单个 token 同时跨越 heading/body 边界，不能用“有重叠就算标题”猜权重；构建器应报 mapping error，先调整规范化边界并重新验证。

### 8.3 统一实现：完整 labels，独立逐 token 权重

推荐通过 ms-swift 的 `--external_plugins` 注册自定义 template，不修改 ms-swift core。完整 CE 继续使用原生 `unlimited_ocr`；插件另注册 `unlimited_ocr_title_weighted`、`unlimited_ocr_title_only` 与按 `channel` 路由的 `unlimited_ocr_mixed_loss`。

三个插件 template 都继承 `UnlimitedOCR`，并在 `super()._encode(inputs)` **完成默认编码和图像 token 展开之后**执行：

1. 从原始 assistant string 解析 Markdown AST，得到 fenced code 外的 heading 字符 span。
2. 对完整 assistant string 一次性 tokenize，并请求 `offset_mapping`；不把 response 拆成多个字符串分别 tokenize。
3. 从完整 labels 的第一个非 `-100` 位置找到 response 起点，断言 `encoded['input_ids']` 中对应片段与完整 response token ids 完全一致。tokenizer 不支持 offset 或断言失败时直接报错。
4. 断言至少映射到一个 heading token；不能因为 EOS 有权重就放过无标题样本。
5. 建立与最终 `input_ids` 等长的 `loss_scale`，prompt、image token 和 padding 都为 0。
6. `title_weighted`：正文为 1，heading 为 2，EOS 为 1。
7. `title_only`：正文为 0，heading 为 1。默认保留 EOS 权重 1，并把实验名写成 `title_plus_eos`；若要求字面意义上的纯标题，EOS 也设 0。
8. **不得修改完整 labels。** 只写 `encoded['loss_scale'] = token_weights`。

启动参数至少包含：

```text
--loss_scale default
--is_binary_loss_scale false
--use_liger_kernel false
--sequence_parallel_size 1
--average_tokens_across_devices false
```

`is_binary_loss_scale=false` 是为了强制保留独立的 `loss_scale` tensor。即使 title-only 权重只有 0/1，也不能让 ms-swift 自动把 0 改写成 label `-100`。

这条路径不会减少序列长度、attention 或 LM head 计算：正文仍参与 teacher forcing 和前向，只是不产生 loss 权重。非 binary `loss_scale` 还会让 ms-swift 走显式逐 token CE，Liger/fused CE 不生效，因此标题 loss 必须独立做 8K smoke 以及 16K/24K/32K 显存探测。正式消融使用该插件与 native full CE 都能稳定运行的最高共同长度，不能把 title loss 当长序列省显存方案，也不能拿不同长度的数据直接比较。

不要通过普通 `LossScale.get_loss_scale()` 把原 response 拆成 heading/body 字符串再分别 tokenize。SentencePiece/BPE 在分段边界可能产生不同 token，使消融实验连输入都发生变化。

### 8.4 必须修正 weighted loss 的归一化

当前 `Seq2SeqTrainer` 会先把逐 token CE 乘 `loss_scale`，随后默认仍除以 `(labels != -100).sum()`。由于本文刻意保留完整 labels，title-only 会被大量权重为 0 的正文 token 稀释。

在同一个 external plugin 中注册一个 `BaseLoss`，例如 `uocr_weighted_token_mean`。调用自定义 loss 时，`outputs.loss` 已经被 trainer 乘过且 `loss_scale` 已经为 next-token 对齐后的扁平 tensor，所以插件只做求和和归一化，不能再次 roll 或再次相乘：

```python
import torch
from swift.loss import BaseLoss, loss_map


class UOCRWeightedTokenMean(BaseLoss):
    def __call__(self, outputs, labels, *, loss_scale=None, **kwargs):
        if loss_scale is None:
            raise RuntimeError('uocr_weighted_token_mean requires loss_scale')
        local_denom = loss_scale.sum().detach().to(outputs.loss.device)
        if local_denom.item() <= 0:
            raise RuntimeError('batch has no active loss token')
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_denom = local_denom.clone()
            torch.distributed.all_reduce(global_denom)
            # DDP averages gradients, so compensate by world_size here.
            return outputs.loss.sum() * torch.distributed.get_world_size() / global_denom
        return outputs.loss.sum() / local_denom

loss_map['uocr_weighted_token_mean'] = UOCRWeightedTokenMean
```

训练时增加：

```text
--loss_type uocr_weighted_token_mean
```

双卡标准 DDP 时不能让每张卡按各自 title token 数独立归一化。上面的 all-reduce 与 `world_size` 补偿使 DDP 梯度平均后等于全局 active-token mean。它已经自行处理跨卡 token 分母，因此 `average_tokens_across_devices` 必须保持 false，避免 Trainer 再补一次 world-size。`local_denom.item()` 会触发一次同步，smoke 阶段用于强校验；正式训练可在数据和单测证明每个 batch 都有 active token 后去掉逐 step 的本地检查。梯度累计仍是“每个 micro-step 的 active-token mean 再平均”，必须把这一口径写入实验记录。

这段通信逻辑只覆盖纯数据并行。若启用 sequence/context/tensor parallel，必须改用正确的 data-parallel process group 并重新做数值对齐；未完成前 title loss 实验固定 `sequence_parallel_size=1`。

标题加权也使用该归一化，使 title/body 的相对权重为 2:1，同时避免不同 batch 的标题密度改变整体 loss 尺度。

### 8.5 为什么这条路径不改变 R-SWA

当前 Unlimited-OCR collator 用完整 labels 的第一个非 `-100` token 推导 R-SWA prefix：

```python
first_non_ignored = (labels != -100).nonzero()[0]
prefix_len = first_non_ignored
```

本文的 full CE、title-weighted 和 title-only 三条路径具有完全相同的 `input_ids` 与 labels，只是 `loss_scale` 不同。因此 response 起点与 R-SWA attention mask 都不变，复用 Unlimited-OCR 权重的模型语义也不变。

只有当实现者坚持把正文 labels 改为 `-100` 时，才需要进一步改 `deepseek.py`：在 mask 前保存显式 `rswa_prefix_len`，collator padding 后传给 `_build_rswa_attention_mask`，禁止再从 masked labels 推断。该方案改动更大，不作为第一选择。

### 8.6 external plugin 交付形态

建议只新增：

```text
plugins/uocr_title_loss.py
tests/test_uocr_title_loss.py
```

插件注册方式：复制 `TEMPLATE_MAPPING['unlimited_ocr']` 的 `TemplateMeta`，分别修改 `template_type` 和 `template_cls` 后调用 `register_template`；把自定义 loss 写入 `loss_map`。启动时使用：

```text
--external_plugins /path/to/plugins/uocr_title_loss.py
--template unlimited_ocr_title_only
--loss_type uocr_weighted_token_mean
```

这样更新 ms-swift master 时无需反复解决 core 文件冲突。实验必须记录 ms-swift commit、plugin commit、template 名、EOS 是否参与 loss 和 title/body 权重。

### 8.7 title loss 验收测试

至少覆盖：

- response 第一行就是 H1。
- H1 前有作者/摘要正文。
- 多个 H2/H3，中间有长正文、表格、公式和 fenced code。
- code fence 内有 `# not a heading`。
- 没有任何 heading 的样本。
- batch 内不同 prefix、不同长度和 padding。
- mixed batch 内同时含 `readoc_full`、`readoc_title_weighted`、`pmc_full` 与
  `pmc_title_weighted` channel。

必须断言：

1. full CE、title-weighted 和 title-only 的 `input_ids` 与 labels 逐元素一致。
2. 三者从 labels 推导的 response 起点一致，R-SWA attention mask 逐元素一致。
3. weighted 只有 `loss_scale` 不同：prompt/image 为 0、body 为 1、heading 为 2。
4. title-only 的非零 `loss_scale` 只覆盖 heading 及声明保留的 EOS；body labels 虽保留，但权重必须为 0。
5. loss 的分母等于 active weight sum，而不是完整 response token 数；构造常数 logits 的小样本做数值单测。
6. 标题 token 数为 0 的样本被构建器拒绝或进入普通 CE 池，不产生除零 loss。
7. stock `token_acc` 因 labels 完整仍包含正文，不能当 title accuracy；另按 `loss_scale` 报 `title_token_acc`、title/body token 数与 active weight sum。
8. 单卡和双卡各做一次 forward/backward，loss 有限，LoRA gradient 非空；双卡结果与拼成单卡 batch 的 loss/gradient 在容差内一致。

### 8.8 READoc + PMC 的可信标题加权方式

不要在第一个 S10 实验里同时改变数据和 loss，否则无法判断收益来自合成数据还是标题权重。
正式训练只保留第 7.1 节四项。`S10-trusted-title-weighted` 的 channel 路由如下：

| 标题状态 | READoc profile | PMC profile | 解释 |
|---|---|---|---|
| 未通过标题审计 | `readoc_full` | `pmc_full` | 完整 response 全部权重 1 |
| 整篇/窗口标题全部可信 | `readoc_title_weighted` | `pmc_title_weighted` | heading 2、body 1，完整 output 不变 |

最终 mix JSONL 在顶层写例如：

```json
{"channel":"pmc_title_weighted","messages":[{"role":"user","content":"<image>document parsing."},{"role":"assistant","content":"# Title\n\nBody..."}],"images":["images/pmc123/page_0003.png"],"meta":{"source":"pmc_synthetic"}}
```

`channel` 是 ms-swift 保留字段，custom template 从 `inputs.channel` 读取并只接受
`readoc_full`、`readoc_title_weighted`、`pmc_full`、`pmc_title_weighted` 四个主线枚举；
未知值立即报错。两个 full channel 都产生 response 权重 1，两个 weighted channel 产生
heading/body=2/1。不要依赖 `meta` 做 loss 路由。

这一路径建议注册 `unlimited_ocr_mixed_loss` template。`S10-full` 仍用原生
`unlimited_ocr`；标题实验使用 mixed template 与 `uocr_weighted_token_mean`，数据行、顺序、
token 预算和其他超参保持一致。先用同一 batch 做 plugin full-parity smoke，证明全 1 权重与
native full 的 loss/gradient 对齐；它不是单独的完整训练实验。title-only 的实现说明保留作
研究备选，但当前数据标签和正文生成目标都不支持把它排进主线。

## 9. 交付文件

建议另一台服务器上的 Codex 实现以下独立脚本，不直接改原始数据：

```text
scripts/data/build_splits.py
scripts/data/build_readoc_ms_swift.py
scripts/data/build_pmc_ms_swift.py
scripts/data/audit_ms_swift_jsonl.py
scripts/data/measure_uocr_lengths.py
scripts/data/build_token_budget_mix.py
scripts/data/eval_title_spans.py
plugins/uocr_title_loss.py
tests/test_uocr_title_loss.py
```

最终产物：

```text
processed/
  split_manifest.jsonl
  readoc/{train,val,test,quarantine}.jsonl
  pmc/{strict_single,strict_window_2to4,quarantine}.jsonl
  lengths/{readoc,pmc}.jsonl
  buckets/{4k,8k,16k,24k,32k,overflow}/...
  mixes/{r0,s10,s20}/...
  reports/data_summary.json
  reports/reject_reasons.json
  reports/manual_audit.md
```

每个构建脚本必须：

- 固定 seed，排序输入，输出可复现。
- 使用临时文件完成后原子 rename。
- 不覆盖原始 PDF/GT/`middle.json`。
- 在输出中写 `builder_git_commit`、配置 hash、serializer version 和生成时间。
- 遇到单条坏数据写 reject manifest；结构性错误或数量异常则整体失败。

## 10. 实施顺序

1. 已完成 READoc PDF/GT inventory、标题层派生、PDF profile、review queue 和只改标题校验；下一步接正文最小清洗与确定性页面渲染，不生成猜测页边界的派生窗口。
2. 用真实 template 测每篇 READoc 的 `R/T/L`，按 source 与长度联合建立并冻结文档级 split manifest，同时冻结 test 内的 `long_eval`。
3. 生成 READoc 整篇 JSONL，并建立 R0 的 4K/8K/16K/24K/32K/overflow 桶；数据准备不因首轮显存上限丢弃长样本。
4. 在单张 H100 上跑 R0 4K/8K 的 5-10 step smoke，随后直接完成 16K/24K/32K forward/backward 峰值与吞吐探测，确定 `L_base`；正式 R0 从 `L_base` 开始。
5. 实现 PMC serializer、严格过滤和 quarantine；人工审计每个 reason/quality bucket。
6. 测 PMC 真实长度，构建与 R0 相同 `L_base` 的 S10 token-budget manifest。
7. 用相同 `L_base`、长度桶和其他超参比较 R0 与 S10；只有真实 READoc held-out 不回退才构建 S20。
8. 完整 CE 数据流程稳定后，先做 title-weighted 消融。
9. 只有确实需要“完整输出但纯标题 loss”时，才启用 external plugin 的 `title_only + active-weight mean` 路径；先证明 `input_ids`、labels 与 R-SWA mask 均未变化。
10. 若原生 ms-swift 的 `L_base < 32K`，再按长序列文档实施等价 R-SWA/FlexAttention 与 fused linear CE；通过 parity 后重测 24K/32K，不通过数据截断规避。

最重要的停止条件：**任何 target 不能明确对应它的全部输入页面时，不进入训练；任何合成数据提升只在合成 validation 上成立、却让真实 READoc 回退时，立即停止扩大合成比例。**
