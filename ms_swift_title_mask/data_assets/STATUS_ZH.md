# READoc / PMC 标题数据状态

> **历史 payload 状态，不能作为当前训练准入。** 2026-08-26 审计确认 PMC 标题 silver 没有覆盖正文、公式、
> 表格、图片资源或跨页 merge 质量，旧 full/single natural union 已停用。下面的数量和 SHA 契约仅用于复现旧
> 4,524-row 快照；当前 recipe 见 [`../TRAINING_PLAN_ZH.md`](../TRAINING_PLAN_ZH.md)。

快照日期：2026-08-23。这里随上传 ZIP 携带标题处理产物，但不携带 PDF、渲染页图或模型权重。
三个 payload 目录在 posttrain Git 中被忽略；打包脚本会显式检查并收入 ZIP，Git 提交本身不携带这些大文件。

## 结论

三个旧训练池保留标题 provenance，但该 provenance 不能代替正文准入。READoc 可作为 R0 候选；PMC 必须重做
`content_review_status`、清理污染并按 S10 比例重新采样。

| 目录 | 性质 | 当前决策 |
|---|---|---|
| READoc full | 1,552 篇、15,290 页、21,400 headings 的可追溯 silver Markdown | R0 候选；重新分桶和评测 |
| PMC full | 1,486 篇、26,690 页、48,726 headings | 不直接训练；页字符串拼接且正文未验收 |
| PMC strict-single | 1,486 paired rows、6,011 headings | 不与 full 全量重复；仅清洗/验收后进入 S10 候选 |

`HUMAN_ACCEPTED` 表示 reviewer 完成标题检查；`SILVER_ACCEPTED` 表示标题规则冻结且 provenance、source/target
SHA 和 PDF 证据完整。两者都只描述标题；完整 OCR 训练还必须单独验收正文内容。

## READoc 快照

来源：READoc v1，原始布局为 `ground_truth/{arxiv,github}/*.md + archives/{arxiv,github}.zip`；
具体服务器根目录不属于数据契约。

- 2,233 篇，22,290 页，派生 Markdown 位于 `ground_truth/{arxiv,github}/`。
- 922 处已应用改动：920 个 arXiv `Abstract` 从 H6 归一为 H2、1 个首页标题 promotion、1 个空 H1 恢复。
- `validation.json` 证明文档覆盖、源 SHA、PDF 配对和“只改标题”约束通过。
- `change_manifest.jsonl` 保存每处改动、PDF 页证据和前后 SHA。
- READoc full 仅纳入 1,552 篇 silver accepted；其余风险文档不进入本版训练。

服务器已有 READoc PDF 时，用 `document_manifest.jsonl` 的 `source + doc_id + pdf_sha256` 对齐，不能只靠
文件名猜测。生成训练 source JSONL 后，保留对应的 `HUMAN_ACCEPTED` 或 `SILVER_ACCEPTED` manifest provenance。

## PMC 快照

来源：PMC v26 synthetic，原始布局为 `documents/<doc_id>/{middle.json,document.pdf}`；具体服务器根目录
不属于数据契约。

- 1,486 篇，26,690 页，来源都是完整多页合成文档。
- 48,844 个候选全部映射到源 block：`normalize_title_text=8,338`、`change_level=31`、`demote=123`、`promote=179`、`retain=40,173`。
- PMC full 为 48,726 headings、180,298,275 chars；strict-single 为 6,011 headings、6,009,454 chars。
- PDF SHA 全部通过；当前 JSONL 已生成并可训练，保留 silver provenance，不再要求重新读取 `middle.json` 构建。

当前 JSONL 已包含每篇 PMC full 一行及 paired strict-single 一行；strict-single 是完整单页 Markdown，output 始终是完整 Markdown，不能把候选列表当模型 output。

## GT 与 PDF 映射

ZIP 携带全部 4,524 份 standalone Markdown GT：READoc full 1,552、PMC full 1,486、PMC single 1,486。
PDF 和渲染页图不进 ZIP。映射不是文件名猜测，而是以下闭环：

`source + doc_id -> manifest relative PDF path/member -> PDF SHA-256 + exact page count -> page_indices -> canonical image path -> JSONL assistant target SHA -> standalone GT`

打包闸门已经核验 JSONL assistant 与 standalone GT、source manifest PDF SHA/页数、页索引和 43,466 个图片路径；
跨 PMC full/single 去重后是 3,038 份 PDF、41,980 页。服务器仍必须读取自己的源 PDF 再做一次实体验证，
因为本地 manifest 校验不能证明服务器上的文件确实存在。

## `train_short.jsonl` 不是单页集

本地核验的 `/Users/guofengjiao/Downloads/train_short.jsonl` 共 1,364 条，全部来自 READoc 且 doc_id 全部
与上述 READoc 清单重合。图片数为 1-10 页，只有 2 条单图，中位数 5 页。它只是 READoc 的短长度子集，
不随正式三池重复打包，也不能混入 READoc full。

## 完整性校验

ZIP 内 `DATA_SHA256SUMS` 覆盖全部 payload 普通文件。解压后在
`ms_swift_title_mask/data_assets` 中运行：

```bash
shasum -a 256 -c DATA_SHA256SUMS
```

缺少 payload、校验失败、服务器图片路径未渲染/对齐或 provenance 不完整时，训练 preflight 必须停止。
本地 JSONL 中的图片路径仍需在服务器完成渲染与路径对齐；本状态文件不宣称图片已存在。
`build_report.json` 中的 `input_root/image_root` 仅记录快照生成时的历史位置，运行时不读取这些字段；目标服务器
必须用 `relocate_image_paths.py` 生成自己的绝对图片路径。
