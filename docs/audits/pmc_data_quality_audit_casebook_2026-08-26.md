# PMC 原始数据质量审计典型案例

更新时间：2026-08-26（Asia/Shanghai）

本账本是给数据开发对账的典型案例，不是全量清单。配套表格见
[`pmc_data_quality_audit_casebook_2026-08-26.xlsx`](pmc_data_quality_audit_casebook_2026-08-26.xlsx)。
所有 JSONL 行号只对下列已核验 SHA-256 有效；源数据和训练 JSONL 均留在仓库外。

## 固定输入

| split | JSONL 路径变量 | 行数 | SHA-256 |
|---|---|---:|---|
| train | `$LEGACY_MIX_ROOT/train.jsonl` | 4073 | `38491b6e5ebb5fa416363e25e536a09a8522d632089c7dc8c9fe9cebc6e4a74e` |
| validation | `$LEGACY_MIX_ROOT/validation.jsonl` | 223 | `e8d738de4a90dedebe9dff53b808ee56ddd3b02fc65f5213cd3ffb02e4c2ccde` |
| test | `$LEGACY_MIX_ROOT/test.jsonl` | 228 | `26957bbb3090829c519c1fc4081cb12d42d16149b74b7faf4c2cece9cb39b201` |

middle 根目录：`$PMC_SOURCE_ROOT/documents/<doc_id>/middle.json`。
JSON Pointer 下标为零基；`/messages/1/content` 是 assistant target。以下 raw→JSONL 传播证据以 JSONL
`meta.doc_id`、`meta.title_source_sha256` 与 middle SHA 对账，不能据此证明正文已人工验收。

证据分两级：`直接传播` 表示问题片段从 middle 原样进入 JSONL；`源完整性风险` 表示源引用/页结构有缺口，
但不夸大为整篇 target 都错误。文件不存在性按上述固定快照和解析后的路径核验。

## XLSX 结构

`Cases` 是给人看的对账主表，只保留 11 列：`问题编号、问题类型、严重度、doc_id、JSONL 定位、
row_id、JSONL 片段、middle 定位、middle 片段、问题说明、修复/验收`。`Sampling` 记录定界抽样
频次；固定 JSONL 路径变量、SHA-256、middle 根目录及各 case 的 middle SHA-256 放在 `Snapshot`。
完整证据链和保守限定以本 Markdown 为准。

## 定界抽样复扫

本次没有遍历全部 middle。先只列出 1,486 个含 `middle.json` 的 doc_id，排序后取
`round(i * (1486 - 1) / 119)`，其中 `i=0..119`，得到固定的 120 篇等距样本；只打开这 120 个
`middle.json`。频次分母都是 `120`，只能描述该样本，不外推为全库精确发生率。`0/120` 也不代表
全库不存在。

| 检查项 | 样本文档频次 | 样本内命中量 | 结论 |
|---|---:|---:|---|
| 长 `text` block 跨页完全重复候选 | 11/120（9.2%） | 69 组、152 blocks、62 个唯一文档页 | 有命中；混有可能合法的摘要/评审材料重复，只将 `PMC-DQ-08` 作为高置信典型 case |
| literal `<!--` 或 `-->` 注释残片 | 4/120（3.3%） | 153 spans、126 blocks、24 个唯一文档页 | 有命中；见 `PMC-DQ-09` |
| replacement/control/mojibake | 0/120（0.0%） | 0 | 本样本未发现高置信乱码或非法控制字符 |
| renderer 不支持但含可见文字的顶层 block type | 0/120（0.0%） | 0 | 本样本未发现 |
| 可渲染全文为空或少于 100 字符 | 0/120（0.0%） | 0 | 本样本未发现 |

重复候选的判定是：只看顶层 `para_blocks[*].type=text`；每行直接拼接
`spans[*].content`，行与行之间插入一个空格，再做 `strip().lower()`；结果长度至少 120 字符，且同一
文档内完全相同并位于至少两个不同 `page_idx`。注释残片只匹配 literal `<!--`/`-->`，不把正常
Markdown autolink、普通 HTML table 或已有的 image-only table 重复计入。

## Cases

### PMC-DQ-01：full/single 同 PMCID 重复

- **JSONL 定位**：`train.jsonl:1067`，row `pmc13425563__full`，`sample_form=full_document`，`page_indices=[0..14]`；对照 `train.jsonl:3225`，row `pmc13425563__p0000`，`sample_form=strict_single_page`，`page_indices=[0]`。两行的 target pointer 都是 `/messages/1/content`；`jsonl_evidence_snippet`：`Research ArticleApplications of Polymer, Composite, and Coating Materials`。
- **middle 定位**：`$PMC_SOURCE_ROOT/documents/pmc13425563/middle.json`，SHA-256 `37eff540f43facda700b9d5c0308414f0e6915d1e5e3afd8de7a3921c0a13722`，page `0`，pointer `/pdf_info/0/para_blocks/0/lines/0/spans/0/content`，`middle_json_snippet`：`Research ArticleApplications of Polymer, Composite, and Coating Materials`。
- **raw→JSONL**：两行 `meta.doc_id=pmc13425563`、`meta.title_source_sha256` 等于上述 middle SHA；single target 共 4,866 字符，是 full target 的精确前缀，并在 full 中出现一次。
- **结论**：同一 PMCID 同时进入 full 与 single pool，属于数据配方重复。
- **修复/验收**：每个 PMCID、split、epoch 只保留一种 sample_form；按 `doc_id` 去重审计应为零冲突。
- **限定**：full 与 single 的目标范围不同；本 case 证明重复采样，不证明两个 target 字符串完全相同。

### PMC-DQ-02：坏 LaTeX 直接进入 train target

- **JSONL 定位**：`train.jsonl:2`，row `pmc13215880__full`，`sample_form=full_document`，`page_indices=[0..15]`，pointer `/messages/1/content`，`jsonl_evidence_snippet`：`v( R) = \sqr t{\fra c{GM( \;<R) }{R}}`。
- **middle 定位**：`$PMC_SOURCE_ROOT/documents/pmc13215880/middle.json`，SHA-256 `dd5a133ea62c21f8c48f671cb3267a48e2342ca0f58128b33e76eb8bbad42821`，page `6`，pointer `/pdf_info/6/para_blocks/9/lines/0/spans/0/content`，`middle_json_snippet`：`v( R) = \sqr t{\fra c{GM( \;<R) }{R}}`。
- **raw→JSONL**：JSONL `meta.doc_id=pmc13215880` 且 `meta.title_source_sha256` 等于 middle SHA；同一坏公式片段从 canonical `para_blocks` 原样进入 train target。
- **结论**：公式控制词被空格拆开，raw middle 的坏 LaTeX 传播到 assistant target。
- **修复/验收**：在序列化前拒绝已知坏模式或回源重建；验收应对 raw 与 target 同页公式做坏模式计数，结果为零。
- **限定**：本 case 只证明格式坏，不判断原公式数学语义能否唯一恢复。

### PMC-DQ-03：元数据粘连

- **JSONL 定位**：`train.jsonl:3363`，row `pmc13424892__full`，`sample_form=full_document`，`page_indices=[0..10]`，pointer `/messages/1/content`，`jsonl_evidence_snippet`：`Systematic ReviewAcademicSubjects/MED00010AcademicSubjects/MED00160AcademicSubjects/MED00240`。
- **middle 定位**：`$PMC_SOURCE_ROOT/documents/pmc13424892/middle.json`，SHA-256 `1e1e94cd0bf9ce413a82ce705f28432a73e4675c35f80e79bc1b98537a52fbb6`，page `0`，pointer `/pdf_info/0/para_blocks/0/lines/0/spans/0/content`，`middle_json_snippet`：同样片段。
- **raw→JSONL**：`meta.doc_id=pmc13424892`、`meta.title_source_sha256` 匹配 middle SHA，且 target 开头原样复制该 block。
- **结论**：分类/元数据字段与正文标题无分隔地粘连，已进入 full target。
- **修复/验收**：元数据字段须单独结构化或按规则加边界；验收不得出现 `AcademicSubjects/` 紧邻标题/正文的粘连模式。
- **限定**：不能仅凭字符串判断这些字段是否来自版面同一行；这里只报告序列化结果及其传播。

### PMC-DQ-04：空页

- **JSONL 定位**：`train.jsonl:978`，row `pmc13252203__full`，`sample_form=full_document`，pointer `/meta/page_indices` 与 `/images/17`–`/images/19`；`jsonl_evidence_snippet`：`"page_indices":[...,16,17,18,19,20,21], "n_pages":22`，并引用 `page_0017.png`、`page_0018.png`、`page_0019.png`。`/messages/1/content` 不含页边界标记。
- **middle 定位**：`$PMC_SOURCE_ROOT/documents/pmc13252203/middle.json`，SHA-256 `3a7868992e005664ec879c54102a8a3b3cd4c68e630ecd180e15337c6f2d494f`；page `17`、`18`、`19` 的 pointers `/pdf_info/17/para_blocks`、`/pdf_info/18/para_blocks`、`/pdf_info/19/para_blocks` 均为 `[]`，`preproc_blocks` 也均为 `[]`；`middle_json_snippet`：`page_idx=17/18/19: para_blocks=[]`。
- **raw→JSONL**：JSONL `meta.doc_id=pmc13252203`、`page_indices` 覆盖 17–19，`meta.title_source_sha256` 匹配 middle SHA；full renderer 将页序范围纳入 target，但不携带空 block 证据。
- **结论**：源 middle 有连续三张空 block 页，属于 page-level 完整性风险。
- **修复/验收**：空页须显式 quarantine/保留标记；验收应逐页比较 source page count 与 target page provenance，不能静默丢失。
- **限定**：空 `para_blocks` 不等于 PDF 视觉上绝对空白，需结合 PDF 页面图像确认。

### PMC-DQ-05：缺失图片路径直接进入 train target

- **JSONL 定位**：`train.jsonl:5`，row `pmc13435129__full`，`sample_form=full_document`，`page_indices=[0..37]`，pointer `/messages/1/content`，`jsonl_evidence_snippet`：`![](nihms-2195067-f0001.jpg)`。
- **middle 定位**：`$PMC_SOURCE_ROOT/documents/pmc13435129/middle.json`，SHA-256 `da3f5f4d43a96b45982e2f4a0ac8fd43641051d9a9ea96e902a52fc66cb692b2`，page `33`，pointer `/pdf_info/33/para_blocks/9/blocks/0/lines/0/spans/0/image_path`，`middle_json_snippet`：`nihms-2195067-f0001.jpg`。
- **raw→JSONL**：`meta.doc_id=pmc13435129` 与 middle SHA 匹配；renderer 将 raw `image_path` 序列化成 Markdown 图片链接。绝对候选路径 `$PMC_SOURCE_ROOT/documents/pmc13435129/nihms-2195067-f0001.jpg` 不存在。
- **结论**：不可读取、也不能从页面像素推断其文件名的资源路径，直接成为 assistant target。
- **修复/验收**：复制/重建资源或将引用转为明确缺失标记；验收对每个 `image_path` 做文件存在性和可读性检查，缺一即 quarantine。
- **限定**：该 case 证明这一处标签不可完整监督，不等于整篇文本全部无效。

### PMC-DQ-06：image-only table

- **JSONL 定位**：`train.jsonl:3288`，row `pmc10933458__full`，`sample_form=full_document`，`page_indices=[0..36]`，pointer `/messages/1/content`，`jsonl_evidence_snippet`：`<table-wrap id="Tab1"><figcap><p>Table 1 Clusters where representation is affected by the genotype are highlighted.</p></figcap><img src="44319_2024_77_Tab1_HTML.jpg">`。
- **middle 定位**：`$PMC_SOURCE_ROOT/documents/pmc10933458/middle.json`，SHA-256 `67261e77ed16e3715e22848b1a5fa9030b3ba1446deff83378189c9422b63963`，第 21 页（`page_idx=20`），pointer `/pdf_info/20/para_blocks/2/blocks/0/lines/0/spans/0/html`，`middle_json_snippet=<table-wrap id="Tab1">...<img src="44319_2024_77_Tab1_HTML.jpg">...</table-wrap>`；相邻 `image_path=assets/table_p20_o2_table.png`，无 HTML `<table>`。
- **raw→JSONL**：`meta.doc_id=pmc10933458`、`meta.title_source_sha256` 匹配；该 table-wrap/image 结构进入 full target，但不是可解析 HTML 表格。
- **结论**：表格内容依赖图片资源，不能按结构化表格训练/评估。
- **修复/验收**：保留 image-only 类型并单独处理 OCR/资源；验收要求结构化表格必须有 HTML table，或显式标记 image-only 且资源可读。
- **限定**：没有视觉渲染复核时，不判断图片内表格是否可完整 OCR。

### PMC-DQ-07：title accepted 但 content status 缺失

- **JSONL 定位**：`train.jsonl:3363`，row `pmc13424892__full`，`sample_form=full_document`，`page_indices=[0..10]`，pointer `/meta/title_review_status` 值为 `SILVER_ACCEPTED`；`/meta/content_review_status` 不存在。target pointer `/messages/1/content`，`jsonl_evidence_snippet`：`Systematic ReviewAcademicSubjects/MED00010...`。
- **middle 定位**：`$PMC_SOURCE_ROOT/documents/pmc13424892/middle.json`，SHA-256 `1e1e94cd0bf9ce413a82ce705f28432a73e4675c35f80e79bc1b98537a52fbb6`，page `0`，pointer `/pdf_info/0/para_blocks/0/lines/0/spans/0/content`，`middle_json_snippet`：`Systematic ReviewAcademicSubjects/MED00010AcademicSubjects/MED00160AcademicSubjects/MED00240`。
- **raw→JSONL**：JSONL meta 记录 title ruleset/source SHA，但没有 content acceptance 字段；同一 row 仍携带包含粘连元数据的完整正文 target。
- **结论**：`SILVER_ACCEPTED` 只能证明标题规则状态，不能证明正文/表格/公式/资源已验收。
- **修复/验收**：增加必填 `content_review_status`、`content_review_reason`、`content_source_sha256`；validator 对缺失或非 accepted 状态拒绝进入正文训练池。
- **限定**：字段缺失本身不证明正文必坏，只证明准入状态不可追溯。

### PMC-DQ-08：相邻页完整正文 block 重复

- **抽样频次**：长 `text` block 跨页完全重复候选命中 11/120 篇（9.2%），共 69 个重复组、152 个 block occurrence、62 个唯一文档页。这个数字是候选频次，不是 11 篇都已人工判错。
- **JSONL 定位**：`train.jsonl:2230`，row `pmc13424070__full`，`sample_form=full_document`，`page_indices=[0..48]`，pointer `/messages/1/content`；同一段在 target 中连续出现两次，`jsonl_evidence_snippet`：`In addition to chemical stability, the mechanical stability of the electrode structure is also crucial. ... thereby accelerating performance degradation.\n\nIn addition to chemical stability, the mechanical stability ...`。
- **middle 定位**：`$PMC_SOURCE_ROOT/documents/pmc13424070/middle.json`，SHA-256 `a2f7110981d5b4c6b75ab560f49f2f5553c383d56284ead1d7f55fe5e04bda05`；前一处 page `36`，pointer `/pdf_info/36/para_blocks/10`；后一处 page `37`，pointer `/pdf_info/37/para_blocks/0`；两个 block 拼接后的正文完全相同，`middle_json_snippet`：`In addition to chemical stability, the mechanical stability of the electrode structure is also crucial. Traditional adhesive-based electrode preparation methods often lead to catalyst detachment ... thereby accelerating performance degradation.`。
- **raw→JSONL**：JSONL `meta.doc_id=pmc13424070` 且 source SHA 匹配；上述首句在 full target 中精确出现两次，两次之间只有空行。
- **结论**：这个典型 case 是相邻页边界处完整正文 block 重叠，重复内容直接成为两份监督标签。
- **修复/验收**：对相邻页边界的长 block 做规范化重复检测并结合 PDF/页图确认；确认重叠后只保留一次。该 case 修复后，目标首句计数应从 2 变为 1。
- **限定**：抽样中的其他候选可能是摘要、评审材料、版权或机构信息的合法重复，不能只按字符串全局盲删。

### PMC-DQ-09：XML/HTML 注释残片进入 target

- **抽样频次**：literal `<!--` 或 `-->` 命中 4/120 篇（3.3%），共 153 spans、126 blocks、24 个唯一文档页。
- **JSONL 定位**：`validation.jsonl:24`，row `pmc13426939__full`，`sample_form=full_document`，`page_indices=[0..35]`，pointer `/messages/1/content`，`jsonl_evidence_snippet`：`-->PONE-D-25-58370-->-->Microbial communities in the rhizosphere of three Mentha species`。
- **middle 定位**：`$PMC_SOURCE_ROOT/documents/pmc13426939/middle.json`，SHA-256 `7fa1a6876a2122152453d79fa5bec36e49070e9ccfdf59c7ebcbf926aa99194f`，page `25`，pointer `/pdf_info/25/para_blocks/3/lines/0/spans/0/content`，`middle_json_snippet`：`-->PONE-D-25-58370-->-->Microbial communities in`。
- **raw→JSONL**：JSONL `meta.doc_id=pmc13426939` 且 source SHA 匹配；该开头残片从 span 原样进入 validation full target。相同文档的 single row `validation.jsonl:183`（`pmc13426939__p0035`）选择 page 35，没有选择问题所在的 page 25，因此该 single target 不含此片段。
- **结论**：上游 XML/HTML 注释边界被拆坏，literal `-->` 和投稿系统标识成为 OCR 监督文本。
- **修复/验收**：回到 XML/HTML 解析层移除完整注释节点及孤立边界残片；进入训练池前，普通正文中的 literal `<!--`/`-->` 计数应为零。
- **限定**：频次规则刻意不匹配普通 HTML tag、Markdown autolink 或 table HTML；本 case 不代表所有 HTML 都有问题。
