# 评测 Prompt 矩阵标准（2026-09-01）

> 本文是所有 Unlimited-OCR posttrain 模型 129-PDF 评测的 prompt 唯一标准。
> `HANDOFF_2026-09-01.md`、各评测文档和新评测都必须引用并遵守本文；与本文冲突的旧口径以本文为准。

## 1. 事实基础：各训练数据的 prompt 契约（2026-09-01 逐行查证）

| 训练数据 | 行数 | 训练 prompt |
|---|---|---|
| `data/readoc-view-16k/train.jsonl`（readoc-1706 Full-CE / Title-weighted） | 1706 | 全部 `<image>Multi page merge.` |
| `data/readoc-heading-prior-16k/train.jsonl`（readoc-heading-prior-16k Full-CE / Title-weighted） | 1697 | 全部 `<image>Multi page merge.` + 标题段 |
| `$WXZ_READOC_ROOT/readoc_uocr/r0/views/train_le_16k_with_title.jsonl`（夏桢无 prior Title-mask） | 多页 view | 全部 `<image>Multi page merge.` |
| `$WXZ_READOC_ROOT/readoc_uocr_heading_prior/views/train_le_16k_with_title.jsonl`（夏桢带 prior Title-mask） | 多页 view | 全部 `<image>Multi page merge.` + 标题段 |

要点：

- 所有训练数据都是多页 view（最少 2 图/行），**训练中从未出现过单页样本**，也从未出现过
  `document parsing.` 作为目标 prompt。单页评测使用 `document parsing.` 是既定评测协议
  （Base 原生 prompt，与 2026-08-27 评测一致），不是训练契约。
- `merge` 是训练时定义的新 prompt，**只用于训练模型的多页评测**，未训练 Base 永不使用。
- 夏桢带 prior 数据的标题段外壳与 `prompts.all_pages.jsonl` 的外壳**逐字一致**
  （仅 `### MinerU headings` 后多一个空行的差别，可忽略）；因此同一份 manifest 可同时
  服务于本轮两个 `checkpoint-212` 和夏桢带 prior 模型。
- prior 内容来源差异：夏桢训练时的 prior 抽自夏桢自己的 MinerU 输出；20260901 manifest
  的 prior 统一抽自 `$MINERU_OUTPUT_ROOT/markdown`。
  重测夏桢带 prior 模型时复用 20260901 manifest，即统一以 MinerU 344 为 prior 来源；
  该选择属于“prior 是外部输入提示”的协议口径，需与训练外壳一致性区分记录。

## 2. 标准矩阵

评测 prompt 由“模型类别 + 页数 + 该文件是否有 MinerU 标题”决定：

| 模型类别 | 单页 | 多页 | 标题段 |
|---|---|---|---|
| 正式主 Base（未训练） | `<image>document parsing.` | `<image>Multi page parsing.` | 永不添加 |
| 无 prior 训练模型（readoc-1706 两种、夏桢无 prior） | `<image>document parsing.` | `<image>Multi page merge.` | 永不添加 |
| 带 prior 训练模型，文件有 MinerU 标题 | `document parsing.` + 标题段 | `Multi page merge.` + 标题段 | 添加 |
| 带 prior 训练模型，文件无 MinerU 标题 | `<image>document parsing.` | `<image>Multi page merge.` | 不添加 |

标题段固定外壳（标题行插在 `### MinerU headings` 之后）：

~~~text
The following headings are extracted from the MinerU parsing result of the same document.
Use them only as structural hints.
Verify and correct the heading text, heading levels, ordering, and document structure according to the document images.
Do not blindly copy the MinerU headings if they conflict with the document images.

### MinerU headings
...

### Final corrected Markdown
~~~

统一生成与图像参数（`serve_unlimited_ocr.py` + `probe_evaluation_pdfs_parallel.py` 入口）：

- `image_size=1024`，`crop_mode=False`；144 DPI 渲染页图
- `max_length=20480`；`no_repeat_ngram_size=35`；单页 `ngram_window=128`，多页 `ngram_window=1024`；`temperature=0`

**max_length 20480 的依据（2026-09-01 决定，替代历史 32768）**：

- 129 个评测 GT 长度中位数 `1065` 字符（≈500 token），p95 `19836`，最长 `33954`
  字符；按实际输出校准 token/char ≈ 0.47，最长 GT 的完整合法输出 ≈ 16K token。
- 2026-08-27 轮 387 行从未触及 32768：合法输出最长 `37702` 字符 ≈ 17.7K token；
  `20480` 对全部合法输出留有约 15% 余量，历史 + 本轮数据中 `0` 行合法输出被截。
- cap 只截断 heading-prior 模型的退化复读尾巴（实测 21-29K token、1.5-6× GT 超量），
  不影响合法内容；cap 命中行照常进入退化诊断，评分不做任何后处理截断。
- 该变更自 2026-09-01 起对随后所有 129-PDF 推理（含夏桢两模型重测）统一生效；
  主 Base 历史结果（32768 协议）不受影响，cap 差异仅存在于退化长尾。

## 3. manifest 使用规则

- 带 prior 模型必须使用逐文件 manifest：`evaluation/input_manifests/heading_prior_20260901/prompts.all_pages.jsonl`
  （129 行 = 129 个 PDF；109 单页 + 20 多页；91 有标题 + 38 无标题）。runner 不得按页数
  自行回退默认 prompt；评测后逐条核对 response 中保存的 prompt 与 manifest 一致。
- 无 prior 模型**不传** `--prompt-manifest`：`prompt_for()` 的默认值恰好等于本矩阵
  （单页 parsing；trained 多页 merge；base 多页 Multi page parsing）。
- Base 请求不加标题段；主 Base 多页是 `Multi page parsing.`，不是 merge。
- 带 prior 模型的逐文件 prompt 一旦发现与 manifest 不一致（如旧批次的“单页漏标题段”），
  该批结果作废，不得进入正式评分。

## 4. 2026-09-01 审计：各轮评测实际使用的 prompt

| 评测 | 实际 prompt（responses 逐条核对） | 判定 |
|---|---|---|
| 主 Base：`inference_eval_20260825_all129_parallel_v3` | 单页 109× `document parsing.`；多页 20× `Multi page parsing.` | ✅ 符合标准；`53.3645 / 0.8271` 继续作为正式主基线 |
| `readoc-view-16k_all129`（8-27，无 prior 两模型，387/387） | base：parsing / `Multi page parsing.`；trained：单页 parsing、多页 merge（各 109/20） | ✅ 符合标准；`44.1397 / 0.8102`、`43.3568 / 0.8017` 保持有效 |
| `wxz-readoc-badcase-20260825`（夏桢两模型，产物已删除） | `eval_badcase.py` 固定 prompt：单页 `document parsing.`、多页 `Multi page merge.`；**标题段从未加入**；对照 Base 是 merge 历史 Base（`51.7663 / 0.7442`）；入口参数（`base_size=1024, image_size=640, crop_mode=True`，多页 `ngram_window=128`）与统一服务不同 | ⚠️ 废弃历史审计：带 prior 模型未按训练契约输入标题段，`33.3329 / 0.6801` 与无 prior 的 `25.9391 / 0.6635` 均不代表正式方案；统一协议重测已完成 |
| `inference_eval_20260901_heading_prior_fullce_weighted_all129_all_pages`（本轮 258 条） | manifest 驱动 | ✅ 完成：258/258 验收通过，评分见 `evaluation_readoc_heading_prior_2026-09-01.md` 第 8 节 |

## 4.1 2026-09-01 审计补充结论

- 本轮 258 条跑完后又发生一次协议变更：`max_length` 32768 → 20480（依据见第 2 节），
  已完成的 ≤cap 行保留（逐字节等价），5 条超限退化行在 20480 下重做；最终 258/258 验收通过。
- 夏桢两模型重测已按第 5 节执行完毕（2026-09-01 晚启动、09-02 完成）：带 prior 复用本 manifest（验收时四类
  数量 73/36/18/2 精确匹配），无 prior 未传 manifest（默认 prompt = 本矩阵），
  `unlimited-ocr-full-ce` 槽位 = 夏桢带 prior 模型、`unlimited-ocr-title-weighted` 槽位 =
  夏桢无 prior 模型，映射已写入 runner note。
- 结果：无 prior Overall `0.5122`、带 prior `0.4500`——带 prior 反而低 `6.22 pp`，
  证实旧 badcase 的 prior 正收益是协议缺陷假象。

## 5. 夏桢两模型按标准重测记录（已完成）

1. **服务**：使用 `serve_unlimited_ocr.py`，两个 adapter 槽位使用夏桢的根目录 adapter
   （已确认 `base_model_name_or_path=$MODEL_PATH`，peft 可直接加载；
   注意其 checkpoint 无 `checkpoint-*` 子目录，最终 adapter 在输出根目录）：
   - 带 prior：`$WXZ_OUTPUT_ROOT/readoc_heading_prior_title_mask_16k`
   - 无 prior：`$WXZ_OUTPUT_ROOT/readoc_title_mask_16k`
   runner 只认 `unlimited-ocr-full-ce` / `unlimited-ocr-title-weighted` 两个 trained model 名，
   槽位映射已写入 runner note，避免与 hyx 同名模型混淆。
2. **带 prior 模型**：复用 `prompts.all_pages.jsonl`（外壳一致）。
3. **无 prior 模型**：未传 manifest，runner 默认 prompt 即标准矩阵。
4. **对照**：复用主 Base `53.3645 / 0.8271`，未重跑 Base。
5. **结果**：评分目录已按 scheme_id 分列；夏桢两行已更新，历史 badcase 分数仅保留为废弃审计记录。

- 2026-09-02 case 级归因完成：hyx 两套的掉分全部由推理时标题段触发（无壳文件与 fc827/Base 完全一致），
  先验内容质量（对 GT 标题 recall 0.860 / precision 0.780）不是瓶颈；详见 `evaluation_readoc_heading_prior_2026-09-01.md` 第 9 节。
