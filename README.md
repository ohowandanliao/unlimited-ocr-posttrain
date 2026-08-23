# Unlimited-OCR ms-swift patch kit

这个分支把仓库收敛为 [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) 的
ms-swift 训练补丁、标题 GT 工具和可审计数据流水线。正式训练不再使用仓库自带 Trainer，而是把
[`ms_swift_title_mask/`](ms_swift_title_mask/) 放到外部 ms-swift 仓库根目录，通过 external plugin 执行。

当前工作分支是 `ms-swift-patch`。测试基线为 ms-swift
`1a1ba3ee86488af323ef9b64ca3d34edee90ab11`，完整参数和停止条件见
[`TRAINING_PLAN_ZH.md`](ms_swift_title_mask/TRAINING_PLAN_ZH.md)。

## 当前训练决策

- 两组正式实验：同一数据、顺序、长度和超参上的原生 full CE 与 title + EOS mask loss。
- 输入为一页或多页图片，assistant 始终输出完整连续 Markdown。title-mask 只改变 loss 权重，不改变
  input IDs、完整 labels、R-SWA mask、attention 或 logits 显存。
- 统一视觉策略为 `base/no-crop 1024`。当前数据同时包含单页与多页，这是同一个进程级视觉配置下允许的。
- 使用 decoder-backbone LoRA；guard 应精确命中 84 个 attention、dense MLP 和 shared-expert Linear，
  不训练 routed experts、视觉编码器、projector、embedding 或 lm_head，不做全参。
- 长度按真实 tokenizer 扫描。32K 是数据准备上限和 H100 探测目标，不代表已完成 32K backward 验收；
  正式实验使用原生 ms-swift 能稳定训练的最高共同长度，超限 target 不截断。
- attention 沿用 Unlimited-OCR 的 dense eager R-SWA。FlexAttention、fused linear CE、FA2/FA3、
  sequence parallel 和 `multi_gundam` 都不在当前补丁中。

## 四种输入模式

| 模式 | stock Unlimited-OCR / ms-swift | 本轮用途 |
|---|---|---|
| `single_gundam` | 官方支持；单页 dynamic crop | 当前不训；若需要高清单页，必须另开 crop 进程和实验 |
| `single_base` | 官方支持；单页 no-crop | 当前 PMC strict-single 实际采用的输入形态 |
| `multi_base` | 官方支持；多页 no-crop | 当前 READoc full 与 PMC full 主线 |
| `multi_gundam` | 不支持逐页 crop forward | 不实现、不作为训练方案 |

ms-swift 的 `CROP_MODE`、`IMAGE_SIZE` 和 `BASE_SIZE` 是进程级设置，不能在同一训练进程逐样本混用
gundam/base。本轮统一设置 `CROP_MODE=false, IMAGE_SIZE=1024, BASE_SIZE=1024`，因此 single-base 和
multi-base 可以自然混合。

## 数据快照

| 数据池 | 行/文档 | 页/图片引用 | 标题 |
|---|---:|---:|---:|
| READoc full | 1,552 | 15,290 | 21,400 |
| PMC full | 1,486 | 26,690 | 48,726 |
| PMC strict-single | 1,486 | 1,486 | 6,011 |
| final mix | 4,524 rows | 43,466 image refs | 76,137 |

final mix 为 train 4,073、validation 223、test 228。PMC full/single 复用源文档，因此共对应 3,038 份
唯一 PDF 和 41,980 个唯一页。三池做确定性 natural union，不混入重复的 `train_short.jsonl`，也不使用
MinerU 输入。

训练行使用 ms-swift messages schema：

```json
{
  "id": "readoc_arxiv_x__full",
  "channel": "title_reviewed",
  "messages": [
    {"role": "user", "content": "<image>Multi page merge."},
    {"role": "assistant", "content": "# Title\n\n## Section\n\nComplete body...\n"}
  ],
  "images": ["/absolute/work/pages/readoc/arxiv/x/page_0000.png"],
  "meta": {
    "source": "READoc-arxiv",
    "doc_id": "x",
    "sample_form": "full_document",
    "page_indices": [0],
    "n_pages": 1,
    "split": "train",
    "title_review_status": "SILVER_ACCEPTED",
    "title_target_sha256": "...",
    "source_pdf_sha256": "..."
  }
}
```

多图 prompt 固定为 `<image>Multi page merge.`，单图固定为 `<image>document parsing.`；output 是完整
Markdown，不是标题列表，也不插 `<PAGE>`。

## 可上传 ZIP

运行打包器会在仓库外生成一个可直接解压到 ms-swift 根目录的 ZIP：

```bash
python ms_swift_title_mask/scripts/build_bundle_zip.py \
  --output /absolute/path/unlimited-ocr-ms-swift-title-mask-20260823.zip
```

ZIP 包含 external plugin、单/双 H100 脚本、执行文档、final mix、source audit manifest 和全部 4,524 份
standalone Markdown GT；不包含 PDF、渲染页图、模型权重或 adapter。解压后先按包内
[`TRAINING_PLAN_ZH.md`](ms_swift_title_mask/TRAINING_PLAN_ZH.md) 重定位图片根目录，并用服务器自己的
READoc/PMC 源 PDF 完成 SHA、真实页数和页索引验证，再渲染 canonical 页图。路径不能只按文件名猜测。

`ms_swift_title_mask/data_assets/` 的大数据快照被 Git 忽略，由打包器显式纳入 ZIP；因此只 clone Git 分支
不会获得冻结的 763 MB payload。需要从原始数据重建时，按
[`ms_swift_from_source_2026-08-23.md`](docs/ms_swift_from_source_2026-08-23.md) 执行。

## 目录

```text
ms_swift_title_mask/          external plugin、训练脚本、数据契约和 ZIP builder
scripts/data/                 READoc/PMC 确定性标题规则、候选和 source audit builder
title_gt_pipeline/            可接 OpenAI-compatible API 的行级标题 review candidate 工具
docs/                         当前决策、长度分析和历史调研
```

[`title_gt_pipeline/markdown-title-repair/`](title_gt_pipeline/markdown-title-repair/) 只允许 API 返回标题行号和
目标层级，本地代码只修改 fenced code 外的 ATX `#`。它适合生成 review candidate，但不读取 PDF、不能恢复
缺失正文，也不能把结果自动升级为人工 gold。当前冻结的 READoc/PMC 数据使用确定性规则和 PDF/source SHA
证据，provenance 为 `SILVER_ACCEPTED`。

## 验证边界

本地测试覆盖 title-mask、DDP active-token mean、数据闸门、GT/PDF/JSONL 映射、路径迁移、长度分桶、
ZIP 和 fake API。依赖 torch、transformers、模型权重与 GPU 的 ms-swift 集成和 H100 backward 必须在目标
服务器执行 `run_preflight.sh` 与长度探测。

## License

MIT。Git 不提交模型权重、PDF、渲染页图、adapter、训练日志或冻结大数据 payload。
