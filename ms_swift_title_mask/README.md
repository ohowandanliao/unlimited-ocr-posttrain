# Unlimited-OCR reviewed-title training bundle

这是一个放进 **ms-swift 仓库**后使用的 external plugin，不是独立训练框架。完整执行口径见
[`TRAINING_PLAN_ZH.md`](TRAINING_PLAN_ZH.md)。

## 当前结论

- 训练准入包括逐篇人工复核通过的 `HUMAN_ACCEPTED`，以及规则冻结、完整可追溯的
  `SILVER_ACCEPTED`；两者 provenance 必须在 manifest 中明确区分。
- 当前三池为 READoc full（1,552 篇/15,290 页/21,400 headings）、PMC full（1,486 篇/26,690 页/48,726 headings），
  以及同批 PMC 的 paired strict-single（1,486 行/6,011 headings）。
- 三池做 natural union：每个 READoc full 一行、每个 PMC full 一行、同一 PMC 再有一条 strict-single；PMC full/single 使用相同 split。
- 正式只比较两组：同一数据顺序的 ms-swift 原生 full CE 和 title+EOS mask loss。`train_short` 不混入正式数据。
- 只用 LoRA，不做全参；LoRA 精确落在 84 个 decoder-backbone Linear 上。
- 单卡/双卡都用 H100、R-SWA、base/no-crop 1024 和 eager attention；同一进程内自然混合单页与多页。

## 放入 ms-swift

ZIP 解压后应得到：

```text
/path/to/ms-swift/
  swift/
  ms_swift_title_mask/
```

只在现有 ms-swift 环境中补一个小依赖，不要重新安装 torch：

```bash
cd /path/to/ms-swift
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r ms_swift_title_mask/requirements.txt
```

ZIP 已直接带 final three-pool mix、4,524 份 standalone GT 和 READoc/PMC source manifest，但不带 PDF 或
渲染页图。使用前必须先执行 manifest dry-run，再对服务器上的 3,038 份源 PDF 做 SHA/实际页数验证，验证全过后
才渲染图片；随后按长度生成 length-fit 目录，再跑 preflight 和真实、预先选出的 <=32K 样本 1-step smoke。

四种输入模式中，本轮使用 `single_base + multi_base`；`single_gundam` 需另开 crop 进程，`multi_gundam`
不受 stock model 支持。完整表格与命令见 `TRAINING_PLAN_ZH.md`。若只有 Git 分支和原始 READoc/PMC 数据，
从源数据重建的完整顺序见仓库根目录 `docs/ms_swift_from_source_2026-08-23.md`；该文件不属于独立 ZIP 的
直接训练必需项。
