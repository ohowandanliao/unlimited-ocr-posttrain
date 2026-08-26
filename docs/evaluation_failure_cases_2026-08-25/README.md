# 2026-08-25 evaluation 输出现象原始案例

这个目录的结构化原始记录来自
`/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/inference_eval_20260825_all129_parallel_v3/`
。仓库只保留最小可复算集合，不重复保存同一 response 的逐条 Markdown 展开。

## 文件说明

- `cases.jsonl`：24 条原始结构化记录；`response.text` 是完整模型回答，另含 prompt、页数、模型和信号。
- `gt__*.md`：8 个 evaluation groundtruth，供分析脚本和人工对照。
- `analysis_2026-08-26.md`：由 `cases.jsonl + gt__*.md` 生成的可读分析。
- 本目录不复制 PDF 和页图；源 PDF 和渲染页仍在原运行目录，路径见下表。

原来的 24 个 `case_*.md` 是 `jsonl_to_markdown.py` 对这 24 行的逐条展开；清理前已验证 24/24 内容
字节一致，因此不再提交。需要完整单条 Markdown 时写到仓库外临时目录：

```bash
python ms_swift_title_mask/scripts/jsonl_to_markdown.py \
  docs/evaluation_failure_cases_2026-08-25/cases.jsonl \
  --output-dir /tmp/uocr-evaluation-cases-20260825
```

## 案例索引

| 案例 | 输入 PDF | 页数 | 重点模型/现象 | `cases.jsonl` 行号 |
| --- | --- | ---: | --- | --- |
| 01 | `3-4长表跨三页以上.pdf` | 26 | full CE 重复 98%，三路对照 | 1-3 |
| 02 | `4-3扫描2.pdf` | 12 | full CE/weighted 重复 92%，base 重复 `(No text)` | 4-6 |
| 03 | `5-1双栏表格1.pdf` | 2 | weighted 重复 90% | 7-9 |
| 04 | `3-1跨页重复表头.pdf` | 25 | full CE/weighted 仅 576 字符且文本完全相同 | 10-12 |
| 05 | `4-1统计图3.pdf` | 31 | full CE 重复 56%，weighted 重复 73% | 13-15 |
| 06 | `4-2段落中带「图x-x」.pdf` | 38 | weighted 重复 73% | 16-18 |
| 07 | `4-3扫描1.pdf` | 5 | full CE/weighted 输出约为 GT 的 14 倍 | 19-21 |
| 08 | `2-5表内公式.pdf` | 1 | full CE 重复 71%，weighted 重复 79% | 22-24 |

每个案例保留 base、full CE、title-weighted 三路。任何原因说明都应视为待确认，不是已验证归因。

## 原始路径

- source PDF：`/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/source/`
- groundtruth：`/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth/`
- 完整 75 条结果：`/home/jovyan/hyx/uocr-ms-swift-title-mask/runs/20260823/inference_eval_20260825_all129_parallel_v3/`

这批材料只记录输出和输出协议现象，不代表正式准确率评测。评测在 75/387 条后停止，
未继续补跑剩余样本。
