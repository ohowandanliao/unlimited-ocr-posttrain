# Unlimited-OCR 推理服务

## 当前契约

同一个服务进程只加载一份 base 模型，并在同一个 `PeftModel` 中加载两个 LoRA
adapter。请求通过 `model` 选择路由，三个模型共用同一张物理 GPU：

| `model` | 权重 |
| --- | --- |
| `unlimited-ocr-base` | 未训练 base，使用 `disable_adapter()` |
| `unlimited-ocr-full-ce` | 由 `FULL_CE_ADAPTER_PATH` 显式指定 |
| `unlimited-ocr-title-weighted` | 由 `TITLE_WEIGHTED_ADAPTER_PATH` 显式指定 |

两个 adapter 只有 LoRA 增量，额外显存很小；主要显存由共享的 base 模型和 KV cache
占用。服务请求在进程内串行化，避免切换 adapter 时并发修改生成状态。

## 推理代码

推理服务代码放在 `ms_swift_title_mask/scripts/serve_unlimited_ocr.py`，启动入口是
`ms_swift_title_mask/scripts/run_inference_service.sh`。

- base 路由在同一份 base 权重上使用 `disable_adapter()`。
- full CE 和 title-weighted 路由都通过 `PeftInferenceAdapter` 调用 Unlimited-OCR 原生 `infer()` / `infer_multi()`；receiver 是带当前 adapter 的 `PeftModel`，所以最终 `generate()` 会经过对应 LoRA。
- 单页使用 `<image>document parsing.`。
- base 多页使用 `<image>Multi page parsing.`。
- full CE/title-weighted 多页使用 `<image>Multi page merge.`。
- 没有加入 `page`，也没有在推理端改写 assistant 输出。

## 评测 Prompt 选择矩阵

评测 prompt 由页数、模型是否为正式主 Base、以及该文件是否有 MinerU 标题提示共同决定。标题
提示只给当前两套 Title Prior trained model，绝不添加到 Base：

| 评测场景 | 单页 prompt | 多页 prompt | 标题提示 |
|---|---|---|---|
| 正式主 Base | `<image>document parsing.` | `<image>Multi page parsing.` | 不添加 |
| READoc Full-CE / Title-weighted（无 Title Prior） | `<image>document parsing.` | `<image>Multi page merge.` | 不添加 |
| Title Prior trained model，文件有 MinerU 标题 | 基础 prompt + 标题段 | 基础 prompt + 标题段 | 添加 |
| Title Prior trained model，文件无 MinerU 标题 | `<image>document parsing.` | `<image>Multi page merge.` | 不添加 |
| 夏桢历史入口 | 依历史入口 | `Multi page merge.` | 仅历史 diff，不用于正式主 Base 结论 |

本次 129-PDF Title Prior manifest 的实际组合为：单页有标题 73、单页无标题 36、多页有标题
18、多页无标题 2（每套 trained model 各一份）。有标题时追加的固定外壳为：

~~~text
The following headings are extracted from the MinerU parsing result of the same document.
Use them only as structural hints.
Verify and correct the heading text, heading levels, ordering, and document structure according to the document images.
Do not blindly copy the MinerU headings if they conflict with the document images.

### MinerU headings
...

### Final corrected Markdown
~~~

带 MinerU Title Prior 的评测必须使用逐文件 `prompts.all_pages.jsonl` manifest；不能只根据
单页/多页选择默认 prompt。唯一标准见 `evaluation_prompt_matrix_2026-09-01.md`。

## 历史 Base 口径

两条 Base 结果使用的是同一批 129 个 evaluation PDF，差异不应理解为“Base 权重不同”或
Overall 算法不同，主要是评测入口和推理协议不同。主 Base 必须以未训练模型的
`<image>Multi page parsing.` 结果为准；`<image>Multi page merge.` 是训练数据和训练后
模型使用的 prompt，不能用于未训练 Base。

| 项目 | 主 Base（parsing，正式主基线） | 夏桢 Base（merge，历史差异参考） |
|---|---|---|
| 模型状态 | 未训练 Base，无 LoRA | 未训练 Base，无 LoRA |
| 评测入口 | 统一推理服务 | `eval_badcase.py` 直接调用 `model.infer/infer_multi` |
| 多页 prompt | `<image>Multi page parsing.` | `<image>Multi page merge.`（不适用于 Base） |
| 单页图像处理 | `image_size=1024`，`crop_mode=False` | `base_size=1024`，`image_size=640`，`crop_mode=True` |
| 多页 `ngram_window` | `1024` | `128` |
| OmniDocBench Total | `53.3645` | `51.7663` |
| AI Builder Overall | `0.8271` | `0.7442` |
| Text accuracy | `0.7995` | `0.4634` |

两次 Overall 都按
`(Text accuracy + Table accuracy + Reading-order accuracy + Title accuracy) / 4` 计算，
所以 `53.3645` 和 `51.7663` 的差异不是 Overall 公式造成的。最明显的差异是 Text accuracy：
`0.7995` 对 `0.4634`。夏桢的 `51.7663 / 0.7442` 是已删除 badcase 产物的废弃历史审计，
不得作为主 Base 或用于训练方案的正式泛化结论；周报和方案登记统一以 parsing Base 为主。

## 启动

显式设置 Base 与两个 adapter 路径后执行：

```bash
source ms_swift_title_mask/scripts/hyx_env.sh
MODEL_PATH=/absolute/path/to/unlimited-ocr-base \
FULL_CE_ADAPTER_PATH="$UOCR_ROOT/output/readoc-heading-prior-16k-full-ce/v0-20260831-173734/checkpoint-212" \
TITLE_WEIGHTED_ADAPTER_PATH="$UOCR_ROOT/output/readoc-heading-prior-16k-title-weighted/v0-20260831-173734/checkpoint-212" \
SERVICE_PORT=18080 \
CUDA_VISIBLE_DEVICES=0 \
  ms_swift_title_mask/scripts/run_inference_service.sh
```

`MODEL_PATH` 必须指向 Unlimited-OCR Base；两个 adapter 变量必须各自指向可加载的 LoRA。
可通过 `SERVICE_HOST`、`SERVICE_PORT`、`SERVICE_DEVICE` 和 `CUDA_VISIBLE_DEVICES` 调整监听和设备。
健康检查应显示三个模型名和 `trained_generation=peft_model.generate`。

请求示例：

```bash
curl http://127.0.0.1:18080/infer \
  -H 'content-type: application/json' \
  -d '{"model":"unlimited-ocr-title-weighted","image_paths":["/path/to/page.png"]}'
```

## 批量评测

正式入口是 `ms_swift_title_mask/scripts/probe_evaluation_pdfs_parallel.py`：每个 endpoint 传入
一个 `--url`，按 `(file, model)` 恢复并保存原始 response。129-PDF 正式评测使用
`image_size=1024`、`crop_mode=False`、`temperature=0`、`no_repeat_ngram_size=35`、单页
`ngram_window=128`、多页 `ngram_window=1024` 与 `max_length=20480`。

## 已清理历史

2026-08 的旧服务、PID、GPU/端口占用、旧 wrapper、`probe_evaluation_pdfs.py` 与
`evaluation_failure_cases_2026-08-25/` 均已删除，不得引用或恢复。当前正式结果、参数和
产物位置分别以评测矩阵、Title Prior 评测记录与周报为准。
