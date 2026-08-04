# ms-swift 4.4.2 与本仓库训练差异（2026-08-04）

## 结论

ms-swift 4.4.2 已能在单张 RTX 4090 上训练 Unlimited-OCR，并自动启用 R-SWA 训练 mask。当前实验只能比较两套现有训练方案，不能把质量差异单独归因于框架，因为 tokenizer 边界、EOS、optimizer 默认值、样本顺序和依赖版本并未全部一致。

图像输入能力方面，官方和本仓库的可用交集仍是 `single_gundam`、`single_base`、`multi_base`。ms-swift 新增了 Unlimited-OCR 模板和多图占位符展开，但**没有补上模型的多页 crop forward**，因此不能把 stock ms-swift 当成可用的 `multi_gundam` 实现。

同一批 300 条训练数据、同一组 6 条 held-out 的结果：

| 方案 | held-out similarity | 训练情况 |
|---|---:|---|
| 本仓库 `lora_decoder_rswa_v1` | **0.905** | 约 9 分钟 |
| ms-swift stock（`beta2=0.95`、监督 EOS） | 0.826 | 8 分 11 秒，峰值 15.07 GiB |
| ms-swift 对齐实验（`beta2=0.999`、不监督 EOS） | 0.767 | 7 分 57 秒；一条样本超长失控 |

6 条 held-out 很小，以上结果用于工程决策和发现差异，不作为框架质量定论。

## 图像 mode 与多页支持

| 输入方式 | 官方 Unlimited-OCR | ms-swift 4.4.2 | 本仓库 |
|---|---|---|---|
| `single_gundam` | 支持；1024 global + 640 dynamic crops | 支持，且是默认配置 | 支持，已实测 |
| `single_base` | 支持；1024 global、无 crop | 支持；需全局设置 `CROP_MODE=false IMAGE_SIZE=1024` | 支持，已实测 |
| `multi_base` | 支持；官方 README 明写 "Multi page / PDF only uses base" | 支持；同样需 `CROP_MODE=false IMAGE_SIZE=1024` | 支持，forward 和训练均已实测 |
| `multi_gundam` | **不支持**；`infer_multi` docstring 明写 `Does NOT support crop mode` | **不支持**；模板会预处理 crops，但 stock model forward 无法按页消费 | **未实现**；需要改 remote model forward |
| 多页合并成一份连续 Markdown | 无内建能力；原生行为是逐页输出并用 `<PAGE>` 分隔 | 框架可用 `multi_base` 喂自定义 SFT target，但不自动提供合并能力 | 数据/训练管线可表达；小规模 smoke 尚未改变基座原生行为 |

ms-swift 的 `CROP_MODE`、`BASE_SIZE`、`IMAGE_SIZE` 是进程级环境变量，不是样本字段。因此 stock 模板可在同一次 base 训练中混合单页和多页，但不能在同一次训练里按样本混合 gundam/base；本仓库的 `mode` 则是逐样本字段，也可由 config 全局覆盖。

### 为什么 ms-swift 看起来支持多页 gundam，实际不支持

ms-swift `UnlimitedOCR._encode` 会把一个 `<image>` 展开成相邻的 N 个 `<image>`，`DeepseekOCR._preprocess_image` 也会逐页生成 global view、local crops 和 `[width, height]`。无 crop 时，这与官方 `infer_multi` 在单个位置串接 N 页视觉 token 的结果等价。

问题在模型端：

1. ms-swift 把一个多页样本的所有 local crops 合成一个 `images_crop`，所有 global views 合成一个 `images_ori`。
2. `UnlimitedOCRModel.forward` 的 crop 分支只取一个 `crop_shape`，把全部 local features 当成一张图的一个网格。
3. 它得到多页 global features `[N, HW, D]` 后直接执行 `view(H, W, D)`；这个写法隐含 `N == 1`，多页时元素数不匹配。
4. 官方 `infer_multi` 因此明确传全零 `dummy_crop`，强制进入 no-crop 分支；该分支才会遍历 `image_ori` 的 N 页。

所以，默认 `CROP_MODE=true` 下给 ms-swift 多页数据，并不代表获得了 `multi_gundam`：大图真正产生 local crops 后会在 forward 失败。所有页都小到不产生 crop 时可能走通，但本质仍是 global-only，不能视为 gundam 支持。

真正实现 `multi_gundam` 至少要同时改三层：processor 保留每页 crop 分组或 offset，collator 保留每个样本的页边界，model forward 按页切 crops、分别与对应 global view 拼接。并且 token 预算很重：典型 12-crop 页面约 1513 个视觉 token，5 页就约 7565 个，还未计 prompt/target；当前 `max_length=8192` 与 eager 训练下不适合直接扩到很多页。

## 已确认一致

- 数据：300 条 olmOCR 训练样本，batch size 1，gradient accumulation 2，150 optimizer steps。
- 优化主参数：LR 1e-4、warmup 15、cosine、weight decay 0.01、max grad norm 1.0。
- LoRA：rank 16、alpha 32、dropout 0.05；相同 84 个模块，3.9782M trainable parameters。
- 模型路径：bf16、eager attention、non-reentrant gradient checkpointing，冻结视觉与 aligner。
- R-SWA：reference 全通、answer 最近 128 token；当前 batch size 1 下两边 mask 语义一致。
- 图像预处理：single_gundam，base 1024、crop 640，image token 布局一致。

## 仍有差异

### 1. Prompt/response 分词边界

本仓库先拼接 prompt 与 target，再联合 tokenize；ms-swift 分开 tokenize 后拼 token IDs。同一真实样本出现过：

```text
text:     document parsing. | An agent ...
ms-swift: "."(16), "An"(2677), ...
本仓库:   ".An"(106718), ...
```

本仓库再按 prompt-only 长度 mask labels，会把跨边界的 `.An` 整个 mask，首词 `An` 不参与监督。推理只编码 prompt、再生成 response，因此分开 tokenize 在语义上更贴近推理。该改动尚未合入训练代码，应作为单变量 A/B，不能和 EOS 等改动混跑。

### 2. EOS

ms-swift 默认在 response 结尾增加并监督 EOS，本仓库不加。stock 实验中两条输出提前结束；去掉 EOS 后另有一条输出失控到 29187 字符。EOS 对停止行为影响大，目前没有足够证据修改本仓库默认值。

### 3. AdamW 默认值

ms-swift 默认 `beta2=0.95`，本仓库直接使用 PyTorch AdamW 默认 `beta2=0.999`。后续实验应把 `betas`、`eps`、`weight_decay` 全部显式写入配置，避免框架默认值造成隐性差异。

### 4. Shuffle 与依赖版本

两边虽然都是 seed 0，但模型/LoRA 初始化消耗 RNG 的时机和 dataset shuffle 实现不同，实际样本顺序不保证一致。本仓库使用 Transformers 4.57.1 / PEFT 0.19.1；ms-swift Unlimited-OCR 路径使用 Transformers 4.46.3 / PEFT 0.15.2。

### 5. Gradient accumulation 的 loss 权重

两边都先对单个 microbatch 内的有效 target token 求平均，但跨 microbatch 的权重不同。设第 i 个 microbatch 有 `n_i` 个受监督 token、平均 loss 为 `L_i`：

```text
本仓库梯度目标:  mean_i(L_i)
ms-swift 梯度目标: sum_i(n_i * L_i) / sum_i(n_i)
```

因此本仓库是“每篇文档等权”，ms-swift 是“每个 target token 等权”；当两个 target 长度差很多时，梯度确实不同。这是当前实验中另一个未对齐变量。

此外，本仓库旧日志在 `grad_accum=2` 时只记录第二个 microbatch。本次已改为记录 optimizer step 内全部 microbatch 的均值，并输出全程 first/last/mean。新日志与本仓库实际优化目标一致，但仍不是 ms-swift 的 token 加权口径；历史 `0.2659 vs 0.2373` 和后续两框架的 step loss 都不能直接逐项比较。

### 6. 长度策略

ms-swift 显式设置 `max_length=8192` 并可删除超长样本；本仓库旧逻辑没有上限保护。本次增加 `max_length` 与 `length_strategy`：默认 `error`，输出样本 ID 和实际长度；可显式设为 `drop`。不提供静默 truncate，因为截断 OCR target 会制造不完整文档标签。

### 7. Mode 配置与多图 collator

本仓库按样本读取 `mode`，并把 `images_spatial_crop` 保持为“batch -> sample -> page/crop shape”的边界；ms-swift 的图像 mode 是进程级配置，collator 会把 `images_spatial_crop` 沿第 0 维拼平。单图 crop 和多页 no-crop 都能工作，但这个布局不能直接复用于未来的多页 crop forward。

### 8. R-SWA 开关与 padding 行

ms-swift Unlimited-OCR 模板在训练时根据模型 `sliding_window_size` 自动构造 R-SWA mask；本仓库由 `rswa_train` 显式开关。batch size 1、无 padding 时两边语义已确认一致。batch 内有右侧 padding 时，ms-swift 还屏蔽 padding query 行，本仓库只屏蔽 padding key 列；padding labels 都是 `-100`，所以当前监督 loss 不受影响，但实现并非逐元素完全相同。

### 9. 工程能力范围

ms-swift 自带 Trainer、DDP/FSDP/DeepSpeed、量化和较完整的断点/指标能力，更适合双卡或多卡扩展。本仓库是单进程薄循环，但有 Unlimited-OCR 专用的逐样本 mode、`multi_base` 数据契约、自定义 full-backbone 冻结范围和更直接的调试面。两者都不会凭框架本身获得多页合并或 `multi_gundam` 能力。

## 后续公平 A/B 要求

1. 只改为 prompt/target 分开 tokenize，保持无 EOS 和现有 optimizer 不变。
2. 明确选择“每文档等权”或“每 token 等权”，让两边 gradient accumulation 使用同一归一化。
3. 固定并导出同一条样本 permutation，两边按同一顺序训练。
4. 锁定 Transformers/PEFT 版本，或至少先验证同一初始 adapter 的 forward/loss 一致。
5. 使用完整 30 条 held-out，除相似度外记录生成 token 数和停止原因。
