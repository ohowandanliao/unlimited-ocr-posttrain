# 夏桢两次 READoc Title-mask 训练归档

更新时间：2026-09-04（Asia/Shanghai，历史结果状态与开源路径整理）

本文记录 `$WXZ_ROOT/unlimited-ocr-finetune` 中两次 2026-08-24 完成的 READoc LoRA 训练。
后续方向以 `posttrain_weekly_report_2026-08-30.md` §5 为准。

## 1. 结论摘要

两次训练都没有实际启用梯度累积。日志中的 `grad_accum=1` 表示累积因子为 1，
配合 `batch_size=1`，每个样本都会触发一次参数更新。

| 方案 | 训练数据行数 | 更新步数 | epoch | 输出权重 |
|---|---:|---:|---:|---|
| READoc-1706 / Title-mask（无 Title Prior） | 1706 | 1706 | 1 | `outputs/readoc_title_mask_16k/` |
| READoc-1697 + Title Prior / Title-mask | 1697 | 1697 | 1 | `outputs/readoc_heading_prior_title_mask_16k/` |

两次训练的 `max_steps` 分别等于对应训练 JSONL 的行数，因此各完整遍历训练集一次。
训练脚本使用 `max_steps` 控制训练，没有记录一个独立的 `num_train_epochs` 字段；上表
的 1 epoch 是依据行数、batch 和 max_steps 计算得到的。

## 2. 训练数据

训练数据和验证/测试数据仍以夏桢目录中的原始 JSONL 为唯一实体，posttrain 不复制大文件。
下面同时记录行数和 SHA256，便于后续确认没有换数据。

### READoc-1706 / Title-mask：无 Title Prior

| split | 路径 | 行数 | SHA256 |
|---|---|---:|---|
| train | `$WXZ_READOC_ROOT/readoc_uocr/r0/views/train_le_16k_with_title.jsonl` | 1706 | `74b21cd95261ed695660c5d99333190199482fcb7e7b7f1987a084e3c1f27fd9` |
| validation | `$WXZ_READOC_ROOT/readoc_uocr/r0/views/val_le_16k.jsonl` | 95 | `3f36bf349c50de2448711b2a1f9b9a5b492d296a1990aa8a9970c71a1c67b72b` |
| test | `$WXZ_READOC_ROOT/readoc_uocr/r0/views/test_le_16k.jsonl` | 95 | `30cff2a18923091e06428bd4342bd25952ee3fddf5c38b0d15056e7747627bd5` |

训练日志的 `train_jsonl` 实际指向上表的 train 文件；该训练脚本没有在配置中传入
validation/test 文件。

### READoc-1697 + Title Prior / Title-mask

| split | 路径 | 行数 | SHA256 |
|---|---|---:|---|
| train | `$WXZ_READOC_ROOT/readoc_uocr_heading_prior/views/train_le_16k_with_title.jsonl` | 1697 | `0a27c0cbfe0af5e8d3a53a3cc2eb65f868bf16dfabeb273ae62da0a86ad67fba` |
| validation | `$WXZ_READOC_ROOT/readoc_uocr_heading_prior/views/val_le_16k.jsonl` | 94 | `eaf8cc3ea8bb974b926f92945643430ed2fcfbfe5c31c9b45a420028c46f1c20` |
| test | `$WXZ_READOC_ROOT/readoc_uocr_heading_prior/views/test_le_16k.jsonl` | 94 | `31e9e099bd379db86ddc4d904ea75d6f22f3797ce4cbe723057aa372b514f867` |

训练日志的 `train_jsonl` 实际指向上表的 train 文件；该训练脚本同样没有在配置中
传入 validation/test 文件。

共同基础模型：

```text
$MODEL_PATH
```

## 3. 训练方案和权重

两次训练日志中的共同配置如下：

| 配置 | 值 |
|---|---|
| train mode | `lora_decoder` |
| loss | `title_mask` |
| 数据模式 | `multi_base` |
| R-SWA | 开启，window=`128` |
| batch size | `1` |
| gradient accumulation | `1`，未做有效累积 |
| seed | `0` |
| learning rate | `1e-4` |
| warmup steps | `85` |
| max grad norm | `1.0` |
| 保存频率 | 每 `100` steps |
| LoRA | `84` 个 decoder 模块，rank=`16`，alpha=`32`，dropout=`0.05` |
| gradient checkpointing | 开启 |
| attention implementation | 日志记录为 `eager` |

### 实验 A

训练日志：

```text
$WXZ_ROOT/unlimited-ocr-finetune/logs/readoc_title_mask_16k.log
```

最终权重目录：

```text
$WXZ_OUTPUT_ROOT/readoc_title_mask_16k/
```

目录根部的 `adapter_model.safetensors` 是训练结束时保存的最终 LoRA；中间权重为
`step_100`、`step_200`，直到 `step_1700`。因为保存频率是每 100 steps，没有
`step_1706`，最终状态在根目录文件中。日志末尾为 `1706/1706` 和 `train_ok`。

### 实验 B

训练日志：

```text
$WXZ_ROOT/unlimited-ocr-finetune/logs/readoc_heading_prior_title_mask_16k.log
```

最终权重目录：

```text
$WXZ_OUTPUT_ROOT/readoc_heading_prior_title_mask_16k/
```

目录根部的 `adapter_model.safetensors` 是训练结束时保存的最终 LoRA；中间权重为
`step_100`、`step_200`，直到 `step_1600`。日志末尾为 `1697/1697` 和 `train_ok`。

## 4. 废弃 badcase 历史审计（产物已删除）

### Base 口径说明：夏桢 Base 与主 Base 的区别

本文的夏桢 Base 是未训练 Base、没有 LoRA，但它不是 posttrain 的正式主基线。主 Base 使用
统一 READoc 评测入口和 `<image>Multi page parsing.`；夏桢历史评测使用
`eval_badcase.py` 直接调用 `model.infer/infer_multi`，多页 prompt 为
`<image>Multi page merge.`。`merge` 是训练模型对应的 prompt，不适用于未训练 Base，
所以不能用夏桢 Base 替代主 Base。

| 项目 | 主 Base（正式主基线） | 夏桢 Base（本文历史参考） |
|---|---|---|
| 多页 prompt | `<image>Multi page parsing.` | `<image>Multi page merge.` |
| 单页处理 | `image_size=1024`，`crop_mode=False` | `base_size=1024`，`image_size=640`，`crop_mode=True` |
| 多页 `ngram_window` | `1024` | `128` |
| OmniDocBench Total | `53.3645` | `51.7663` |
| AI Builder Overall | `0.8271` | `0.7442` |
| Text accuracy | `0.7995` | `0.4634` |

两边都是未训练 Base，Overall 算法也相同，均为四项准确度的平均；分数差异来自 prompt、
评测入口和推理配置，不能解释为 Base 权重差异。本文中的两套 Title-mask 只在夏桢自己的
badcase 入口内与 `0.7442` 比较；posttrain 周报和正式泛化结论统一以 parsing Base
`0.8271` 为主。

这批评测使用了不同入口和 prompt，且带 prior 模型没有收到标题段。相关预测、渲染页图、
评分 JSON 与旧评测产物均已删除；下列数字只用于说明该协议为何废弃。

### 4.1 结果对应关系

| 方案 | 历史分数 | 产物状态 |
|---|---|---|
| READoc-1706 / Title-mask（无 Title Prior） | `25.9391 / 0.6635` | 已删除；废弃历史审计 |
| READoc-1697 + Title Prior / Title-mask | `33.3329 / 0.6801` | 已删除；废弃历史审计 |

### 4.2 OmniDocBench 分数摘要

| 指标 | Base | READoc-1706 / Title-mask | READoc-1697 + Title Prior / Title-mask |
|---|---:|---:|---:|
| Total Score | 51.7663 | 25.9391 | 33.3329 |
| text_block Edit_dist | 0.5366 | 0.5843 | 0.5704 |
| display_formula Edit_dist | 0.5745 | 0.9997 | 0.9254 |
| table TEDS | 0.6641 | 0.3622 | 0.4958 |
| reading_order Edit_dist | 0.0853 | 0.1485 | 0.1342 |

Total Score 越高越好；Edit_dist 越低越好；table 的 TEDS 越高越好。Title Prior
表面差异是标题段缺失造成的协议假象，不是 prior 正收益；不得用于方案比较。

### 4.3 AI Builder 四项与 Overall

Text/Table/Reading-order accuracy 由对应的 OmniDocBench `Edit_dist` 转换为
`1 - Edit_dist`。标题格式可能没有学会，因此 Title accuracy 不读取原始 Markdown
ATX heading 分，而是从原始 grounding 输出中提取 `<|det|>title [...]<|/det|>`，
对标题内容做去空格、标点和符号后的 Levenshtein 匹配；标题数量差异按较大标题数惩罚。
该算法已用 READoc-1706 的三份已知结果复核通过。

本批 GT 有标题的 88 个文档、共 581 个标题进入标题分母；GT 无标题的文档不进入标题分母；
缺少的预测 `Handwriting document.md` 按 0 计入。Overall 严格按四项算术平均：
`(Text accuracy + Table accuracy + Reading-order accuracy + Title accuracy) / 4`。

| 指标 | Base | READoc-1706 / Title-mask | 相对 Base | READoc-1697 + Title Prior / Title-mask | 相对 Base |
|---|---:|---:|---:|---:|---:|
| Text accuracy | 0.4634 | 0.4157 | -4.77 pp | 0.4296 | -3.38 pp |
| Table accuracy | 0.8124 | 0.7501 | -6.23 pp | 0.7743 | -3.81 pp |
| Reading-order accuracy | 0.9147 | 0.8515 | -6.33 pp | 0.8658 | -4.89 pp |
| Title accuracy（格式归一/grounding 内容匹配） | 0.7862 | 0.6368 | -14.95 pp | 0.6506 | -13.56 pp |
| **AI Builder Overall（四项算术平均）** | **0.7442** | **0.6635** | **-8.07 pp** | **0.6801** | **-6.41 pp** |

Base Overall 为 `0.7442`；无 Title Prior 为 `0.6635`；有 Title Prior 为 `0.6801`。这些
废弃协议下的历史数字不支持任何 prior 收益结论。

## 5. 归档边界

- 训练 JSONL、PDF、GT 和模型 base 没有复制，路径变量和校验值记录在本文。
- 旧 badcase 预测 Markdown、渲染页图和评分 JSON 已删除，不再保留归档副本。
- READoc-1706 的 Full-CE/Title-weighted 权重仍在
  `output/readoc-view-16k-*/`，不受本次历史归档影响。
- 本文只记录事实，不将这两次归档训练设为下一轮默认启动入口。
