# READoc 长度审计与训练改造方案：ms-swift + R-SWA（更新于 2026-08-19）

> **长度分析保留，训练队列已被取代。** 当前唯一执行文档是
> [`../ms_swift_title_mask/TRAINING_PLAN_ZH.md`](../ms_swift_title_mask/TRAINING_PLAN_ZH.md)。本文件中的
> `lora_attn`、full-backbone/full-decoder、FlexAttention 等仅是历史分析，不进入 2026-08-23 的
> reviewed-title 训练；正式只用 ms-swift decoder-backbone LoRA。

> 状态：基于 READoc 全量原始数据审计的工程决策。目标是**复用 Unlimited-OCR 权重，以原始页面图像输入，输出完整、连续的 Markdown，并保持官方 R-SWA 语义**。

## 0. 先看结论

1. **主训练方案**：最新 `ms-swift` + 原生 `multi_base` 图像输入 + `lora_decoder + R-SWA`。不把 MinerU/MinerU-Popo 的 OCR/Layout 输出作为模型输入。
2. **监督目标**：完整 Markdown（标题、正文、表格、公式、图片说明和跨页内容），不是只输出标题。第一版对完整 assistant target 计算 loss。
3. **四套方案定位**：`lora_attn` 只做 4K smoke；`lora_decoder` 是正式主线；`full_backbone` 只做短序列容量 A/B；`full_decoder/full_lm` 暂不进入队列。
4. **H100 长度顺序**：数据完整建立 4K/8K/16K/24K/32K/overflow 桶；4K/8K 只跑 5-10 step smoke，随后在单张 80GB H100 上直接探测 16K、24K、32K，正式 baseline 使用原生 ms-swift 能稳定 backward 的最高档 `L_base`。
5. **长序列改造是条件性的**：若 H100 上原生 dense eager R-SWA 的 `L_base < 32K`，且需要覆盖剩余长样本，才在 ms-swift 中接入解析 `BlockMask` 的 FlexAttention，并同时接入 fused linear cross-entropy。
6. **标题专门 loss 不是主线**：若把完整 Markdown 的正文 labels 全部 mask 掉，必须给 R-SWA collator 额外传真实 `prefix_len`；否则 loss mask 会改变 attention 边界。若任务本身只输出标题层级 JSON，则只需换 target，不需改通用 Trainer。
7. **不承诺单卡 113K backward**。FlexAttention 只能消除 attention 的二次方物化；线性 activation、LM head 和 `T x R` 计算仍需实测。

## 1. 固定语义

### 1.1 什么叫“复用 Unlimited-OCR 权重”

本文要求的不只是 checkpoint 能加载，还要求 attention 行为保持一致：

- Q/K/V/O projection 及参数 shape 不变。
- `10` 个 query heads、`10` 个 KV heads、head dim `128` 不变。
- RoPE 的位置编号、旋转方式和 attention scale 不变。
- reference、target 和 padding 的边界定义不变。
- R-SWA window 保持 `W=128`。
- residual、normalization、MoE 和层顺序不变。
- 训练得到的 LoRA adapter 仍能加载回未修改的官方推理模型。

允许的差异只有 fused kernel 的浮点归约顺序。FlexAttention 的 online softmax 不要求与 eager bitwise 一致，但短序列 logits、loss 和梯度必须在预先设定的数值容差内一致。

### 1.2 R-SWA 的精确规则

设：

- `R`：reference 长度，即 prompt + 所有图像 token。
- `T`：assistant target 长度。
- `L = R + T`：完整 teacher-forcing 序列长度。
- `W = 128`：output window，包含当前位置。

允许关系为：

```text
                         Key
                  Reference       Output
Query Reference   causal prefix     forbidden
Query Output      all reference     recent W outputs
```

逐 token predicate 为：

```text
causal = kv_idx <= q_idx

allowed = valid_query
          and valid_key
          and causal
          and (
              q_idx < prefix_len
              or kv_idx < prefix_len
              or q_idx - kv_idx < W
          )
```

这和 ms-swift 当前 dense R-SWA mask 的语义相同。要改变的是执行方式，不是这个 predicate。

## 2. FA3 的边界

官方 SGLang/论文中的 FA3 结果针对 R-SWA 的**逐 token 推理 decode**；公开训练设置是 Megatron-LM + DeepEP、最大 32K，并没有公开可直接复用的 FA3 teacher-forcing kernel。当前 Unlimited-OCR remote code 在 Transformers/ms-swift 中仍注册 `mha_eager -> SlidingWindowLlamaAttention`，没有可直接切换的 `mha_flash_attention_3` 训练实现。

因此第一版不通过 `--attn_impl flash_attention_3` 宣称解决训练 OOM。先跑现有 eager baseline，只有长度 gate 证明必要时才做本文件第 5 节的条件性改造。

## 3. 32K 对 READoc 实际是多少页

当前 `multi_base` 每页约 273 个视觉 token。

### 3.1 只看图片 prefill

```text
32,768 / 273 ~= 120 pages
96 x 273 = 26,208 reference tokens
```

所以 96 页图片 reference 本身能放进 32K，但只剩：

```text
32,768 - 26,208 = 6,560 tokens
```

这 6,560 还要容纳 prompt、完整 OCR target、EOS 和少量格式 token。

### 3.2 看完整 READoc 训练样本

READoc 最大样本可粗略拆为：

```text
R = 26,213
T = 113,193 - 26,213 = 86,980

average target/page = 86,980 / 96 ~= 906
average total/page  = 273 + 906 ~= 1,179
```

因此相同文本密度下：

```text
32,768 / 1,179 ~= 27 pages
```

未扣 prompt/EOS 时理论值约 27 页，实际安全预算约 25-27 页。

| 平均 target token/页 | 每页总 token | 32K 理论页数 |
|---:|---:|---:|
| 300 | 573 | 57 |
| 500 | 773 | 42 |
| 906（READoc 最大样本） | 1,179 | 27 |

因此“96 页能 prefill”和“96 页完整 OCR 能在 32K 内训练/生成”是两件不同的事。官方 2-50 页训练数据能够 pack 到 32K，是因为页面文本密度和样本长度不同；不能把 50 页直接外推到 READoc 的密集 96 页样本。

### 3.3 READoc 全量长度审计（2026-08-18）

本次审计直接读取 `READoc` 原始目录中的 2,233 篇文档（1,009 篇 arXiv、1,224 篇 GitHub）：PDF 页数从 ZIP 流式读取，target 使用服务器上的 Unlimited-OCR `AutoTokenizer` 复核。长度估算采用当前 `multi_base` 的 273 visual tokens/page、BOS/EOS 和 `Multi page parsing.`/`Multi page merge.` 的 4 个 prompt tokens。没有把图片解码或模型权重加载纳入统计。

| 指标 | P50 | P90 | P95 | P99 | 最大 |
|---|---:|---:|---:|---:|---:|
| 页数 | 7 | 21 | 28 | 48 | 120 |
| reference `R` | 1,916 | 5,738 | 7,649 | 13,109 | 32,765 |
| target `T` | 3,354 | 15,403 | 20,659 | 36,188 | 86,980 |
| 总长度 `L` | 5,440 | 20,845 | 28,209 | 46,878 | **113,193** |

| 长度上限 | 覆盖样本 | 覆盖率 | 超限样本 |
|---:|---:|---:|---:|
| 8K | 1,458 / 2,233 | 65.29% | 775 |
| 16K | 1,900 / 2,233 | 85.09% | 333 |
| 32K | 2,167 / 2,233 | **97.04%** | 66 |
| 64K | 2,223 / 2,233 | 99.55% | 10 |
| 98,304 | 2,231 / 2,233 | 99.91% | 2 |

最长样本是 `arxiv/2210.15829`：96 页，`R=26,213`、`T=86,980`、`L=113,193`。它是全量数据中的单个极端点；达到 96 页的文档只有 4 篇（0.18%）。页数不是充分的长度指标：一个 120 页 GitHub 文档总长度约 48K，而 47 页的高密度 arXiv 文档可超过 77K。因此训练必须按 `L` 分桶，不能按“页数 <= N”代替。

非累计长度桶如下：

| 总长度桶 | 样本数 | 占比 |
|---:|---:|---:|
| `<=8K` | 1,458 | 65.29% |
| `(8K,16K]` | 442 | 19.79% |
| `(16K,24K]` | 168 | 7.52% |
| `(24K,32K]` | 99 | 4.43% |
| `(32K,48K]` | 46 | 2.06% |
| `(48K,64K]` | 10 | 0.45% |
| `(64K,96K]` | 8 | 0.36% |
| `>96K` | 2 | 0.09% |

arXiv 与 GitHub 的长度分布差异很大：

| 来源 | 样本数 | `L` P50 | `L` P95 | `L` P99 | 最大 `L` | 8K 覆盖 | 16K 覆盖 | 32K 覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arXiv | 1,009 | 11,316 | 35,407 | 62,308 | 113,193 | 29.93% | 67.69% | 93.66% |
| GitHub | 1,224 | 3,162 | 8,692 | 13,611 | 81,765 | 94.44% | 99.43% | 99.84% |
| 全部 | 2,233 | 5,440 | 28,209 | 46,878 | 113,193 | 65.29% | 85.09% | 97.04% |

最长的 5 条为：

| 文档 | 页数 | `R` | `T` | `L` |
|---|---:|---:|---:|---:|
| `arxiv/2210.15829` | 96 | 26,213 | 86,980 | 113,193 |
| `arxiv/2209.11383` | 85 | 23,210 | 76,670 | 99,880 |
| `arxiv/2205.09653` | 55 | 15,020 | 69,181 | 84,201 |
| `arxiv/1409.6015` | 72 | 19,661 | 62,722 | 82,383 |
| `github/318522019` | 105 | 28,670 | 53,095 | 81,765 |

这项审计改变了实施顺序：数据侧直接准备到 32K 并保留剩余 66 条 overflow；H100 上只用 4K/8K 验证链路，随后直接探测 16K/24K/32K 的真实 backward 峰值。只有原生路径无法达到 32K、且业务确认需要覆盖被排除长样本时，才进入 FlexAttention、fused CE 和序列并行的工程路径。上述统计仍是训练前审计；拿到最终训练 JSONL 后，需要再检查实际 prompt、页分组、截断和 packing 是否引入额外 token。

### 3.4 分布对训练方式的直接影响

1. `8K` 只适合工程 smoke，不适合作为 H100 上的正式 READoc 训练口径。它虽然覆盖 65.29% 全量数据，却只覆盖 29.93% 的 arXiv，会把训练分布严重推向短 GitHub 文档。
2. `16K` 是 H100 上第一个正式候选，覆盖 85.09% 全量、67.69% arXiv；通过后立即继续探测 24K 和 32K，不为 16K 预先设硬上限。是否能 backward 必须由显存曲线决定，不能由 LoRA 参数量或 GPU 型号推断。
3. `32K` 才接近完整训练分布，覆盖 97.04% 全量、93.66% arXiv。但当前 eager 的单层 fp32 score 理论上已达 40 GiB，因此“数据应以 32K 为上限”和“当前实现能训练 32K”是两个不同结论。
4. ms-swift 跨 gradient-accumulation microbatch 按有效 target token 加权。长文档天然获得更大梯度权重；比较不同训练方案时必须复用相同的长度桶、样本顺序和每档 token 预算，不能只对齐 optimizer steps。
5. 训练预算应同时报告文档数、target token 数和总 token 数。`150 steps` 在 4K 与 32K 上不是同一训练量，也不是公平的 LoRA/全参对照。
6. 不允许静默右截断完整 target。超限样本要么进入独立清单，要么按可验证的页/章节边界重建监督；READoc 原始 Markdown 没有天然可靠的逐页 target，不能仅按 token 生切。

### 3.5 原四套训练方案的重新评估

仓库存在两个历史口径。最初设计的四个模式是 `lora_attn`、`lora_decoder`、`full_decoder`、`full_lm`；后来又增加了更实用的 `full_backbone`。当前四个主要实验配置则是“全 causal LoRA、R-SWA LoRA、R-SWA full-backbone、R-SWA full-decoder”。两种口径的最终决策一致：READoc 第一阶段只保留一条正式主线。

#### 按参数更新范围看

| 模式 | 当前实际训练范围 | 可训练参数量 | 对长序列峰值的作用 | 决策 |
|---|---|---:|---|---|
| `lora_attn` | 12 层 Q/K/V/O LoRA | rank 8 实测约 0.98M；rank 16 约 1.96M | 只省 optimizer/梯度参数；不减少 attention、activation 或 logits | **保留为 smoke/最小消融**，不作为效果主线 |
| `lora_decoder` | Q/K/V/O + dense MLP + shared experts LoRA；排除 64 routed experts | rank 16 实测 3.98M | 同样不解决长度 OOM，但参数稳定、adapter 小 | **唯一正式主线** |
| `full_decoder` | `model.layers.*`，含 64 routed experts 与 router | 约 2,604M | 在相同 activation 之外再增加约 53 GiB 级 master/grad/optimizer 压力 | **暂停**；2,233 篇数据不足以支持这一级别全参更新 |
| `full_lm` | `full_decoder` + `lm_head`，embed 可选 | 默认再加约 165M；训练 embed 再加约 165M | 不解决 logits 物化；仅 lm_head 的 fp32 AdamW 相关状态就再增加约 2.47 GiB | **移出当前实验队列**；保留代码，词表和输出 token 未变化时没有训练 head/embed 的必要 |

`full_backbone` 是后来增加的第五种范围：全参训练 attention、dense MLP 和 shared experts，排除 routed experts，约 182M 参数。它比 `full_decoder` 合理，但仍远大于 LoRA，并且 24 GiB“已跑通”只证明短样本可行，不能外推到 8K/16K。它只作为 LoRA 出现明确容量上限后的第二阶段 A/B，且先固定在 `<=8K` 的同一数据子集。

所有模式在给定 `L` 下都执行相同的 decoder forward 和 LM head。LoRA 只减少可训练参数、梯度与 optimizer state，不会减少 `(L,L)` mask、attention scores、hidden activation 或 `[L,V]` logits。因此“换成 LoRA/全参”不是长度方案。

#### 按当前四个主要配置看

| 当前配置 | 角色 | 结合 READoc 分布后的结论 |
|---|---|---|
| `lora_decoder_olmocr_v1`（全 causal） | 负对照 | 只保留用于证明 train/infer mask 不一致有害；已有 held-out `0.465` vs R-SWA `0.709`，不再作为 READoc 候选 |
| `lora_decoder_rswa_v1` | 正式主线 | 迁移到 ms-swift；4K/8K 只 smoke，H100 直接探测 16K/24K/32K，正式 baseline 使用最高稳定档 `L_base` |
| `full_backbone_rswa_v1` | 容量对照 | 暂不跑长序列；只有 LoRA 在同一数据/预算下表现出容量瓶颈时，才在 `<=8K` 做单变量 A/B |
| `full_decoder_rswa_v1` | 大规模全参实验 | 暂停；它同时需要参数分片和长序列方案，会把两个问题混在一起，当前数据规模也容易造成 expert 稀疏更新与过拟合 |

#### 映射到 ms-swift 时的约束

- `lora_attn` 可用明确的 `q_proj/k_proj/v_proj/o_proj` targets 表达。
- `lora_decoder` 不能直接使用默认 `all-linear`。默认值会扩大到 routed experts，改变现有 3.98M 参数口径；必须复用当前模块 regex，并在启动日志中核对 `attn=48、dense_mlp=3、shared_experts=33、routed_experts=0`。
- ms-swift 的 Unlimited-OCR `language_model` 注册范围包含 `embed_tokens`、`layers`、`norm` 和 `lm_head`。因此 `--train_type full --freeze_vit true --freeze_aligner true` 更接近 `full_lm`，并不等价于本仓库的 layers-only `full_decoder`；若以后做全参实验，必须显式冻结或解冻这些模块。
- `full_backbone` 也不能只靠 `freeze_vit/freeze_aligner` 表达，需要先冻结 LLM，再用精确 parameter regex 只解冻 attention、dense MLP 和 shared experts。
- 四个方案的公平比较必须固定：R-SWA、分开 tokenize、EOS 口径、optimizer betas、样本 permutation、长度桶和累计 target-token 预算。否则结果不能归因于参数更新范围。

最终实验顺序收敛为：

1. `lora_attn + R-SWA`：4K 小样本 smoke，只验证数据、loss、保存和 reload。
2. `lora_decoder + R-SWA`：4K/8K 各跑 5-10 step smoke；只验证工程链路，不把 8K 结果作为正式 READoc baseline。
3. 同一个 `lora_decoder` 在单张 80GB H100 上依次探测 16K/24K/32K；以原生 ms-swift 可稳定 backward 的最高档作为正式 `L_base`，只有 `L_base < 32K` 时才评估 memory-efficient attention。
4. 只有 LoRA 显示容量不足，才增加 `full_backbone <=8K` 对照；`full_decoder/full_lm` 不进入当前实验队列。

### 3.6 训练改造方案（当前执行口径）

#### A. 第一阶段 baseline：不改 ms-swift 核心

这一阶段的目的，是先得到一个可归因的完整 Markdown 基线，而不是同时验证新 kernel、标题任务和新输入格式。

| 项目 | 当前决定 |
|---|---|
| 训练框架 | 最新 `ms-swift`，使用其 SFT/LoRA trainer |
| 模型输入 | READoc 原始页面图像，`multi_base`；沿用 Unlimited-OCR 官方 SAM+CLIP/projector |
| 外部 OCR 输入 | **不使用 MinerU/MinerU-Popo 的文本、bbox 或类别作为模型输入** |
| 输出 target | 一份完整连续 Markdown：标题、正文、表格、公式、图片说明和跨页内容 |
| loss | 对完整 assistant target 计算普通 token CE；不做 title-only mask |
| 参数范围 | `lora_decoder`，精确排除 routed experts；复用现有 regex |
| attention | 当前 ms-swift 的 dense R-SWA，`W=128` |
| tokenize | reference/prompt 与 target 分开统计；EOS、shift 和 `ignore_index=-100` 固定 |
| 长度 | 4K/8K 短 smoke → H100 16K/24K/32K memory/throughput probe → 最高稳定档 `L_base` 正式训练；超限样本分桶或留存，不静默截断 |

训练 JSONL 应保留图像和完整 assistant 内容。训练日志至少同时报告文档数、有效 target token 数、总序列 token 数、每个长度桶的 loss 和显存峰值；只比较 optimizer steps 会让长样本获得不公平的权重。

#### B. 标题 loss 的两个不同含义

**任务本身只输出标题层级 JSON/列表。** 这是一个独立的标题结构任务，可以把标题层级结果作为 assistant target，使用正常 SFT，不需要修改 ms-swift 通用 Trainer。它不再是完整 Markdown OCR 主线，也不能用来证明正文、表格或公式能力。

**仍输出完整 Markdown，但只让标题 token 参与 loss。** 这不是当前主线，且不能只通过把正文 labels 改成 `-100` 完成。当前 Unlimited-OCR R-SWA collator 用第一个 `labels != -100` 的位置推断 `prefix_len`；正文被 mask 后会把 loss mask 错当成 attention 边界。若未来要做这个消融，需要：

1. 在 token loss mask 生成前保存真实的 `response_start/prefix_len`。
2. 将标题 mask 作为独立 `loss_scale`/labels 信息传入 collator。
3. 让 R-SWA 只读取未 mask 前的 `prefix_len`，而 loss 只读取标题 mask。
4. 用正文、表格和跨页样本验证 mask 后仍生成完整 Markdown；不能只看 title loss。

ms-swift 已支持自定义 `loss_scale`。标题加权（例如标题 2 倍、正文 1 倍）比 title-only 更合理，但需要 `is_binary_loss_scale=false`，并单独报告加权与未加权 loss。无论哪一种 title loss，都不会消除 attention、hidden activation 或完整序列的计算，因此不是长序列显存方案。

#### C. 何时修改 ms-swift

只有原生 ms-swift 在 H100 上的 16K/24K/32K 探测确认 dense eager attention 或 LM head 阻止达到所需上限时，才进入第 5 节的条件性改造：

1. 保留完整 Markdown target 和 `lora_decoder` 参数范围。
2. 将 dense `(B,1,L,L)` mask 替换为解析构造的 R-SWA `BlockMask`，只改训练/no-cache 分支。
3. 同时将全量 `lm_head -> logits.float() -> CrossEntropyLoss` 替换为 training-only fused linear CE。
4. 通过 512/4K 的 mask、logits、loss、gradient parity 后，再逐档测试 24K/32K；不先承诺 96 页或 113K。

FlexAttention、fused CE 和 sequence/context parallel 都是后续的长度工程，不改变本阶段的输入、输出或监督语义。

## 4. 为什么当前 96 页 OOM

### 4.1 仅 reference 已到 eager OOM 点

```text
96 x 273 = 26,208 visual reference tokens
```

这与实测约 26K 开始 OOM 一致。也就是说，在加入 87K target 之前，当前 eager attention 已不可行。

### 4.2 dense eager 的单层 score

Unlimited-OCR 有 10 个 attention heads。eager softmax 使用 fp32 时，完整 score tensor 为：

```text
10 x L x L x 4 bytes
```

| L | 单层 fp32 scores | 结果 |
|---:|---:|---|
| 3,921 | 0.57 GiB | 可运行 |
| 8,192 | 2.50 GiB | 只用于 smoke，不代表正式上限 |
| 16,384 | 10.00 GiB | H100 第一个正式候选，必须实测完整 backward |
| 24,576 | 22.50 GiB | H100 探测档，临时张量和反向可能使其不可行 |
| 32,768 | 40.00 GiB | 单个 score 已占 80GB H100 的一半，不能预设可训练 |
| 113,193 | 477.31 GiB | 完全不可行 |

这还不包含同层 softmax/cast 临时张量、其他层为 backward 保留的 activation、模型权重、Q/K/V、hidden states、MoE、LoRA、梯度、LM head 和 optimizer state。H100 的显存容量允许跳过完整 8K baseline，但没有改变 dense eager 的 `O(L^2)` 增长，因此 16K/24K/32K 都要用真实样本逐档测量。

### 4.3 ms-swift 的 R-SWA mask 仍是逐 token 方阵

ms-swift 当前 collator 构造 `(B, 1, L, L)` additive mask。`L=113,193` 时：

```text
one bf16 mask = 23.87 GiB
one bool mask = 11.93 GiB
```

所以“打开 R-SWA”本身不会省显存。当前实现只是先构造完整方阵，再让 eager 计算完整 scores，最后把禁止位置遮掉。

### 4.4 reference 多了，FlexAttention 是否仍会爆

可能仍会，但原因会改变。

精确 R-SWA 的有效 attention 对为（`T >= W`）：

```text
R(R+1)/2 + T*R + W(W+1)/2 + (T-W)*W
```

对最长样本的 `R=26,213`、`T=86,980`、`W=128`：

```text
allowed pairs/head = 2,634,705,843
full causal pairs/head = 6,406,384,221
```

其中最大项是：

```text
T * R = 2,280,006,740 pairs/head
```

FlexAttention 的作用是分块计算并在线归一化，不保存所有 score，也不创建逐 token 方阵：

| 资源 | dense eager R-SWA | FlexAttention 等价 R-SWA |
|---|---|---|
| `L x L` additive mask | 需要 | 不需要 |
| `H x L x L` scores | 需要 | 不物化，按 tile 流式计算 |
| Q/K/V、hidden states | 随 L 线性增长 | 仍随 L 线性增长 |
| block metadata | 无 | 在 block 粒度增长 |
| `T x R` 计算 | 实际还会算大量被 mask 项 | **仍必须计算允许的 `T x R`** |

所以 FlexAttention 会移除当前 25-477 GiB 的结构性 score 峰值，但不会让 reference 免费：

- reference K/V 随 `R` 线性增长。当前 `R=26,213`、hidden size 1280、bf16 下，每层 reference 的 K+V 约 127.99 MiB；不是 `T x R` 大小。
- target/hidden activation 随总长度 `L` 线性增长；本样本中 `T=87K` 反而是长度主体。
- `T x R` 计算量保持不变，step 可能非常慢。
- 若单卡线性 activation 仍超限，DDP/FSDP 不会自动切分单样本序列，需要真正的 sequence/context parallel。

在“所有 target 必须看全部 reference”的语义下，想进一步减少 `T x R` 只能增加硬件并行度；top-k 页面、视觉摘要或 reference pooling 都会改变模型行为，不属于本工程决策。它们可以成为独立的训练型研究方法，但必须用 full-reference 质量、匹配 token 预算和实际 wall time 单独验证，不能借用“等价 R-SWA”的结论。

这里必须区分“计算对数”和“驻留显存”。`T x R=2,280,006,740` pairs/head 必须被计算，但 FlexAttention 使用 tile + online softmax，不应保存一个 `T x R` score tensor。若 profiler 中真的出现这个矩形分配，说明实现退化成 dense，不是精确 R-SWA 天生要求它常驻显存。

### 4.5 attention 修好后，官方 LM head 仍会确定性 OOM

官方 `UnlimitedOCRForCausalLM.forward` 当前执行：

```python
logits = self.lm_head(hidden_states)
logits = logits.float()
loss = CrossEntropyLoss()(shift_logits, shift_labels)
```

词表大小为 `129,280`。在当前最大样本上：

| logits 范围 | shape | bf16 | 官方 FP32 |
|---|---:|---:|---:|
| 全序列 | `113,193 x 129,280` | 27.26 GiB | **54.51 GiB** |
| 仅 target | `86,980 x 129,280` | 20.95 GiB | **41.89 GiB** |

所以仅删除 reference 对应 logits 仍不够。生产训练必须使用 fused linear cross-entropy：按小块完成 `hidden @ lm_head.weight.T + CE`，直接产生 loss 和梯度，不保存全量 logits。普通 Python `for chunk` 后把 loss 相加不一定省 backward 显存，因为 autograd 可能为每个 chunk 都保留 softmax 状态；应使用 Liger fused linear CE 或具有自定义 backward/recompute 的等价实现。

这个改动不改变模型权重、词表或 loss 定义，只改变线性层与交叉熵的执行方式。训练短序列时仍需与官方 FP32 logits + CE 比较 loss 和 hidden/lm-head gradient。

## 5. 长序列条件性改造（仅在 gate 触发后）

本节是 H100 上原生 ms-swift 的 16K/24K/32K 探测证明无法达到所需上限之后的施工方案，不是当前立即修改 `ms-swift` 的承诺。4K/8K 仅用于链路 smoke；如果原生路径已经能稳定训练 32K，本节全部暂缓。

### 5.1 当前调用链

当前 ms-swift Unlimited-OCR 训练路径是：

```text
UnlimitedOCR.data_collator
  -> build dense [B,1,L,L] R-SWA mask
  -> loader patch passes 4D mask through
  -> SlidingWindowLlamaAttention eager forward
  -> materialize [B,H,L,L] scores
```

目标路径改为：

```text
data_collator
  -> keep labels + 1D valid-token mask

top-level model forward
  -> derive prefix_lens and valid seq_lens
  -> analytically build one BlockMask per batch

DeepseekV2Model
  -> pass BlockMask unchanged through all decoder layers

SlidingWindowLlamaAttention training branch
  -> original Q/K/V projection + original RoPE
  -> compiled flex_attention(block_mask=...)
  -> original output projection

generation branch
  -> unchanged official ring-buffer implementation
```

BlockMask 必须在进入 12 层 decoder 前只构造一次，不能每层重复创建。

### 5.2 prefix 边界来源

baseline 保持 ms-swift 当前定义：每个样本第一个 `labels != -100` 的位置是 `prefix_len`。如果未来启用第 3.6 节的 title-only loss，不能继续从已经 mask 的 labels 推断边界，必须在 loss mask 生成前把真实 `response_start/prefix_len` 单独保存并传入 collator。

```text
prefix_lens[b] = first_non_ignored_label_position
seq_lens[b] = attention_mask_1d[b].sum()
```

第一版只支持当前实际需要的：

- microbatch size 1。
- 右 padding 或无 padding。
- `use_cache=false`。
- `output_attentions=false`。
- `attention_dropout=0`（官方配置即为 0）。

长序列本来就是 microbatch 1。先把这一条路径做对，再扩 batch 内不同 prefix/padding，避免一开始同时调试多个变量。

### 5.3 不能直接使用 `create_block_mask`

PyTorch 的 `create_block_mask(mask_mod, ..., Q_LEN=L, KV_LEN=L)` 当前实现会先调用 `create_mask`，生成 token 级 bool tensor，再压缩成 block metadata。

因此下面这种看似正确的写法在 113K 上仍会 OOM：

```python
# 仅可用于短序列正确性测试，禁止用于真实长序列。
block_mask = create_block_mask(rswa_mask, 1, 1, L, L, device="cuda")
```

生产路径必须根据 R-SWA 的固定结构直接构造 block rows：

```python
BlockMask.from_kv_blocks(
    kv_num_blocks=...,
    kv_indices=...,
    full_kv_num_blocks=...,
    full_kv_indices=...,
    BLOCK_SIZE=128,
    mask_mod=rswa_mask,
    seq_lengths=(L, L),
)
```

### 5.4 解析构造 block rows

选择 `BLOCK_SIZE=128`，与 R-SWA window 对齐。

对每个 query block：

1. 若整个 query block 位于 reference 内，候选 KV blocks 为从 0 到 causal diagonal block。
2. 若 query block 包含 target，则加入所有 reference blocks。
3. 再加入覆盖最近 128 个 target token 的 1-2 个 local blocks。
4. 对 prefix 边界、causal diagonal、local-window 边界和最后一个 padding block 使用 `mask_mod` 做逐元素精确判断。
5. 只有能证明整个 tile 全部允许时，才放进 `full_kv_indices`；其余放进 partial blocks。

候选范围可写成：

```text
if q_block_end < prefix_len:
    candidates = [0 .. q_block_id]
else:
    candidates = all reference blocks
    local_start = max(prefix_len, q_block_start - W + 1)
    candidates += blocks(local_start .. q_block_end)
```

最终是否允许仍由第 1.2 节的精确 predicate 决定，因此 prefix 未按 128 对齐也不会改变语义。

对当前最大样本：

```text
Q blocks   = ceil(113,193 / 128) = 885
Ref blocks = ceil(26,213 / 128)  = 205
max candidate blocks/row ~= 207
```

一个 `[885, 207]` 的 int32 `kv_indices` 主表约 0.70 MiB。加上 backward transpose、num-blocks 和 full/partial 表仍是 MiB 级，而不是 11.94-23.88 GiB 的逐 token mask。

实现顺序：

1. 短序列用 `create_block_mask` 生成 oracle。
2. 实现解析 block builder。
3. 在 512/4K 上比较两者 `to_dense()` 完全相等。
4. 长序列只使用解析 builder，禁止调用 `to_dense()`。

### 5.5 patch `SlidingWindowLlamaAttention`

只给 training/no-cache 分支增加 FlexAttention；官方 decode 分支原样保留。

核心结构如下，实际实现应直接复用 remote model code 的 Q/K/V 和 RoPE 逻辑：

```python
def flex_training_forward(self, hidden_states, block_mask, position_ids):
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    q = reshape_heads(q)
    k = reshape_kv_heads(k)
    v = reshape_kv_heads(v)

    cos, sin = self.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)

    out = compiled_flex_attention(
        q, k, v,
        block_mask=block_mask,
        scale=1.0 / sqrt(self.head_dim),
    )
    return self.o_proj(merge_heads(out))
```

必须显式检查：

- q/k/v layout 为 `[B, H, L, D]`。
- bf16 forward/backward。
- head dim 128。
- softmax 累积精度与 eager 的 fp32 softmax误差范围。
- LoRA 包装后的 `q_proj/k_proj/v_proj/o_proj` 仍被调用。
- gradient checkpointing 重算时 BlockMask 仍可用。

### 5.6 ms-swift 接入点

建议在 ms-swift fork 中做三个局部改动：

1. `swift/template/templates/deepseek.py`
   - 删除 UnlimitedOCR collator 中的 dense `_build_rswa_attention_mask` 调用。
   - 保留 1D valid-token mask 和 labels。
2. `swift/model/models/deepseek.py`
   - 在 `UnlimitedOCRLoader` 安装 FlexAttention training patch。
   - top-level forward 在模型 device 上构造 BlockMask，避免 Trainer 搬运自定义对象。
   - `_prepare_4d_causal_attention_mask` 遇到 `BlockMask` 时原样传递。
3. 新增独立 `rswa_flex.py`
   - 精确 predicate。
   - 解析 BlockMask builder。
   - patched training forward。
   - 短序列 oracle 和自测。

要求：

- `BlockMask` 每个 batch 构造一次并在所有层共享。
- 显式设置 `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`。
- pin 已验证的 PyTorch/CUDA/Triton 组合；FlexAttention 仍是 PyTorch prototype feature。
- 使用 `torch.compile` 后记录首次编译时间与稳定 step time，不能把 compile 时间混进吞吐。
- 长度变化导致反复 recompile 时，采用少量长度 bucket 或验证 dynamic shape；不能静默产生几十份 kernel。

### 5.7 必须同时 patch fused linear cross-entropy

训练分支不再调用官方全量 logits 路径，而是先选出真正参与 next-token loss 的 hidden rows：

```python
shift_hidden = hidden_states[..., :-1, :]
shift_labels = labels[..., 1:]
keep = shift_labels != -100

loss = fused_linear_cross_entropy(
    lm_head.weight,
    shift_hidden[keep],
    shift_labels[keep],
)
```

要求：

- 使用 Liger `LigerFusedLinearCrossEntropyLoss` 或经过同等测试的 fused 实现；不能在 forward 中累计保存所有 chunk logits。
- ms-swift 的通用 `--use_liger_kernel` 不会自动改写 Unlimited-OCR remote code 里手写的 `CrossEntropyLoss`，loader patch 必须显式接入。
- 只在 `training and labels is not None` 的路径启用；eval logits 和 generation 保持官方实现。
- reduction、shift、`ignore_index=-100` 和 FP32 softmax 累积语义保持一致。
- profiler 中不得出现完整 `[L,V]` 或 `[T,V]` logits allocation。

### 5.8 视觉路径不是当前主峰值

96 张 bf16 `1024 x 1024` RGB 图片的原始张量约为：

```text
96 x 3 x 1024 x 1024 x 2 bytes ~= 576 MiB
```

但官方 `multi_base` forward 已在 `torch.no_grad()` 中逐页调用 SAM/CLIP，并不是把 96 页同时送入视觉编码器。因此视觉中间 activation 不会简单放大 96 倍。它仍应单独记录峰值，但当前不需要先改视觉模型；decoder attention 才是已证实的 26K OOM 主因。

## 6. 分阶段验证与停止条件

### Gate A：长度与数据策略（先于任何 kernel 改动）

- 用最终训练 JSONL 重新统计 `R/T/L`，确认 prompt、页分组、EOS、截断和 packing 没有改变本节审计结论。
- 数据 manifest 一次性建立到 4K/8K/16K/24K/32K/overflow，不让显存探测结果反向删除数据。
- 单张 80GB H100 上先做 4K/8K 各 5-10 step smoke，再以 `micro_batch_size=1` 使用接近档位上界的真实样本探测 16K、24K、32K forward/backward；不以 96 页样本作为第一轮验收条件。
- 将连续稳定运行且显存、吞吐留有合理余量的最高档记为 `L_base`，所有 R0/S10 主实验固定同一个 `L_base`、长度桶和 target-token 预算。
- 超过当前上限的样本保留在独立清单，不静默截断 target。若要纳入训练，必须定义页/段落边界切分，并单独验证切分后的监督质量。
- 训练集与 READoc 评估集必须按文档 ID 分离；不能用同一篇 ground truth 训练后再报告它的 benchmark 分数。

长度 gate 未通过时，停止 FlexAttention 施工，先修正数据构造或训练预算。

### Gate 0：mask 正确性

- 构造 prefix 未对齐 128、target 小于/等于/大于 128、带 padding 的小样本。
- dense oracle 与解析 BlockMask `to_dense()` 逐元素完全一致。
- 特别检查第一个 target、窗口第 128/129 个位置、prefix 边界 block。

失败则停止，不进入模型测试。

### Gate 1：attention 数值正确性

- `L=512` 和 `L≈4K`。
- dropout=0、固定 seed、相同 bf16 输入。
- 比较 eager 与 FlexAttention 的 attention output、最终 logits、loss、Q/K/V/O LoRA gradient。
- 做一次 optimizer step 后比较参数更新。

允许浮点误差，不允许 mask 边界或梯度结构差异。

### Gate 2：显存曲线

原生 ms-swift 的硬件基线先按以下长度测试 forward 和 backward：

```text
4K/8K smoke -> 16K -> 24K -> 32K
```

进入 FlexAttention/fused CE 条件性实现后，先用 512/4K 做数值 parity，再重跑 `16K -> 24K -> 32K`。只有 32K 正确性、显存和质量 gate 全部通过，才追加 `64K -> 113K`，不因 H100 直接跳档。

每档记录：

- BlockMask 构造峰值和耗时。
- vision forward 后峰值。
- 单层 attention forward/backward 峰值。
- fused linear CE forward/backward 峰值，确认没有全量 logits。
- 整步 peak allocated/reserved。
- compile time、稳定 step time、tokens/s。

预期结果不是“113K 必须成功”，而是确认不再出现逐 token `L x L` mask/scores。

### Gate 3：reference 仍然导致 OOM 时怎么判断

| 新瓶颈 | 证据 | 下一步 |
|---|---|---|
| BlockMask 构造 OOM | attention 尚未进入就爆 | builder 错误；检查是否误用了 `create_block_mask/to_dense` |
| Flex kernel workspace OOM | Q/K/V 已建立，进入 kernel 后爆 | 调 block size/kernel option；记录可复现最小长度 |
| LM head/logits OOM | decoder 已完成，在 `lm_head` 或 CE 爆 | fused linear CE patch 未生效；检查是否出现 `[tokens,129280]` |
| hidden/MoE activation OOM | 单层 attention 能过，完整 decoder 失败 | non-reentrant GC；仍失败则 sequence/context parallel |
| optimizer/参数状态 OOM | forward/backward 能过，step 时爆 | LoRA、optimizer state/ZeRO；这时参数分片才有用 |
| 不 OOM但 step 极慢 | `T x R` kernel 时间占主导 | 增加 GPU 并行；否则降低训练长度，不能宣称 96 页已解决 |

普通双卡 DDP 会让两张卡各自持有完整样本，只提升全局吞吐，不降低单个 113K 样本显存。真正需要的是沿 sequence/context 维切分 query/KV；而 ms-swift 现有 sequence-parallel 路径未必能直接接 FlexAttention，必须单独验证或实现。

### Gate 4：长度质量

官方公开训练最大为 32K。即使 64K/113K kernel 能运行，也要分别验证：

- 32K 内结果不回退。
- 32K 以上 RoPE/位置外推是否稳定。
- 生成是否提前 EOS、复读或遗漏后半文档。
- 跨页段落、表格和章节是否随长度恶化。

如果 113K loss 正常但生成失败，应先归因到长度分布/位置泛化，而不是继续改 attention kernel。可以做 32K -> 64K -> 113K 长度 curriculum，但每一档都必须保持同一 R-SWA 语义。

## 7. 验收标准

### 正确性

- 解析 BlockMask 与短序列 dense oracle 完全一致。
- eager 与 FlexAttention 的 logits/loss/gradient 在约定容差内。
- prefix、window、padding 边界测试全过。
- 新 backend 不增加或修改任何模型参数 key。
- LoRA adapter 可加载到未修改的官方 ring-buffer inference model。

### 显存

- profiler 中不存在逐 token `[L,L]` mask allocation。
- profiler 中不存在 `[H,L,L]` score allocation。
- BlockMask metadata 为 block 粒度，113K 构造过程不调用 `to_dense/create_mask`。
- loss 路径不存在完整 `[L,129280]` 或 `[T,129280]` logits allocation。
- 报告每个 Gate 长度的真实 peak，而不是只报告理论复杂度。

### 性能

- 分离 compile time 与 steady-state step time。
- 报告 attention、MoE、vision、optimizer 各阶段耗时。
- 113K 若无法接受，明确记录是显存失败还是 `T x R` 速度失败。

### 质量

- 单页和短多页 held-out 不回退。
- 32K 内与 eager R-SWA 生成一致。
- 64K/113K 单独报告位置外推和后半文档完整性。
- 不用 teacher-forcing loss 代替长生成评估。

## 8. 实施顺序

1. 用最终训练 JSONL 完成 Gate A，按 `L` 分桶并锁定第一阶段上限。
2. 在未改动的 ms-swift 上跑 4K/8K 短 smoke，再跑 H100 16K/24K/32K 的 forward、backward、显存和吞吐探测，锁定正式 `L_base`。
3. 不把 title-only 或 MinerU 输入混入 baseline；如需 title 消融，先单独验证 prefix metadata 和 loss mask。
4. 只有基线确认需要更长长度时，才在 ms-swift fork 新增 `rswa_flex.py`，先实现短序列 predicate 和 oracle。
5. 实现 `BlockMask.from_kv_blocks` 解析 builder，并证明与 oracle 一致。
6. patch Unlimited-OCR training attention；推理分支保持不动。
7. patch training-only fused linear CE，禁止全量 logits。
8. 跑 512/4K attention、loss 和 gradient parity，再重跑 `16K/24K/32K` 显存曲线。
9. 只有 32K gate 通过后才跑 64K/113K，并按 Gate 3 分类剩余瓶颈；单卡 memory-efficient 路径正确后，才评估 sequence/context parallel，不把 DDP/FSDP 误当作序列切分。

在 Gate A 和 Gate 2 完成前，不再宣称“FlexAttention 能训练 96 页”；准确表述是：**FlexAttention 只能消除 attention 的二次方物化，必须与 fused linear CE 一起验证；两者之后仍可能因线性 activation OOM，或因 `T x R` 计算过慢而不具备训练价值。**

## 9. 依据

- Unlimited-OCR 官方论文，第 7 页 FA3 decode kernel study；第 8 页 32K、2-50 页、Megatron-LM + DeepEP 训练设置：<https://github.com/baidu/Unlimited-OCR/blob/main/Unlimited-OCR.pdf>
- 官方 SGLang FA3 启动参数：<https://github.com/baidu/Unlimited-OCR/blob/main/README.md>
- 官方模型配置（10 heads、12 layers、hidden size 1280、vocab 129280）：<https://huggingface.co/baidu/Unlimited-OCR/blob/main/config.json>
- 官方训练 forward（全量 `lm_head`、`logits.float()`、`CrossEntropyLoss`）：<https://huggingface.co/baidu/Unlimited-OCR/blob/main/modeling_unlimitedocr.py>
- ms-swift Unlimited-OCR dense R-SWA collator：<https://github.com/modelscope/ms-swift/blob/main/swift/template/templates/deepseek.py>
- ms-swift Unlimited-OCR loader：<https://github.com/modelscope/ms-swift/blob/main/swift/model/models/deepseek.py>
- ms-swift 自定义 `loss_scale`：<https://github.com/modelscope/ms-swift/blob/main/docs/source/Customization/Architecture.md>
- MinerU-Popo 输入与后处理任务说明：<https://github.com/opendatalab/MinerU-Popo/blob/main/README_zh.md>
- PyTorch FlexAttention / BlockMask API：<https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html>
- Liger fused linear cross-entropy：<https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/fused_linear_cross_entropy.py>
