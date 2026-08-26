# Unlimited-OCR ms-swift post-training bundle

这是 ms-swift 的 external plugin 与训练工具，不是独立训练框架。当前入口是
[`TRAINING_PLAN_ZH.md`](TRAINING_PLAN_ZH.md)，脚本分类见 [`scripts/README.md`](scripts/README.md)；
旧 2026-08-23 title-mask run 不再是默认方案。

## Recipe 与 loss

| `TRAINING_RECIPE` | 数据要求 | 允许的主实验 |
|---|---|---|
| `readoc_r0` | 仅 READoc full document | `full_ce` |
| `replay_r1` | READoc + 独立中文 replay；无 PMC | `full_ce` |
| `pmc_s10` | PMC 字符代理占比不超过 12%，正文已接受，无重复 PMCID/已知污染 | `full_ce` |
| `trusted_title` | 标题全部 `HUMAN_ACCEPTED`；PMC 仍受 S10 gate | `full_ce`、`uniform_ce`、`title_weighted` |
| `legacy_20260823` | 只复现旧快照 | 需额外设置 `UOCR_ALLOW_LEGACY_20260823=1` |

`title_mask` 只允许历史复现。新标题消融先证明 `uniform_ce` 与 native `full_ce` 在同一
gradient-accumulation batch 上 loss/gradient 一致，再比较 `title_weighted`。新 custom CE 使用 ms-swift 提供的
`num_items_in_batch`，与 native CE 采用相同累计 token 分母；旧 1:2 checkpoint 的逐 micro-step active-weight
归一化没有被追溯改写。

标题权重通过 `TITLE_WEIGHT` 显式设置，默认 `1.5`；EOS 默认 `EOS_WEIGHT=1.0`。只有人工验收标题可进入该消融。

## 安全边界

- `MODEL_PATH`、`TRAIN_JSONL`、`VAL_JSONL`、`OUTPUT_DIR`、`TRAINING_RECIPE`、`LOSS_MODE` 均为必填。
- 非 legacy 默认 LR `2e-5`、WD `0.01`、warmup `0.03`；不再继承旧 `1e-4/0.1/0.05`。
- 每次训练前自动执行行级数据校验和 recipe composition gate，并在外部写 recipe audit JSON。
- 只允许 1/2 卡、4K-32K、SDPA/eager；正式建议 SDPA 32K，不截断、不 packing。
- LoRA callback 必须精确命中 84 个 decoder 模块。
- `hyx_env.sh` 只提供环境骨架，不设置模型、训练 JSONL、输出目录、recipe 或 loss 默认值。

`build_recipe_mix.py` 输出 `recipe_mix_pool`；`build_length_buckets.py` 同时兼容该字段和旧快照的 `mix_pool`，
因此新 recipe 可以直接进入 tokenizer 长度分桶。PDF、页图和 JSONL 始终保留在仓库外。
