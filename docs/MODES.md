# 输入 mode:base/gundam × single/multi（现状、目标、怎么训）

## mode 是什么
输入图像的分辨率/切块策略，决定 image token 数、能看多细、一次几页。
- **base**：整张 1024×1024，不切块，~256（本模型实测 273）token。省 token、小字易糊。
- **gundam**：动态切块高清 —— n×640×640 局部 crop + 1×1024×1024 全局，token = n×100 + 256。本模型 gundam 用 12 块（3×4 网格）→ ~1513 token，看得清小字。
- **single / multi**：一次喂 1 页 / 多页。

## Unlimited-OCR 的两种公开视觉配置
- **gundam**：1024 global + 640 dynamic crops；官方只用于单页。
- **base**：1024 global、无 crop；单页和多页都支持。
- 因此公开组合是 `single_gundam`、`single_base`、`multi_base`，没有 `multi_gundam`。R-SWA 负责长文/多页生成时的 KV cache 有界，不会自动增加图像分辨率。
- 对比：DeepEncoder 原生有 Tiny512 / Small640 / Base1024 / Large1280 + Gundam；Unlimited-OCR 的公开调用只给出 base + gundam 两种配置。

官方 README 明写 `Multi page / PDF only uses base (image_size=1024)`，`infer_multi` 的 docstring 也明写 `Does NOT support crop mode`。它用全零 `dummy_crop` 进入 forward 的多图 no-crop 分支；crop 分支只按一张图的 `[W,H]` 组织 local crops，不能处理“每页各自一组 crops”。

ms-swift 4.4.2 没有改变这个模型边界。它会把多图展开并逐图预处理，但最终仍把所有页的 crops/global views 合成一个样本交给官方 forward；真正产生 crop 时会维度失配。stock ms-swift 的可用多页配置仍是：

```bash
CROP_MODE=false IMAGE_SIZE=1024 BASE_SIZE=1024 swift sft ...
```

这些是进程级环境变量，所以 ms-swift 可在一次 base 训练中混合单页/多页，不能按样本混合 gundam/base。

## DeepSeek-OCR 是「多 mode 混合训练」
- 5 种分辨率 mode **一起训**（gundam 与 4 个原生 mode 同时训），让**一个模型支持所有分辨率**，推理时按需选 token 预算（这就是"contexts optical compression"的灵活性）。
- 训练分阶段：Stage1 单训 DeepEncoder（视觉）；Stage2 编码器-解码器联合训，数据是 OCR+视觉+文本混合；再 SFT。
- 结论：**不是只用 gundam 训 —— 是多 mode 混着训。**

## 我们的现状 vs 目标
- **现状**：`single_gundam`、`single_base` 的 forward 已实测；`multi_base` 的 forward 与 R-SWA 训练已实测。各正式 config 目前仍会用 `mode:` 统一覆盖数据，这是具体实验选择，不是 processor 限制。
- **风险**：只训一种 mode 可能弱化其它 mode（尤其全参微调时）；LoRA 冻结 base 相对安全。
- **多页主目标**：多页输入 → 一份跨页合并的连续 Markdown；输入侧先用已验证的 `multi_base` 即可验证这个行为目标。
- **`multi_gundam`**：属于额外的高清多页输入能力，**超出 Unlimited-OCR 和 stock ms-swift 的原生能力**。它需要新写 forward 的逐页 crop 分支，而且典型 12-crop 页面约 1513 个视觉 token，5 页已约 7565 个，不能只改一个 mode 名称。
- **建议**：正式训练按任务混合 `single_gundam` + `multi_base`；是否投入 `multi_gundam`，应与跨页合并行为的训练分开评估。

## 来源
- DeepSeek-OCR 论文 arXiv:2510.18234；GitHub `deepseek-ai/DeepSeek-OCR`
- Unlimited-OCR：`baidu/Unlimited-OCR`（HF）、MarkTechPost 报道、`deepwiki/baidu/Unlimited-OCR`
