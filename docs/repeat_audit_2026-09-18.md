# Multi page merge 复读根因排查(2026-09-18,只读)

> 性质:按给定清单对多页 LoRA SFT 做只读排查——不复述文档结论,全部数字从服务器原始文件/既有评测产物取得;全程未跑 GPU、未重跑任何评测、除 `/tmp` 外未写任何远端文件。
> 范围:两轮训练路线 `pmc-fullce-16k`(1160 步)与 `pmc-readoc-spage-16k`(1623 步);对照官方 Unlimited-OCR 原始 pipeline(`~/hyx/models/Unlimited-OCR/` 自带 modeling 代码)与 0827 基线(base/full-ce/title-weighted)。
> ms-swift 框架本身按可信处理(用户拍板),只审查自有改动:posttrain 仓库 `ms_swift_title_mask/`、数据构建脚本、训练入口、评测服务。
> 产物:数据审计脚本与报告留档 `evaluation/audit_repeat_20260918/`(本目录);case 取证报告见 §7 所注。

---

## 1. 结论速览

按可能性从高到低(证据链见 §8):

| # | 原因 | 一句话证据 | 判定 |
|---|---|---|---|
| 1 | **R-SWA 128-token 答案窗口的固有自循环倾向** | 未训练的 base 同解码配置下已有复读(0827 重复行≥10% 的文档 base 9/129;四规则严格退化 base 3 篇);case 解剖 6 篇中多数**第 0 个 token 即偏离 GT** 后进入循环,循环单元 115c~32k | 主因(架构固有) |
| 2 | **解码侧无有效防重复,加的 ngram 处理器只防精确重复** | 官方 `infer/infer_multi` 默认 `no_repeat_ngram_size=0`;项目加的处理器(35, window 128/1024)对计数器递增/近似重复无效,且会诱导"次优 token 变异"(8228→8229、210→211) | 放大因子 |
| 3 | **训练放大(非超参过强)** | LoRA 超参与 0827 fc 参考完全一致、routed experts 冻结、1 epoch、loss 无崩坏形态;但严格退化 base 3→pmc 9→mix 9(四规则口径,注意 max_length 跨轮 20480→32768 未对齐) | 放大因子(与步数混淆未分离) |
| 4 | 数据/截断/EOS 问题 | 0 截断、0 EOS 缺失、0 超预算、sha 全量对账一致、GT 无段落级复制(char 加权行重复率≤0.12%) | 基本排除 |
| 5 | 去掉 page boundary 导致状态丢失 | 不适用:词表本无 page 类 token,官方多页即"273 visual token/页 + 页间 1 个分隔 token",训练与评测完全一致,没有"去掉"动作 | 不成立 |
| 6 | 其他代码 bug(packing/拼接/label mask) | packing=false、padding_free=false、bs=1 无拼接;labels=模板原生;EOS 恰 1 个;LoRA target guard 校验 84 模块通过 | 未发现 |

---

## 2. 相对官方 pipeline 的全部改动(逐项 diff)

### 2.1 改动清单

| 维度 | 官方/原始 | 本项目 | 出处 |
|---|---|---|---|
| 多页 prompt | `<image>Multi page parsing.` | `<image>Multi page merge.`(单页两边同为 `<image>document parsing.`) | 官方:`modeling_unlimitedocr.py:1163`(infer_multi docstring)、`serve_unlimited_ocr.py:150`(base 默认);本项目:`ms_swift_title_mask/core.py:15`、数据全部行、`serve_unlimited_ocr.py:152` |
| 任务数据 | 官方无训练 | GT=跨页合并 Markdown(PMC v2.2.1 / READoc-1706 / spage 页拆);每行 1 个 PDF;构建期硬过滤 `total = 273×页数 + 4 + target ≤ 16384`(16k)/`≤32768`(32k) | `~/hyx/dataset/pmc-fullce-data/build_pmc_fullce_jsonl.py:16-19`、`build_meta.json` |
| 特殊 token/词表 | `<image>`(128815)、`<\|ref\|>`、`<\|/ref\|>`、`<\|det\|>`、`<\|/det\|>` | **零新增**;无任何 page 类 token | token 化审计:added tokens 全集 + input/target 0 命中 |
| loss | —(不训练) | `full_ce` = 原生 CE:labels 由模板构造(prompt 段 −100,response + 恰 1 个 EOS 监督);无加权、无 loss_scale | `_run_train.sh:206-247`(full_ce 不挂自定义 loss_type);plugin 的 title_mask/title_weighted 仅历史路由使用 |
| LoRA | — | rank 16 / alpha 32 / dropout 0.05;`target_regex` = attn q/k/v/o + dense-MLP + shared-experts(共 84 模块,guard 实测 attn 48/dense 3/shared 33/**routed experts 0 冻结**);freeze_vit/aligner | `_run_train.sh:196,212-215`;guard: `plugin/uocr_title_mask.py` UOCRLoRATargetGuard |
| 优化器/调度 | — | lr 2e-5、cosine、warmup 3%、wd 0.01、adam β2 0.95、clip 1.0、1 epoch、bs 1 × grad_accum 8(全局 8)、bf16、sdpa | `_run_train.sh:117-124,222-236` |
| 训练 max_length | — | 16384(16k)/32768(32k);`--truncation_strategy None`(依赖构建期硬过滤) | `run_pmc_fullce.sh:27-31`;`_run_train.sh:218-219` |
| **解码:防重复** | **官方默认 `no_repeat_ngram_size=0, ngram_window=0`(无任何防重)**、greedy(temperature=0) | 加了 `SlidingWindowNoRepeatNgramProcessor(35, window 单页128/多页1024)`,其余同官方 greedy | 官方:`modeling_unlimitedocr.py:811,1163`(签名默认 0);本项目:`serve_unlimited_ocr.py:45-48`、`probe_evaluation_pdfs_parallel.py:352-367`、responses.jsonl gen 实录 |
| 解码 max_length | 32768 | **两轮新模型 32768;0827 基线 20480**(跨轮不一致) | 两轮:responses.jsonl `gen.max_length=32768`;0827:`docs/evaluation_readoc_view_16k_2026-08-27.md:114-115` |
| 视觉输入 | infer_multi 默认 image_size=640(111 token/页) | image_size=1024、crop 关(273 token/页)——**与训练一致**(训练 env `CROP_MODE=false, IMAGE_SIZE=1024`) | 官方默认:`modeling_unlimitedocr.py:1163`;本项目:`serve_unlimited_ocr.py:182-193`、`_run_train.sh:193-195` |
| swift 仓库改动 | 上游 `1a1ba3ee8` | 仅 `swift/trainers/mixin.py` 6 行(compute_loss_func 兼容旧 transformers 构造签名);full_ce 路径无 loss_type,不触发 | `git diff HEAD`(ms-swift-uocr) |

### 2.2 未改动的官方机制(核对过,训练/推理一致)

- **R-SWA 训练 mask**(上游模板实现,窗口 = `config.sliding_window_size` = **128**):前缀(全部页图+prompt)对答案全可见,答案侧只看最近 128 个答案 token(`repos/ms-swift-uocr/swift/template/templates/deepseek.py:483-527`;窗口值:`models/Unlimited-OCR/config.json` `sliding_window_size: 128`)。
- **R-SWA 推理 ring buffer**(模型自带):prefill(全部页图+prompt)整段保留在 KV cache,之后每生成 1 token 写入 **W=128 的环形区**,覆盖最旧生成 token;注意力 = 全部前缀 + 最近 128 个已生成 token(`models/Unlimited-OCR/modeling_deepseekv2.py:1232-1397`;`config._ring_window=sw, config.sliding_window=None` 于 `modeling_unlimitedocr.py:1022-1024,1258-1260`)。**训练窗口 128 = 推理窗口 128,一致。**
- **多页视觉布局**:所有页的 visual token 在单个 `<image>` 位置拼接,**页间仅 1 个 `image_token_id` 分隔,没有任何文本页边界标记**(每页 16×17=272+1=273 token)。官方 `infer_multi` 如此(`modeling_unlimitedocr.py:1198-1214`),训练模板逐 `<image>` 占位展开同构。**这是官方约定,不是本项目引入或删除的。**
- greedy 解码、EOS 停止:两边一致。

### 2.3 一处无影响的观察

`plugin/uocr_title_mask.py` 的 `ReviewedTitleMaskUnlimitedOCR._encode` 里,`runtime_messages[0]` 的替换只传给了 `validate_conversation`,实际编码仍走 `super()._encode(inputs)`(占位符展开由原生模板完成)——逻辑冗余但无影响;且两轮 full_ce 根本不走该自定义模板。

---

## 3. 训练数据审计(token 化,全量 CPU 重算)

报告:`evaluation/audit_repeat_20260918/data_audit_report.json`(全数字)+ `data_audit_samples.txt`(3 样本);脚本 `uocr_audit.py`。

### 3.1 行数与 prompt

| 数据集 | 行数 | channel | `Multi page merge.` | `document parsing.` |
|---|---:|---|---:|---:|
| pmc-fullce-16k/train | 9284 | 全 title_reviewed | 8911 | 373 |
| pmc-readoc-spage-16k/train(mix) | 12990 = pmc 9284 + spage 2000 + readoc 1706 | 同上 | 10617 | 2373 |
| readoc-view-16k/train | 1706 | 同上 | 1706 | 0 |

spage 2000 行全部是单页样本(1 页/行),不属 merge。

### 3.2 merge 行页数分布(len(images))

| 集 | min | p50 | p90 | p95 | max | 桶计数 |
|---|---:|---:|---:|---:|---:|---|
| pmc | 2 | 10 | 13 | 14 | 20 | 2页:400 / 3:356 / 4:356 / 5:372 / 6-10:3758 / >10:3669 |
| mix | 2 | 9 | 13 | 14 | 35 | (readoc 部分 2:105 / 3:165 / 4:190 / 5:233 / 6-10:750 / >10:263) |
| readoc | 2 | 6 | 12 | 15 | 35 | 同上 |

### 3.3 target token 长度与截断

| 集 | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|
| pmc | 9098 | 12077 | 12462 | 13487 |
| mix | 7193 | 11791 | 12315 | 13643 |
| readoc | 2592 | 8480 | 10117 | 13643 |

- **截断比例 = 0**:按构建口径 `est_total = 273×页 + 4 + target`,三套 est>16384 与 est>32768 均为 **0 行**;`meta.total_tokens` 最大值恰 16384。唯一出入:实测模板 prompt 为 5 token(`<image>` 占 1),按此口径 pmc/mix 各有 5 行"超 16384" 1 个 token——纯口径差。
- **训练步数反推同样支持"无丢弃"**:9284/8 = 1160.5 → 实际 1160 步;12990/8 = 1623.75 → 实际 1623 步。两轮都全量消费,没有样本被截断策略丢弃。

### 3.4 EOS / page token

- `eos_token_id=1`(`<|end▁of▁sentence|>`);assistant 文本内含 eos 字符串的行 = **0**。
- swift 模板 `TEMPLATE_MAPPING['unlimited_ocr'].suffix = [[eos_token_id]]` → 每个样本 labels 末尾**恰好 1 个 EOS**,无其他尾巴;截断不存在,故不存在"截断样本缺 EOS"。
- `<page`/`<PAGE`/`PAGE>` 在 input 与 target 中 **0 命中**;词表无 page 类 token。

### 3.5 GT 自身重复(8/16/32-gram,token 级)

| 指标(行占比) | pmc | mix | readoc |
|---|---:|---:|---:|
| 任一 8-gram 重复≥2 次 | 99.5% | 98.0% | 91.6% |
| 8-gram ≥5 | 77.5% | 65.4% | 48.7% |
| 8-gram ≥20 | 18.5% | 14.0% | 5.9%(pmc max_cnt p50=8, p90=28, p99=76, max=344) |
| 16-gram ≥2 | 91.4% | 81.1% | 70.2% |
| 32-gram ≥2 | 57.8% | 46.9% | 37.0% |
| 行级重复率≥0.2(count) | 0.5% | 2.8% | 17.6% |
| 行级重复率≥0.2(**char 加权**) | ≤0.12% | ≤0.12% | ≤0.12% |

**解读**:8-gram 高重复几乎全部来自表格行、作者署名、参考文献等短行在长文档中的自然复现;按字符量加权的整段复制几乎不存在(≤0.12%)。GT 侧不存在"训练数据教模型复读段落"的证据——这与此前"循环内容在训练数据 0 命中"的取证一致。

### 3.6 trainer 所见 vs 原始数据

- 随机 300 行(seed 1234):`sha256(assistant)` vs `meta.title_target_sha256` **0 不一致**;全量 9284 行 sha 校验 **0 不一致**;`visual_tokens == 273×页数`、`n_pages == len(images)` 全对。
- vs `length_manifest.jsonl`(300 行,seed 5678):`target_tokens` **0 不一致**。manifest 共 10292 行 = 9284 train + 505 validation + 503 test,即全部 split 都在清单里,无样本被预算外淘汰。
- 3 个典型样本 roundtrip(`decode(encode(target))` 前 200 字符)全 True:

| 样本 | 页数 | target tokens | 内容形态 |
|---|---:|---:|---|
| `pmc_pmc10000488`(中位页数) | 10 | 11256 | 综述:署名区→分节正文→大表格结尾 |
| `pmc_pmc10538276`(最长 target) | 10 | 13487 | 论文:关键词→正文→参考文献列表结尾 |
| `readoc_arxiv_1805.05913` | 6 | 5444 | arXiv 论文含公式/引用 |

样本开头/中间/结尾原文见 `data_audit_samples.txt`(开头=署名/标题区,中间=正文段落,结尾=表格或参考文献,均为连贯 Markdown,无拼接痕迹)。

---

## 4. packing / 拼接 / label mask 核查

| 检查项 | 结果 | 依据 |
|---|---|---|
| packing | `--packing false` | `_run_train.sh:217` |
| sample concat / padding_free | `--padding_free false`、`per_device_train_batch_size 1` → batch 内不拼接序列 | `_run_train.sh:216,224` |
| 不同 PDF 的 target 无 EOS 拼接 | 不可能发生(无 packing;每行独立编码,模板 suffix=1 EOS) | 同上 + §3.4 |
| label mask | 原生模板构造:prompt/图段 −100,response+EOS 监督;title 系模板另有运行时对齐断言(`locate_response_slice` 校验 response_ids 与 labels 逐位相等,suffix 必须==[eos]) | `core.py:200-260`;两轮 full_ce 用原生路径 |
| 训练前数据门禁 | `validate_reviewed_jsonl.py` + `validate_training_recipe.py` 在 swift 启动前强制执行(channel/prompt/sha/heading 计数) | `_run_train.sh:186-199` |
| 陈旧缓存 | `--load_from_cache_file false` | `_run_train.sh:209` |
| LoRA 挂载位置 | guard callback 在 step 1 前校验:恰好 84 个模块、attn48/dense_mlp3/shared_experts33/routed0,越界即报错 | `plugin/uocr_title_mask.py` UOCRLoRATargetGuard |

**结论:数据管道无结构性 bug 证据;不存在"两个 PDF 的 target 无 EOS 拼在一起"的可能路径。**

---

## 5. LoRA 配置与训练强度

| 参数 | 值 | 备注 |
|---|---|---|
| rank / alpha | 16 / 32(有效 scale 2.0) | 与 0827 readoc-view-16k-full-ce 参考实验**写死一致**(`run_pmc_fullce.sh` 头注释;fc 路由由同一 `_run_train.sh` 训练) |
| dropout | 0.05 | |
| lr / 调度 | 2e-5,cosine,warmup 3% | 非 legacy 配方;legacy_20260823 才是 1e-4/0.05 |
| epoch / 步数 | 1 epoch;pmc 1160 步、mix 1623 步、fc 参考 213 步 | **步数逐轮上升** |
| 全局 batch | 8(1 GPU × bs1 × accum8) | |
| 冻结 | vit/aligner 冻结;LLM routed experts 不挂 LoRA(=冻结);仅 84 个 attn/MLP 模块可训 | guard 实测 |
| 期末状态 | pmc:train_loss 0.0725、eval_loss 0.0546、eval_token_acc 0.987;mix:train_loss 末10 0.0589、eval_loss 0.0521 | 无 loss 爆炸/塌缩形态 |

**判断**:超参本身温和且三路一致,"LoRA 更新过强破坏基座分布"缺乏形态学证据(无 loss 崩坏、eval_loss 持续下降、eval_token_acc 0.987)。但**未训练 base 本身就会复读**(0827 同解码下重复行≥10% 的文档 base 9/129、fc 7/129、tw 10/129;四规则严格退化 base 3 篇),训练后严格退化升到 9 篇——训练相关,但更可能是"merge 长文档任务 + 步数增多(213→1160→1623)放大了固有倾向",而非超参强度异常;与两轮复核发现的"步数 vs 配方混淆"一致,尚未分离。

---

## 6. 训练 vs 推理:窗口与解码参数对照

| 参数 | 训练 | 推理(129 评测) | 官方默认 | 一致? |
|---|---|---|---|---|
| R-SWA 答案窗口 | 128(训练 mask) | 128(ring buffer) | 128 | ✅ |
| 前缀可见性 | 全部页图+prompt 全可见 | prefill 全保留 | 同 | ✅ |
| 解码 | — | greedy(temperature=0) | greedy | ✅ |
| no_repeat_ngram_size | — | 35 + 自定义滑窗处理器 | **0(无防重)** | ❌ 项目新增 |
| ngram_window | — | 单页 128 / 多页 1024 | 0 | 项目新增 |
| max_length | 16384(序列上限,靠构建过滤) | 32768(两轮)/ **20480(0827 基线)** | 32768 | ⚠️ 跨轮不一致 |
| 视觉 | 273 token/页、crop 关、image_size 1024 | 同 | infer_multi 默认 640(111 token/页) | 项目与训练一致,官方 demo 默认不同 |
| 页边界表示 | 273-token 块 + 1 分隔 token,无文本标记 | 同 | 同 | ✅ |

### ngram 处理器为什么挡不住复读(实现语义)

`SlidingWindowNoRepeatNgramProcessor`(`modeling_unlimitedocr.py:354-384`):在**最近 `window` 个 token** 内寻找与"当前 (n−1)-gram 前缀"相同的 n-gram,把它的延续 token 分数置 −inf。因此:

1. **只防精确重复**。周期循环的唯一续写被禁后,greedy 取次优 token → 产生变异后循环继续:观测到 `TEL: 0755-822-8228`×4 后变 `8229`、"### 210、分部信息"变"211、212…"(计数器递增)——变异体与计数器恰恰是处理器的"挤压"产物形态。
2. **窗口外不设防**:重复子串跨度 >1024(多页档)时完全不管;而"最长重复单元"实测 115c~32k。
3. **官方本来就没有它**(`no_repeat_ngram_size=0`):它只能减少"精确复读",不能阻止"近似复读/计数器复读",两者在四类循环 case 中都存在。

---

## 7. 复读 case 首偏离解剖(6 case + base 交叉)

口径:严格退化名单与路径**逐字复用** `docs/review_20260918/verify1.py` 的四规则实现(dup≥0.3 / len≥2×GT / 塌缩≤1/3 / zlib<0.15)。k_strict = pred 与 GT 的 token 公共前缀;k_eff = 先对齐 GT 开头、再算 pred 跟随 GT 的 token 数;**k_eff=0 = 输出从第 0 个 token 就没跟 GT**。报告全文:`evaluation/audit_repeat_20260918/case_forensics.md`(+json)。

### 7.1 严格退化名单(复算,与既有记录一致)

- **pmc 9 篇**:2-1横向合并(dup0.33,len4.3x)、3-2跨页分页切断(len5.0x)、4-1统计图1(len2.5x)、4-3扫描2(dup0.37)、5-1双栏表格2(dup0.92)、5-1双栏表格3(len4.4x)、5-2表旁水印(dup0.33)、中车株洲_方案技术_6(len129.8x)、珠海机场_三防工作预案_23(len2.6x)
- **mix 9 篇**:3-1跨页重复表头(dup0.69)、3-2跨页分页切断(len6.4x)、4-1统计图2(len4.0x)、4-3扫描2(dup0.52)、5-1双栏表格2(len2.3x)、中车株洲_方案技术_16(len43.3x)、广东城规院_惠玩甘青_11(len6.3x)、广东特检院_安全报告2_6(len22.2x)、珠海机场_工作证管理细则_6(dup0.98,len118x)
- **base 3 篇**(`3-2跨页分页切断` len2.0x、`珠海机场_三防工作预案_23` len12.3x、`蓝德_办公电脑配置标准_2` dup0.30);**fc 2 篇**。base 有逐篇 pred(129 篇),可定位。

### 7.2 case 总表

| case | 路由 | k_eff | 偏离点位置 | pred 内 `<PAGE>` | 循环形态 |
|---|---|---:|---|---|---|
| 珠海机场_工作证管理细则_6 | mix | 0 | 输出起点 | 0 | 67 tok 页脚单元 ×123(全输出即 TEL 循环,28502 tok) |
| 4-1统计图2 | mix | 0 | 输出起点 | 0 | 年份轴漂移循环(无精确周期,LRS 2703×2) |
| 中车株洲_方案技术_16 | mix | 1 | 第 0 token(`<table`) | 0 | 表格计数器漂移 32k tok(LRS 16×4) |
| 3-1跨页重复表头 | mix | 0 | 输出起点(含格式错位) | 0 | 封面块重复(dup 0.69,LRS 69×2) |
| 2-1横向合并 | pmc | 86 | 中段:GT 即将进入 `<table>` 处 | 1(距偏离点 86 tok) | "### 210、分部信息"递增标题循环(LRS 5537×2) |
| 中车株洲_方案技术_6 | pmc | 0 | 输出起点(GT 以图链开头) | 0 | 1800 tok 递增数字串 ×17.7(3.2 万 tok,占输出 98%) |

### 7.3 输出原文摘录(偏离点/输出开头 vs GT 期望)

**珠海机场_工作证管理细则_6(mix)**:GT 是 445 字符的机房临时工作证细则;pred 28502 token,从第 0 token 起就是页脚电话循环,且**末位数字阶梯递变**:

```text
# pred 开头(循环+变异阶梯)
TEL: 0755-822-8228
TEL: 0755-822-8228
TEL: 0755-822-8228
TEL: 0755-822-8229
TEL: 0755-822-8229
TEL: 0755-822-8230
TEL: 0755-822-8230
TEL: 0755-822-8240

# GT 期望(GT[0:96])
# 二、 总则
为规范管理临时工作人员进入机房办公区域,方便各协作单位在办公区域内有效、有序、顺畅的开展工作。……
```

**4-1统计图2(mix)**:GT 是《北京市东城区 2023 年国民经济和社会发展统计公报》;pred 从第 0 token 起输出图表横轴标签并漂移循环:

```text
# pred 开头
_ 2017 年_ 2018 年_ 2019 年_ 2020 年_ 2021 年_ 2022 年_ 2023 年_
 2018 年_ 2019 年_ 2020 年_ 2021 年_ 2022 年_ 2023 年的_
 2018 年_ 2019 年_ 2020

# GT 期望(GT[0:96])
# 北京市东城区 2023 年国民经济和社会发展统计公报
2023 年,东城区坚持以习近平新时代中国特色社会主义思想为指导,……
```

**3-1跨页重复表头(mix)**:pred 输出国标封面套话(内容方向正确但插入分类号行、标题层级不同),随后整块重复(dup 0.69):

```text
# pred 开头(注意"# 数据基础设施…"块出现两次)
CBS 35.240
CCS L 70
# 中华人民共和国国家标准
GB/T XXXXX—XXXX
# 数据基础设施 数据目录描述要求
Data infrastructure—Data catalog description requirements
(点击此处添加与国际标准一致性程度的标识)
(征求意见稿)
在提交反馈意见时,请将您知道的相关专利连同支持性文件一并附上。
# 数据基础设施 数据目录描述要求
Data infrastructure—Data catalog description requirements
……(同一块再来一遍)
```

**中车株洲_方案技术_16(mix)**:第 0 token 即 `<table`,随后 32k token 表格单元计数器漂移(`k_strict=1`,字符公共前缀 7 = `<table>`):

```text
# pred 偏离前上下文(仅 1 token)
<table

# GT 期望(GT[0:97])
<table>
<tr><td rowspan="2" colspan="2">仿真参数</td><td colspan="4">仿真结果</td></tr>
……

# 循环内最长重复子串(LRS 16 tok × 4)
</td><td colspan="2">IGBT最大结温(°C)
```

**2-1横向合并(pmc,唯一中段偏离)**:pred 先自发 `<PAGE>`+页眉,跟随 GT 86 token(对齐后 64 tok 内容),偏离点恰在 **GT 即将进入表格**处,随即进入递增标题循环:

```text
# pred 偏离前 128 tok(含其自发的 <PAGE>)
<PAGE>宁波市天普橡胶科技股份有限公司
招股说明书
的准则自 2017 年 6 月 12 日起施行,对于 2017 年 1 月 1 日存在的政府补助,要求采用未来适用法处理;……

# GT 期望(偏离点之后紧跟表格)
……要求按照修订后的准则进行调整。
财政部于 2017 年度发布了《财政部关于修订印发一般企业财务报表格式的通知》,……
本公司执行上述规定的主要影响如下:
<table><tr><td>会计政策变更的内容和原因</td><td>受影响的报表项目名称和金额</td></tr>……

# pred 偏离后的循环(LRS 5537 tok × 2)
、分部信息
### 210、分部信息
### 211、分部信息
### 212、分部信息
```

**中车株洲_方案技术_6(pmc)**:GT 以图链 `![](images/aipaas.jpg)` 开头(pred 从不输出图链,属格式错位),pred 内容方向正确但 3.2 万 token 的 98% 是 1800-token 递增数字串 ×17.7:

```text
# pred 开头
四方氢能源城际(GSYE51)牵引变压器(永磁+SiC)
图 2 牵引变流器主电路图
4.2 内部接口规划
牵引变流器内部器件多采用模块化设计,……(此后进入递增数字循环)

# GT 期望(GT[0:96])
![](images/aipaas.jpg)
图 2 牵引变流器主电路图
# 4.2 内部接口规划
牵引变流器内部器件多采用模块化设计,……
```

**base 交叉(未训练)**:`3-2跨页分页切断` base 同样第 0 token 偏离并复读(len 2.0x),开头自发 `<PAGE>`+样本说明块循环(LRS 55×2)——同一病态在无 LoRA 的 base 上已存在。

### 7.4 清单四问的回答

| 问题 | 回答 |
|---|---|
| 偏离前刚跨页? | **否**。5/6 篇 pred 从头就没有 `<PAGE>`;唯一有标记的 2-1横向合并,偏离点在其第 1 个 `<PAGE>` 后仅 86 tok |
| 偏离前刚进标题/列表/表格? | 2/6 与表格边界相关:中车株洲_16 第 0 token 即 `<table`;2-1横向合并 偏离点恰在 GT 进入 `<table>` 处。其余为输出起点的普通段落 |
| 距上一个 `<page>` 很远? | pred 普遍 **0 个** `<PAGE>`(无标记可言);有标记的那篇距离 86 tok(很近) |
| 偏离前最近 128 tok 有强重复? | **没有**:偏离前窗口 8-gram 最大重复 5 篇=0、1 篇=2。重复模式是偏离的**结果**,不是前兆 |

### 7.5 机理小结

复读的主形态不是"写到一半掉进循环",而是**起步即偏**:对低信息量/特殊版式首页(工作证页脚、统计图轴标签、标准封面、表格框架),模型从第 0 token 就输出高频模板内容而非 GT;随后循环在 128-token 环形窗口内自我延续,精确重复被 ngram 处理器禁断后,以变异阶梯(8228→8229→8230→8240)与计数器递增(210→211→212)的形式继续(§6 机制与该形态相容,不能唯一归因)。唯一的中段偏离恰发生在 GT 进入表格处——表格边界是另一个高风险点(样本量 1,仅登记不外推)。另注:2 篇 k=0 案例含格式错位成分(图链/标题层级/分类号行),其实质失败仍是循环本身。

---

## 8. 原因排序与证据链

**1)R-SWA 128-token 答案窗口的固有自循环(主因,架构固有)**
- base 未训练即复读:0827 重复行≥10% 文档数 base 9/129;四规则严格退化 base 3 篇(与 pmc/mix 同口径检测);case 交叉:`3-2跨页分页切断` base 同样第 0 token 起偏离复读。
- case 解剖:复读主形态是**起步即偏**(6 篇中多数第 0 token 即偏离 GT,偏离前窗口无重复模式),随后循环在窗口内自我延续;推理时模型只看"全部页图+prompt+最近 128 个已生成 token",一旦进入周期模式(周期 ≪128),更早的已写内容被环形覆盖,模型无法"看到自己已写过"。训练期答案窗口同为 128,模型从未学过"回看更远的历史输出"。
- 循环取证(前轮复核):四类循环单元 115c~32k,其中计数器族每次 ngram 都是新的;训练数据 0 命中;GT 侧无自重复 → 解码期行为。

**2)解码侧放大(官方无防重;项目防重只覆盖精确重复)**
- 官方默认 `no_repeat_ngram_size=0`;项目加的处理器(35, 128/1024)对计数器/近似重复无效,且"禁精确→变异继续"的机制与观测到的 8228→8229、210→211 形态吻合。
- 0827 基线 max_length=20480,两轮 32768:生成预算变大后循环可跑更长,检出概率上升——**跨轮比较退化篇数必须带此口径**。

**3)训练放大(与步数混淆,非超参过强)**
- 超参三路一致、routed experts 冻结、1 epoch、无 loss 崩坏;但严格退化 base 3 → pmc 9 → mix 9,步数 213 → 1160 → 1623 同向。"任务分布(长文档连贯输出)+更多步数"放大固有倾向的假说与"配方问题"假说仍混淆——32k 续跑诊断臂是现成分离实验(前轮复核建议,维持)。

**4)数据/截断/EOS(基本排除)**
- 0 截断、0 丢弃(步数反推)、每样本恰 1 EOS、GT 无段落级复制(char 加权行重复≤0.12%)、循环内容训练数据 0 命中、全量 sha 对账一致。

**5)page boundary 丢失(不成立)**
- 词表无 page token,官方多页即"visual 块+分隔 token",训练/评测一致;不存在"去掉了 page boundary"这一改动。真正的"页边界信息"只有每 273 token 一个分隔 token,这是官方设计,base 同样如此。

**6)其他代码 bug(未发现)**
- packing/拼接/label mask/EOS/LoRA 挂载/缓存均核查通过(§4);swift patch 仅 6 行且不作用于 full_ce 损失路径。

---

## 9. 证据缺口与口径备注

1. R-SWA 自循环是**机制推断 + base 复读事实**的支持,未做 attention/上下文消融取证;若要直接证明,需旁路推理改窗口对照(属 GPU 实验,本次未做)。
2. 0827 base/fc 的 responses.jsonl 无逐行 gen 记录(旧格式),`max_length=20480` 来自当时文档记载;ngram 35/128-1024 按当时服务默认与文档推定。
3. §7 case 数量为 6 篇(清单要求≥5),全部来自既有 129 评测 pred_clean,未重跑推理。
4. 训练端"所见数据=原始数据"以**构建清单对账 + sha 全量校验 + 步数反推 + 模板 suffix 断言**为准;未逐 token 复现 trainer 编码(ms-swift 框架按可信处理)。
5. "严格退化篇数"跨轮对比(base/fc 的 pred 来自 0827,max_length 20480)与"重复行≥10%"诊断口径(10% 阈值)是两个不同口径,本文均已分开标注,未混用。

## 附录:产物与脚本

| 文件 | 内容 |
|---|---|
| `evaluation/audit_repeat_20260918/data_audit_report.json` | 三套训练集全量 token 化统计(行数/页数/长度/截断/EOS/page/GT n-gram/对账) |
| `evaluation/audit_repeat_20260918/data_audit_samples.txt` | 3 个典型 merge 样本的 prompt/页数/头中尾原文/末 12 token id |
| `evaluation/audit_repeat_20260918/uocr_audit.py` | 审计脚本(远端 venv python CPU 运行) |
| `evaluation/audit_repeat_20260918/case_forensics.{md,json}` | 复读 case 首偏离取证(§7) |
| 远端镜像 | posttrain `docs/repeat_audit_2026-09-18.md`(+脚本) |
