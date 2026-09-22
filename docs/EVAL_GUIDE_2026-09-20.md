# 129 评测指南(runbook)— 训完模型后怎么评

> 2026-09-20。**合并自 `PMC_EVAL_GUIDE_2026-09-17.md` + `evaluation_pitfalls_2026-09-17.md`**(两份已删),并纳入 09-18 pmc-readoc-spage-16k 评测轮的实践(一键脚本、AGGRESSIVE 白名单、straggler 容错、口径洞)。
> 定位:**拿到新 checkpoint 后,从权重到 Overall 修正分的完整操作手册 + 全部坑**。新会话/新人跑评测前必读。
> 远端权威副本:posttrain `docs/EVAL_GUIDE_2026-09-20.md`。

## 0. 链路总览(20 秒版)

```
checkpoint(output/<run>/v0-*/v0-*/checkpoint-*)
  → P1 起 4 实例推理服务(GPU1)→ probe 129 推理(prompt 按页数路由)→ responses.jsonl
  → P2 pred/ ← responses.jsonl 重建 → 剥壳 pred_clean(--check 必须 rc=0)→ 重建标题 pred_titles
  → P3 逐篇 sweep(swpN_,4 并发,600s 超时)→ P4 聚合+合成 official 件
  → P5 AgentBuilder(Overall 原值)→ P6 title S5(87 篇)+ corrected_overall splice + 退化四板斧
  → 落账:RUNS_REGISTRY.md + 周报(主结果)/支线文档(支线)
```

**快捷路径**:复制 `ms_swift_title_mask/scripts/run_pmc_readoc_spage_eval.sh`,改 3 处:`RUN_NAME`、`PREFIX`(swp 轮次号)、`AGGRESSIVE_FILES`(必须重新人工复核,见坑14),然后 `bash <脚本>.sh --skip-wait`(训练已完)或裸跑(等训完自动开评)。断点续跑:`--skip-inference` 复用已有 responses.jsonl。

## 一、协议契约(勿改;改任何一项即与历史基线不可比)

- **prompt 按页数路由**(`probe_evaluation_pdfs_parallel.py:114-117`):单页 `<image>document parsing.`;多页 trained 模型 `<image>Multi page merge.`(**base 基线用 `Multi page parsing.`,不得混用**——SOP §4.1)
- **解码**:dpi 144、max_length 32768、no_repeat_ngram_size 35、ngram_window 单页 128/多页 1024、temperature 0
- **固定集**:129 PDF @ `/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/source`;GT @ 同目录 `groundtruth`(130 份 md;`Handwriting document.md` 无预测属正常,零样本不进平均)
- **推理走 adapter 挂载**(与全部历史路线同一推理代码路径;merged 权重是部署件,数学等价但协议优先)
- **结果目录结构**(与 `readoc-view-16k_all129/` 同构):
  - `responses.jsonl` = 对账主档(请求+响应原文+gen 参数)
  - `<pdf名>__unlimited-ocr-full-ce.md` = 展示文件(带报告头,**不进评分**)
  - `pred/<run>/<pdf名>.md` = 原始 grounding 输出(仅对账留档,**不进评分**)
  - `pred_clean/<run>/<pdf名>.md` = 剥壳后 → **同时喂 OmniDocBench 和 AgentBuilder**
  - `pred_titles/<run>/` = 标题重建件 → **只喂 title 指标**,不得替代 pred_clean
  - pred 必须从 `responses.jsonl` 的 `response.text` 生成,不是从展示文件拷

## 二、手动全流程(6 步;一键脚本内部就是这些)

```
GT   = /home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth
PDFS = /home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/source
OMNI = /home/jovyan/hyx/pdf2text-auto-label-eval/OmniDocBench_v1.5   # CWD 必须在这里跑
UOCR = ~/hyx/uocr-ms-swift-title-mask
VENV = $UOCR/evaluation/tooling/omni-eval-venv/bin/python
EVAL = $UOCR/evaluation/<run>_all129          # 本轮产物根目录
S    = ~/hyx/unlimited-ocr-posttrain/ms_swift_title_mask/scripts
```

### P1 推理 129(占 GPU1;先查共享与僵尸)

1. GPU 约定:GPU0 常驻 zxr 系 vllm **勿动**;GPU1 用前 `nvidia-smi` 确认空闲,用完即还。
2. 接手先查僵尸:`pgrep -af run_md2md.py`(上场遗留 131 个已清,历史教训见坑1)。
3. 起 4 实例服务(transformers 串行全局锁,提速靠多实例;vLLM 需自研架构注册,未做):
   ```bash
   for PORT in 18191 18192 18193 18194; do
     MODEL_PATH=/home/jovyan/hyx/models/Unlimited-OCR \
     FULL_CE_ADAPTER_PATH=<checkpoint> TITLE_WEIGHTED_ADAPTER_PATH=<checkpoint> \
     SERVICE_PORT=$PORT SERVICE_DEVICE=cuda:0 \
     nohup bash $S/run_inference_service.sh > $EVAL/service_$PORT.log 2>&1 < /dev/null &
   done
   # 轮询 http://127.0.0.1:$PORT/health 直到 ok(一键脚本等 60×20s)
   ```
4. probe 推理(**必须 `python -u`**,否则缓冲让 responses.jsonl 一直 0 行像挂了):
   ```bash
   cd $S && "$VENV" -u probe_evaluation_pdfs_parallel.py \
     --pdf-root $PDFS --gt-root $GT --output-dir $EVAL \
     --model unlimited-ocr-full-ce --all-files --dpi 144 --max-length 32768 \
     --page-cache-dir $EVAL/page-cache \
     --url http://127.0.0.1:18191/infer --url ...:18192/infer --url ...:18193/infer --url ...:18194/infer
   ```
   验收:`wc -l responses.jsonl` ≥ 128(129 零失败是常态);完事 kill 服务进程。

### P2 剥壳 + 重建标题(评分输入纯净性铁律)

**铁律:喂分输入必须是纯文本,零协议 special token(`<|...|>`)。** 历史事故(坑9):raw grounding 直接喂分,壳文字渗进 norm_pred,TextEdit 污染到 0.5679(正确 0.2342)、Total 假跌 48.29→37.35,该轮全部产物删除废弃。

协议 token 四类结构与处理(权威实现 `strip_grounding_shell.py`):

| 结构 | 例子 | 处理 |
|---|---|---|
| 闭合块 | `<\|det\|>title [270,125,776,189]<\|/det\|>` | **删**(块内纯标签+坐标;正文在闭合标记后,保留) |
| 孤儿开标签 | `<\|det\|>text [101,749,895,822]<\|det\|>正文` | 保守模式不删,告警 |
| 坐标截断 | `<\|det\|>header [786, 52,`(无闭合) | 保守模式不删,告警 |
| 闭合块内嵌正文 | `<\|det\|>table [c] …<table>…<\|/det\|>` | 保守模式不删,告警 |

设计原则(09-17 拍板):**尽量保守**——畸形结构可能是模型输出问题,预处理不得擅自解释;逐文件告警 + **拒绝写盘**,人工复核确认是模型畸形输出后才 `--aggressive` 显式清理(只删 det 开标签 + 已知标签白名单 + 可截断坐标括号;**正文永不删**)。幂等。`<PAGE>` 默认保留(见四-3)。

```bash
# 1) pred 重建(从 responses.jsonl)+ 保守剥壳(全部干净才落盘,否则 ABORT rc=1)
python3 $S/strip_grounding_shell.py --in-dir $EVAL/pred/<run> --out-dir $EVAL/pred_clean/<run>
# 2) ABORT 文件人工复核后,对确认文件显式清理(白名单机制见坑14)
python3 $S/strip_grounding_shell.py --aggressive --in-dir ... --out-dir ...
# 3) 验收:任何准备喂分的目录必须 check 通过
python3 $S/strip_grounding_shell.py --check --in-dir $EVAL/pred_clean/<run>   # rc=0 才能进评分
# 4) 标题重建:壳+bbox 剥除、det title 文本重建为 # 行(grounding 无层级 → 统一一级)
python3 $S/strip_grounding_shell.py --rebuild-titles --in-dir $EVAL/pred/<run> --out-dir $EVAL/pred_titles/<run>
```

### P3 逐篇 sweep(CPU,不占卡)

**唯一可靠路径 = 逐文档 sweep + 自聚合**(全量 md2md 在 jovyan 有病态聚合段,见坑1)。
新建本轮 sweep 目录 `$EVAL/sweep/`(可从上一轮拷 `resweep_pred_clean.sh` 改 `swpN_` 前缀,N 递增:swp2=pmc 轮、swp3=mix 轮),对每篇建单文件 tmp gt/pred 目录后:

```bash
cd $OMNI && $VENV run_md2md.py --gt_dir <tmp_gt> --pred_dir <tmp_pred> \
  --save_name swpN_<文档名>
```

4 并发 xargs,每篇 600s 超时,逐篇 rc 记录 results.txt。每份产出 `result/swpN_<文档名>_{metric_result,text_block_per_page_edit,table_per_table_TEDS,display_formula_per_page_edit,...}.json`。**straggler(病态复读篇 matching 不收敛)按坑12 容错出分**,缺口有界时先出总分再慢磨补。

### P4 聚合 + 合成 official 件

```bash
cd $S && python3 aggregate_sweep.py "swpN_*"      # ⚠️ 前缀必须带通配符(坑10)
python3 synthesize_combined_metric.py "swpN_" $EVAL/omnidocbench_sweep_combined <run>
```

- **公式**(`task/end2end_run_eval.py::_calculate_total_score`):`Total = ((1−TextEdit)×100 + TEDS×100 + (1−FormulaEdit)×100)/3`;TextEdit/FormulaEdit = 全局所有页平均(doc 各贡献 1 个篇内均值),**TEDS = 全局所有表平均(pooled)**。
- 聚合器可信判据:对 0827 旧基线精确复现 `44.13970499986459`。
- sanity:pages≈116(另 12 篇表格型无文本块属正常,坑11)/ formula_pages=3;**tables=本轮计分表数,必须记进报告**(见四-2)。
- synthesize 产出 `<run>_metric_result.json` + 4 个 `*_result.json` 样本拼接件;验证模式 `... <旧前缀> -` 可对旧基线逐位核对。

### P5 AgentBuilder(Overall 原值)

```bash
cd $UOCR/evaluation/tooling/agentbuilder_pkg && "$VENV" -m agentbuilder_eval.run_eval \
  --gt-dir $GT --pred-dir $EVAL/pred_clean/<run> \
  --raw-metric $EVAL/omnidocbench_sweep_combined/<run>_metric_result.json \
  --output-dir $EVAL/agentbuilder/<run> --name <run>-remote --allow-missing
```

- `--raw-metric` 只读 6 个字段(text/table/formula/order 的 ALL_page_avg + TEDS.all + TEDS_structure_only.all);**Overall = 0.3·text_acc + 0.3·TEDS + 0.3·order_acc + 0.1·title(此 title 为原始值,通常≈0,见 P6)**。公式项不在 Overall。
- 可选 `--formula-cdm`(需 GPU,历史各轮均跳过,不影响 Overall);`--allow-missing` 处理缺预测。

### P6 title 修正 + 退化四板斧(Overall 修正分在这里)

1. **title S5(87 篇)**:用 `tooling/agentbuilder_pkg` 的 `title_metric.score_markdown_pair` 对 `pred_titles/<run>/` 评分。分母集 = 87 篇 GT 标题文档(88 − Handwriting),S5 = 0.3 内容 + 0.7 级别(与 Overall 原 title 权重定义一致)。**引用一律用 87 篇标准化值;`summary.json` 的全 GT 集均值(90-93 篇,pred 有标题 GT 无记 0)各路人口不等,不可跨路比。**
2. **corrected_overall splice**:`Overall_修正 = Overall_原值 + 0.1 × (title_S5_新 − title_raw)`;分数件 `corrected_overall.json`。历史各轮 splice 数学合法性已验证(同 87 篇人口)。注意 rebuild 件全 `#` 一级 → 级别分退化为 GT 顶层占比(四路同、绝对值偏低),勿与未来真层级输出模型直接比。
3. **退化四板斧**(pred_clean vs GT):行 dup≥0.3 / 输出长度 ≥2x 或 ≤1/3 / zlib<0.15(行内循环)+ 塌缩 ≤1/3;另建议监控 det-title 密度与计分表覆盖率("第五模式:长 merge 结构简化",四板斧盲)。
4. 一键脚本 P6 会自动完成以上并写支线评测文档(posttrain `docs/`)。

### 落账

`~/hyx/uocr-ms-swift-title-mask/RUNS_REGISTRY.md`(每轮必登记);主结果进周报 `~/hyx/unlimited-ocr-posttrain/docs/posttrain_weekly_report_*.md`,支线实验只写支线文档不合周报。

## 三、口径与结果解读纪律

1. **Overall 与 Total 是两个分,方向可能相反**(pmc 轮:Total +4.15pp vs readoc-fc,Overall −3.01pp),引用必须写明是哪个。Overall = 0.3/0.3/0.3/0.1;Total 里公式项占 1/3。
2. **pooled TEDS = 全部配对表摊平求平均(表加权)**,与 text/order 的 doc 加权不同:表多的篇权重大。**两个口径洞**:①池 = 配上对的 (GT表, pred表),模型没发射的 GT 表从分母消失,"不发表"不直接扣分 → **跨路比 TEDS 必须同时报计分表数**(pmc 194 → mix 149);②AgentBuilder per-doc 在 table_teds=None 时按 /0.7 重归一。未来"健康子集"比较优先用 union 同文档集口径(review_20260918 §4-C)。
3. **`<PAGE>` 页标记**:GT 0/130;pred_clean 对称保留(0827 契约)。`--strip-page-markers` 可删但删后不能与历史基线对列;改动须两边同做并整体重评。实测其占多页 pred ≤0.46% 且 base 发最多多页 text 反而最好,与缺口无关,勿为此重评。
4. **per-doc 检查三板斧**:①重复行比例(dup≥0.3 即病态);②输出长度 vs GT(≥2x 或 ≤1/3 看原文);③pred_clean 抽样直接读。掉分集中度先算再下结论(剔退化文档后的均值差才是"广谱漂移"成分);退化统计看中位数/截尾,均值会被少数灾难篇绑架。
5. **跨 run 对比前**:prompt/dpi/max_length/剥壳口径/评测集逐项核对一致;口径不同的列(如老口径 title 诊断分 0.7630)不可直接对列。
6. GT 自身可能有的问题先排查(科欣环保_产业结构_5 四路全 collapse → 训练前已存在)。

## 四、坑清单(13 条历史 + 3 条新增,勿重复)

1. **全量 md2md 在 jovyan 有病态聚合段**:matching 1.5 分钟跑完后单 worker 磨 60 分钟+ 不收敛(旧团队 `OmniDocBench failed` 同因)。唯一可靠路径 = 逐文档 sweep + 自聚合;接手先 `pgrep -af run_md2md.py` 清僵尸。
2. TEDS 慢与表格大小/相似度无关(单文档 9-74s 都快),是全量管线的聚合/并行实现问题。
3. probe 必须 `python -u`,否则缓冲让 responses.jsonl 一直 0 行像挂了。
4. `pkill -f` / `kill -9 $(pgrep -f ...)` 会匹配 ssh 自身命令行自杀——用 `pgrep -f` + `ps -o comm=` 过滤 python 再 kill。
5. ms-swift checkpoint 在双层 `v0-<ts>/v0-<ts>/` 里,`find -maxdepth 2` 会漏;一键脚本用 `ls -td` 取最新。
6. 评测走 adapter 挂载(与历史路线同一推理代码路径),merged 权重是部署件;协议优先。
7. 推理服务 transformers 单请求串行(全局锁),提速靠 4 实例(80G 卡 4×~7G 余量充足)+ probe 多 `--url`;vLLM 未做(需自研 DeepSeek-V2+视觉架构注册)。
8. GPU 共享:GPU0 常驻 zxr 系 vllm 勿动;GPU1 用前 `nvidia-smi` 确认,用完即还。
9. **【最重要】必须喂剥壳 pred_clean**:首轮喂 raw grounding,Total 48.29 假跌 37.35,全轮废弃。发现方式 = 抽旧 run `text_block_result.json` 样本原文核对输入形态;**别信文档文字,信样本**。
10. `aggregate_sweep.py` 前缀必须带通配符(`swp2_*`):`aggregate("swp2_")` 拼出双下划线 glob 到 0。聚合器对旧基线精确复现 44.13970499986459 才可信。
11. 表内文档缺 `*_per_page_edit.json` 是正常的:整篇表格型 GT 无文本块,`text_block_result.json = []` 不进平均,非配对失败。判断依据 = result 文件是否 `[]`。
12. **病态复读篇卡 matching**:`5-1双栏表格2` 输出 92% 重复行,match_quick 不收敛(600s×2 超时,疑似 O(N²·L)),与体量无关。pmc 轮 nohup 慢磨(极值边界 Total ∈ [48.01,48.38]);mix 轮一键脚本按缺口容错直接出分。先看 pred 重复行比例再怀疑管线。
13. `_gt_skip1/_skip2/_pred_skip*` 是弃用全量跑残留目录,非评分输入,可不理会。
14. **【新】一键脚本的 `AGGRESSIVE_FILES` 白名单是上一轮的人工复核记录,新轮必须重新复核后再改**。mix 轮 3 篇(经复核确认为畸形 grounding 脚手架、正文完好):`1-3单元格内换行2 / 4-2段落中带「图x-x」 / 5-1双栏表格1`。白名单为空而 conservative ABORT 时脚本会 FATAL——这是故意的,人工复核不可跳过。
15. **【新】评分两个口径洞**(见三-2):未发射 GT 表不进 TEDS 分母;per-doc 缺 TEDS 时 /0.7 重归一。引用跨路 TEDS 必带计分表数。
16. **【新】title 引用纪律**:87 篇标准化值为准;rebuild 全 `#` 一级导致级别分退化,绝对值偏低是口径属性;`<PAGE>`/`<table>` 发射计数随训练深度变化是格式层过拟合迹象,属数据配方杠杆,不为此重评历史分。

## 五、已评路线速查(勿重跑;工作约定:不重跑已完成评测,覆盖类操作先确认)

| 路线 | Overall(title 修正,splice) | OmniDocBench Total | 产物 |
|---|---:|---:|---|
| base | **0.7554** | 53.3645 | `evaluation/readoc-view-16k_all129/`(0827 老跑) |
| readoc-view full-ce | **0.7400** | 44.1397 | 同上 |
| readoc-view title-weighted | **0.7313** | 43.3568 | 同上 |
| pmc-fullce-16k(128/129) | **0.7092** | 48.2908 | `evaluation/pmc-fullce-16k_all129/` |
| pmc-readoc-spage-16k | **0.6709** | —(TextEdit 0.2473/TEDS 0.3908) | `evaluation/pmc-readoc-spage-16k_all129/` |

- 明细:`agentbuilder/<route>/metrics.json + per_doc_scores.json`;pred_clean 在同目录 `pred_clean/<route>/`;OmniDocBench pooled 重算件在 `$OMNI/result/`(swp2_*=pmc、swp3_*=mix)。
- 上表 Overall 均为 title 修正后;metrics.json 里 raw title 各路 ≈0 是剥壳口径伪像(四路真实 title 在 0.63-0.73 窄带,title87:0.6862/0.6668/0.6697/0.6150)。
- 0827 老 pred_clean 有 4 文件孤儿 det 残留(~231 字符,Total 影响 <0.01pp,方向 baseline 略被压)——登记在案,历史分数不改;若重评须两边同做。

## 六、脚本与产物索引

| 项 | 本地(handoff/evaluation/) | 远端 |
|---|---|---|
| 一键评测脚本 | — | `$S/run_pmc_readoc_spage_eval.sh`(复制改 RUN_NAME/PREFIX/白名单) |
| `strip_grounding_shell.py` | ✅ 镜像 | `$S/`(权威) |
| `aggregate_sweep.py` | ✅(已修前缀 bug) | `$S/` + 各轮 `$EVAL/sweep/` |
| `synthesize_combined_metric.py` | ✅ | `$S/` + 各轮 `$EVAL/sweep/` |
| `resweep_pred_clean.sh` | — | 各轮 `$EVAL/sweep/`(改前缀复用) |
| `probe_evaluation_pdfs_parallel.py` | — | `$S/`(prompt 路由在 :114-117) |
| 评分口径细解 + 两轮复核 | `TWO_ROUNDS_REVIEW_2026-09-18.md`、`review_20260918/verify{1-4}.py` | posttrain `docs/two_rounds_review_2026-09-18.md` |
