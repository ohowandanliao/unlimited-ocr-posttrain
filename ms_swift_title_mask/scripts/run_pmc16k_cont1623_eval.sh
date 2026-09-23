#!/usr/bin/env bash
# run_pmc16k_cont1623_eval.sh — PMC Full-CE 续训至 1623 步的诊断评测（129 全管线）
#
# 目的: 步数消融——同 PMC 数据,1160 vs 1623 步;并与同为 1623 步的 mix 路线对照。
# 由 run_pmc_readoc_spage_eval.sh 复制修改: RUN_NAME/PREFIX(swp6_)/AGGRESSIVE_FILES。
# GPU1 门放宽(外部常驻 ~36GB 空闲占用,余 45GB 够推理)/退化检测与 train_loss 门取 cont1623 自己的
# trainer_state(经 /tmp/cont1623_ckpt_run 符号链接传入 RUN_DIR)。
#
# 用法:
#   RUN_DIR=/tmp/cont1623_ckpt_run bash run_pmc16k_cont1623_eval.sh --skip-wait
#
# 管线: 起服务推理129→剥壳(pred_clean)+重建标题(pred_titles,均过--check)→逐篇sweep(swp6_,缺口容错)
#       →聚合→合成→AgentBuilder→title S5(87篇)+Overall splice+退化检测→写评测目录报告
# 约束: 只用 GPU1；不重跑历史评测；straggler(病态篇 matching 不收敛)按坑12容错出分。
set -uo pipefail

UOCR_ROOT=${UOCR_ROOT:-/home/jovyan/hyx/uocr-ms-swift-title-mask}
REPO=/home/jovyan/hyx/unlimited-ocr-posttrain
SCRIPTS=$REPO/ms_swift_title_mask/scripts
GT=/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth
PDF_ROOT=/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/source
OMNI=/home/jovyan/hyx/pdf2text-auto-label-eval/OmniDocBench_v1.5
VENV=$UOCR_ROOT/evaluation/tooling/omni-eval-venv/bin/python
RUN_NAME=pmc-fullce-16k-cont1623
EVAL=$UOCR_ROOT/evaluation/${RUN_NAME}_all129
PREFIX=swp6_
# 2026-09-21 修正: 4 服务 × ~10-17GB + 外部常驻 35GB → 首次推理 KV OOM。
# 改 2 服务(省 ~20GB,推理墙钟约翻倍),并加 expandable_segments 防碎片。
PORTS=(18191 18192)
STRIP=$SCRIPTS/strip_grounding_shell.py
LOG=$EVAL/oneclick.log
SKIP_WAIT=0
SKIP_INFERENCE=0
for _a in "$@"; do
  case "$_a" in
    --skip-wait) SKIP_WAIT=1 ;;
    --skip-inference) SKIP_INFERENCE=1; SKIP_WAIT=1 ;;
    *) echo "[oneclick] unknown arg: $_a"; exit 2 ;;
  esac
done

mkdir -p "$EVAL"
exec > >(tee -a "$LOG") 2>&1
echo "[oneclick] start $(date '+%F %T')"

# ---------- P0 等训练结束 + 定位 checkpoint ----------
if [[ $SKIP_INFERENCE -eq 1 ]]; then
  echo "[P0] skipped (--skip-inference): reuse existing responses.jsonl"
else
if [[ $SKIP_WAIT -eq 0 ]]; then
  echo "[P0] waiting for training to finish..."
  while pgrep -f "swift/cli/sft.py" > /dev/null; do sleep 120; done
  sleep 30
fi
RUN_DIR=${RUN_DIR:-$(ls -td $UOCR_ROOT/output/$RUN_NAME/v0-*/v0-* 2>/dev/null | head -1)}
[[ -z "$RUN_DIR" ]] && { echo "[P0] FATAL: no run dir under $UOCR_ROOT/output/$RUN_NAME"; exit 1; }
CKPT=$(ls -td "$RUN_DIR"/checkpoint-* 2>/dev/null | head -1)
[[ -z "$CKPT" ]] && { echo "[P0] FATAL: no checkpoint under $RUN_DIR"; exit 1; }
echo "[P0] RUN_DIR=$RUN_DIR CKPT=$CKPT"
# 2026-09-21: GPU1 有外部进程常驻占用 ~36GB(利用率 0%,评测期间持续存在);
# 余 ~45GB 满足推理(与历史共存情况一致),门改为 <20GB 可用才 FATAL。
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
(( GPU_USED > 60000 )) && { echo "[P0] FATAL: GPU1 busy (${GPU_USED}MiB)"; exit 1; }
fi

# ---------- P1 推理 129 ----------
if [[ $SKIP_INFERENCE -eq 0 ]]; then
PIDS=()
for i in "${!PORTS[@]}"; do
  PORT=${PORTS[$i]}
  MODEL_PATH=/home/jovyan/hyx/models/Unlimited-OCR \
  FULL_CE_ADAPTER_PATH="$CKPT" TITLE_WEIGHTED_ADAPTER_PATH="$CKPT" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SERVICE_PORT=$PORT SERVICE_DEVICE=cuda:0 \
  nohup bash "$SCRIPTS/run_inference_service.sh" > "$EVAL/service_$PORT.log" 2>&1 < /dev/null &
  PIDS+=($!)
done
echo "[P1] services launching on ${PORTS[*]} (pids ${PIDS[*]}), wait for health..."
for PORT in "${PORTS[@]}"; do
  ok=0
  for _ in $(seq 1 60); do
    if curl -s "http://127.0.0.1:$PORT/health" | grep -qi "ok\|healthy\|ready"; then ok=1; break; fi
    sleep 20
  done
  (( ok == 0 )) && { echo "[P1] FATAL: service :$PORT unhealthy; see $EVAL/service_$PORT.log"; exit 1; }
done
URLS=(); for PORT in "${PORTS[@]}"; do URLS+=(--url "http://127.0.0.1:$PORT/infer"); done
CACHE=$EVAL/page-cache
cd "$SCRIPTS" && "$VENV" probe_evaluation_pdfs_parallel.py \
  --pdf-root "$PDF_ROOT" --gt-root "$GT" --output-dir "$EVAL" \
  --model unlimited-ocr-full-ce --all-files --dpi 144 --max-length 32768 \
  --page-cache-dir "$CACHE" "${URLS[@]}" 2>&1 | tail -20
N_ROWS=$(wc -l < "$EVAL/responses.jsonl")
echo "[P1] responses rows=$N_ROWS/129"
kill "${PIDS[@]}" 2>/dev/null; sleep 10
(( N_ROWS < 128 )) && { echo "[P1] FATAL: <128 rows"; exit 1; }
fi

# ---------- P2 剥壳 + 重建标题（评分输入纯净性铁律）----------
python3 - <<'PYEOF'
import json, os
from pathlib import Path
ev = Path("/home/jovyan/hyx/uocr-ms-swift-title-mask/evaluation/pmc-fullce-16k-cont1623_all129")
out = ev / "pred" / "pmc-fullce-16k-cont1623"; out.mkdir(parents=True, exist_ok=True)
n = 0
for line in open(ev / "responses.jsonl", encoding="utf-8"):
    r = json.loads(line)
    stem = r["file"][:-4] if r["file"].endswith(".pdf") else r["file"]
    (out / (stem + ".md")).write_text(r["response"]["text"], encoding="utf-8")
    n += 1
print(f"[P2] wrote {n} pred files from responses.jsonl")
PYEOF
PC_DIR="$EVAL/pred/$RUN_NAME"
CLEAN_DIR="$EVAL/pred_clean/$RUN_NAME"
# 2026-09-21 人工复核记录：两篇各有未闭合 grounding 脚手架，正文完好；
# 白名单只对这两篇启用 --aggressive，正文永不删。
AGGRESSIVE_FILES=${AGGRESSIVE_FILES:-"1-3单元格内换行1.md 5-1双栏表格1.md"}
if python3 "$STRIP" --in-dir "$PC_DIR" --out-dir "$CLEAN_DIR"; then
  echo "[P2] conservative strip clean on first pass"
else
  if [[ -z "$AGGRESSIVE_FILES" ]]; then
    echo "[P2] FATAL: conservative strip ABORTed and AGGRESSIVE_FILES empty (畸形输出需人工复核)"; exit 1
  fi
  echo "[P2] conservative ABORT -> applying --aggressive to reviewed files: $AGGRESSIVE_FILES"
  TMPD=$(mktemp -d)
  for F in $AGGRESSIVE_FILES; do mv "$PC_DIR/$F" "$TMPD/" || exit 1; done
  python3 "$STRIP" --in-dir "$PC_DIR" --out-dir "$CLEAN_DIR" || exit 1
  python3 "$STRIP" --aggressive --in-dir "$TMPD" --out-dir "$CLEAN_DIR" || exit 1
  mv "$TMPD"/*.md "$PC_DIR/"; rm -rf "$TMPD"
  echo "[P2] aggressive pass done ($(echo $AGGRESSIVE_FILES | wc -w) files); pred/ restored"
fi
python3 "$STRIP" --check --in-dir "$EVAL/pred_clean/$RUN_NAME" || exit 1
python3 "$STRIP" --rebuild-titles --in-dir "$EVAL/pred/$RUN_NAME" --out-dir "$EVAL/pred_titles/$RUN_NAME" || exit 1
python3 "$STRIP" --check --in-dir "$EVAL/pred_titles/$RUN_NAME" || exit 1
echo "[P2] pred_clean + pred_titles ready and check-passed"

# ---------- P3 逐篇 sweep（swp6_，4 并发，600s 超时，缺口容错）----------
pgrep -af run_md2md.py | grep -v grep | awk '{print $1}' | while read -r p; do
  [[ "$(ps -o comm= -p "$p" 2>/dev/null)" == python* ]] && kill "$p" && echo "[P3] killed stale run_md2md pid $p"
done
WORK=/tmp/rswp6; rm -rf "$WORK"; mkdir -p "$WORK"
PC=$EVAL/pred_clean/$RUN_NAME
export GT PC VENV OMNI WORK
run_one() {
  local D="$1"; local gt="$WORK/gt_$D" pd="$WORK/pred_$D"
  rm -rf "$gt" "$pd"; mkdir -p "$gt" "$pd"
  if [ ! -f "$GT/$D.md" ] || [ ! -f "$PC/$D.md" ]; then
    echo "$D|rcMISSING_INPUT|$(date +%H:%M:%S)" >> "$WORK/results.txt"; return; fi
  cp "$GT/$D.md" "$gt/" && cp "$PC/$D.md" "$pd/"
  ( cd "$OMNI" && timeout 600 "$VENV" run_md2md.py --gt_dir "$gt" --pred_dir "$pd" \
      --save_name "swp6_$D" > "$WORK/log_$D.txt" 2>&1 )
  echo "$D|rc$?|$(date +%H:%M:%S)" >> "$WORK/results.txt"
  rm -rf "$gt" "$pd"
}
export -f run_one
sed 's/\.pdf$//' "$EVAL/evaluation_names.txt" | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "[P3] sweep done: $(grep -c rc0 "$WORK/results.txt") rc0 / $(grep -cE 'rc[1-9]|rcMISSING' "$WORK/results.txt") failed (失败篇见 $WORK/results.txt，按缺口有界口径出分)"

# ---------- P4 聚合 + 合成 ----------
cd "$SCRIPTS" && python3 aggregate_sweep.py --result-dir "$OMNI/result" "${PREFIX}*" 2>&1 | tee "$EVAL/aggregate.log"
python3 synthesize_combined_metric.py --result-dir "$OMNI/result" "${PREFIX}" "$EVAL/omnidocbench_sweep_combined" "$RUN_NAME" 2>&1 | tail -5

# ---------- P5 AgentBuilder ----------
(cd "$UOCR_ROOT/evaluation/tooling/agentbuilder_pkg" && "$VENV" -m agentbuilder_eval.run_eval \
  --gt-dir "$GT" --pred-dir "$PC" \
  --raw-metric "$EVAL/omnidocbench_sweep_combined/${RUN_NAME}_metric_result.json" \
  --output-dir "$EVAL/agentbuilder/$RUN_NAME" --name ${RUN_NAME}-remote --allow-missing) 2>&1 | tail -5
echo "[P5] agentbuilder done"

# ---------- P6 title S5 splice + 退化四板斧 + 支线文档 ----------
"$VENV" - <<'PYEOF'
import json, re, zlib, unicodedata
from pathlib import Path
U = Path("/home/jovyan/hyx/uocr-ms-swift-title-mask")
EV = U / "evaluation/pmc-fullce-16k-cont1623_all129"
GT = Path("/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth")
OMNI = Path("/home/jovyan/pdf2text-auto-label-eval/OmniDocBench_v1.5/result".replace("/home/jovyan/pdf2text", "/home/jovyan/hyx/pdf2text"))
OMNI = Path("/home/jovyan/hyx/pdf2text-auto-label-eval/OmniDocBench_v1.5/result")
import sys
sys.path.insert(0, str(U / "evaluation/tooling/agentbuilder_pkg"))
from agentbuilder_eval.title_metric import score_markdown_pair, extract_title_lines

NEW = "pmc-fullce-16k-cont1623"
mn = json.load(open(EV / f"agentbuilder/{NEW}/metrics.json"))["metrics"]
old_title = mn["title_accuracy"]
# 87 篇 GT 标题文档集（88 减无预测 Handwriting），S5=0.3/0.7，与 corrected_overall 口径一致
set87 = [p.stem for p in sorted(GT.glob("*.md")) if extract_title_lines(p.read_text(encoding="utf-8")) and p.stem != "Handwriting document"]
ptd = EV / f"pred_titles/{NEW}"
vals = []
for d in set87:
    pp = ptd / (d + ".md")
    if not pp.is_file():
        continue
    vals.append(score_markdown_pair((GT / (d + ".md")).read_text(encoding="utf-8"), pp.read_text(encoding="utf-8"), level_weight=0.3, content_weight=0.7) or 0.0)
title87 = sum(vals) / len(vals)
corr_overall = mn["overall"] + 0.1 * (title87 - old_title)

# 退化四板斧（dup>=0.3 | len>=2x | len<=1/3 | zlib<0.15）
pc = EV / f"pred_clean/{NEW}"
deg = {}
for f in sorted(pc.glob("*.md")):
    t = f.read_text(encoding="utf-8")
    gtf = GT / f.name
    gtxt = gtf.read_text(encoding="utf-8") if gtf.is_file() else ""
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    dup = 1 - len(set(lines)) / len(lines) if lines else 0.0
    lr = len(t) / max(1, len(gtxt))
    zr = len(zlib.compress(t.encode(), 9)) / max(1, len(t.encode()))
    why = []
    if dup >= 0.3: why.append(f"dup{dup:.2f}")
    if lr >= 2: why.append(f"len{lr:.1f}x")
    if gtxt and lr <= 1 / 3: why.append(f"collapse{lr:.2f}x")
    if zr < 0.15: why.append(f"zlib{zr:.3f}")
    if why: deg[f.stem] = ",".join(why)

# 健康子集 pooled（剔除退化篇后重算官方口径）
def pooled(suffix, agg):
    a, k = [], []
    for p in OMNI.glob(f"swp6_*{suffix}"):
        doc = p.name[5: -len(suffix)]
        j = json.load(open(p))
        vv = []
        if suffix == "_table_per_table_TEDS.json":
            vv = [x.get("TEDS") for x in j.values() if isinstance(x, dict) and x.get("TEDS") is not None]
        else:
            for v in j.values():
                vv += [x for x in (v if isinstance(v, list) else v.values() if isinstance(v, dict) else [v]) if isinstance(x, (int, float))]
        a += vv
        if doc not in deg: k += vv
    f = (lambda x: x) if agg == "avg" else (lambda x: 1 - x)
    return (sum(a) / len(a) if a else None, sum(k) / len(k) if k else None, len(a), len(k))

teds_all, teds_ex, tn, tk = pooled("_table_per_table_TEDS.json", "raw")
txt_all, txt_ex, xn, xk = pooled("_text_block_per_page_edit.json", "inv")
ord_all, ord_ex, on, ok_ = pooled("_reading_order_per_page_edit.json", "inv")

# 验证门(train_loss 取 cont1623 自己的 trainer_state)
ts_p = Path("/tmp/cont1623_ckpt_run")
sh = sorted(ts_p.glob("checkpoint-*/trainer_state.json"),
            key=lambda p: int(p.parts[-2].split("-")[-1]))[-1]
hist = json.load(open(sh))["log_history"]
tl = [x["loss"] for x in hist if "loss" in x and "eval_loss" not in x][-10:]
train_loss_tail = sum(tl) / len(tl)

gates = [
    ("退化篇数(四板斧) <= 3", f"{len(deg)} 篇: {dict(sorted(deg.items())[:8])}", len(deg) <= 3),
    ("pooled TEDS(剔除退化) >= 0.50", f"{teds_ex:.4f} (n={tk}/{tn})", (teds_ex or 0) >= 0.50),
    ("pooled text_acc(剔除退化) >= 0.80", f"{1 - txt_ex:.4f} (n={xk}/{xn})", (1 - txt_ex) >= 0.80),
    ("pooled order_acc(剔除退化) >= 0.925", f"{1 - ord_ex:.4f} (n={ok_}/{on})", (1 - ord_ex) >= 0.925),
    ("train_loss 末10均值 >= 0.10", f"{train_loss_tail:.4f}", train_loss_tail >= 0.10),
]

refs = {"base": ("readoc-view-16k_all129/agentbuilder/base", 0.7554),
        "readoc-view full-ce": ("readoc-view-16k_all129/agentbuilder/full-ce", 0.7400),
        "pmc-fullce-16k": ("pmc-fullce-16k_all129/agentbuilder/pmc-fullce-16k", 0.7092)}
rows = []
for name, (rel, corr) in refs.items():
    m = json.load(open(U / "evaluation" / rel / "metrics.json"))["metrics"]
    rows.append((name, m["text_accuracy"], m["table_teds"], m["reading_order_accuracy"], corr))
rows.append((NEW, mn["text_accuracy"], mn["table_teds"], mn["reading_order_accuracy"], corr_overall))

doc = []
doc.append(f"# 支线评测：{NEW}（129 统一协议，{__import__('datetime').date.today()}）\n")
doc.append("> 支线文档，不合入周报；口径与 0827 契约一致（pred_clean 剥壳、AgentBuilder 0.3/0.3/0.3/0.1、title=87 篇 S5 splice）。\n")
doc.append("## 1. 总分对比（Overall 为 title 修正后）\n")
doc.append("| 路由 | text_acc | TEDS | order_acc | Overall(修正) |")
doc.append("|---|---:|---:|---:|---:|")
for r in rows:
    doc.append("| " + " | ".join([r[0]] + [f"{x:.4f}" for x in r[1:4]] + [f"**{r[4]:.4f}**"]) + " |")
doc.append(f"\nOmniDocBench：TextEdit {mn['TextEdit']:.4f} / TEDS {mn['table_teds']:.4f} / TEDS-S {mn['table_teds_structure_only']:.4f} / FormulaEdit {mn['formula_edit']:.4f} / OrderEdit {mn['reading_order_edit']:.4f}")
doc.append(f"title 分量（87 篇 S5）：{title87:.4f}（旧塌零口径 {old_title:.5f}）；Overall splice：{mn['overall']:.4f} -> **{corr_overall:.4f}**\n")
doc.append("## 2. 验证门\n")
doc.append("| 门 | 实测 | 判定 |"); doc.append("|---|---|---|")
for g, v, ok in gates:
    doc.append(f"| {g} | {v} | {'PASS' if ok else 'FAIL'} |")
doc.append(f"\n（Train Fit 验收 B 未含在本一键流程；GPU 空时跑 probe_train_fit.py 后补）\n")
doc.append("## 3. 退化文档清单（四板斧：行dup>=0.3 / len>=2x / 塌缩<=1/3 / zlib<0.15）\n")
if deg:
    doc.append("| 文档 | 信号 |"); doc.append("|---|---|")
    for d, w in sorted(deg.items(), key=lambda x: -len(x[1])): doc.append(f"| {d} | {w} |")
else:
    doc.append("无")
doc.append(f"\n## 4. 健康子集（剔除退化篇，官方 pooled 口径）\n")
doc.append(f"TEDS {teds_all:.4f}(全量 n={tn}) -> {teds_ex:.4f}(剔除退化 n={tk}) | text_acc {1 - txt_all:.4f} -> {1 - txt_ex:.4f} | order_acc {1 - ord_all:.4f} -> {1 - ord_ex:.4f}\n")
doc.append("## 5. 评分覆盖与口径备注\n")
res = Path("/tmp/rswp6/results.txt")
if res.is_file():
    fails = [l for l in res.read_text().splitlines() if "rc0" not in l]
    doc.append(f"sweep 覆盖 {sum(1 for l in res.read_text().splitlines() if 'rc0' in l)}/129；失败篇：{fails if fails else '无'}")
doc.append("\n- Overall 公式 0.3·text+0.3·TEDS+0.3·order+0.1·title（title=87 篇 GT 标题文档 S5 均值，splice 与 title_rebuild_2026-09-17 同口径）")
doc.append("- 公式项在 Total 不在 Overall；引用 Total 时注明。")
doc.append("- 退化检测含 zlib 板斧（行内循环），比 09-17 三板斧更严。")
doc.append("- 本轮 AGGRESSIVE_FILES 为两篇人工复核白名单；仅清理 grounding 脚手架，正文不删。")
out = EV / "EVAL_BRANCH_REPORT.md"
out.write_text("\n".join(doc) + "\n", encoding="utf-8")
print("[P6] branch doc:", out)
print("[P6] corrected Overall:", round(corr_overall, 4), "| degenerate docs:", len(deg), "| title87:", round(title87, 4))
PYEOF
echo "[oneclick] ALL DONE $(date '+%F %T')"
