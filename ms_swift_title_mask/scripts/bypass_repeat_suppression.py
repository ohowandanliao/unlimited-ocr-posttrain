#!/usr/bin/env python3
"""解码期复读抑制旁路验证(bypass,只读既有评测、不动主协议)。

目的: 验证"数字归一 + 尾块连续重复"停止判据能否止住复读。推理路径与主协议完全一致
(OCRService.infer + InferRequest:no_repeat_ngram_size=35、window 128/1024、temperature 0、
max_length 32768、按页数路由 prompt),仅在 logits processor 链末尾追加抑制器——每 64 个
生成 token 检查一次:把已生成文本做数字归一(数字串→#、<td>/<tr>/<th> 标签去属性、空白折叠)
后,若文末存在 ≥3 次"链式相邻"的 ≥32 字符重复块,则强制 EOS。未触发的文档,其生成结果与
主协议逐位一致(触发前不修改任何分数)。

对象: 两轮失效文档并集(脚本内按四规则从两轮 pred_clean 自行重算,与 P6 检测同逻辑)。
产物: <out>/pred_raw/*.md → strip → <out>/pred_clean/*.md → 逐篇 sweep(swp5_)
      → bypass_report.json + bypass_report.md(触发诊断 + 失效文档数 + 逐表 TEDS/text/order 对照)。

用法(GPU1,须等主评测 P1 结束释放显存;评测 sweep 是 CPU 可并行):
  source $POSTTRAIN_ROOT/ms_swift_title_mask/scripts/hyx_env.sh
  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" bypass_repeat_suppression.py \
      --ckpt <adapter_dir> --out-dir $UOCR_ROOT/evaluation/bypass_repeat_ckpt1623
可选: --limit N(冒烟)/ --max-length 32768 / --min-gen 256 / --check-every 64 / --gap 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
POSTTRAIN_ROOT = SCRIPTS.parent
STRIP = SCRIPTS / "strip_grounding_shell.py"
UOCR_ROOT: Path
BASE_MODEL: Path
GT: Path
PDF_ROOT: Path
OMNI: Path
OMNI_RESULT: Path
VENV_OMNI: Path
PRED_DIRS: dict[str, Path]
SWEEP_PREFIX_OLD = {"pmc16k": "swp2_", "mix": "swp3_"}
SWEEP_PREFIX_NEW = "swp5_"

# ---------- 四规则检测(与评测脚本 P6 同逻辑) ----------

def detect_sick(pred_dir: Path) -> dict[str, str]:
    sick = {}
    for f in sorted(Path(pred_dir).glob("*.md")):
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
        if why: sick[f.stem] = ",".join(why)
    return sick

# ---------- 渲染(probe 同逻辑,自含) ----------

def render_pdf(pdf_path: Path, page_root: Path, dpi: int) -> list[Path]:
    import fitz
    with fitz.open(pdf_path) as document:
        scale = dpi / 72.0
        pages = [page_root / f"page-{index:04d}.png" for index in range(1, document.page_count + 1)]
        existing = {p for p in page_root.glob("page-*.png") if p.is_file()} if page_root.exists() else set()
        if existing == set(pages):
            return pages
        page_root.mkdir(parents=True, exist_ok=True)
        for p in page_root.glob("page-*.png"):
            p.unlink()
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pixmap.save(pages[index - 1], output="png")
    return pages


def page_cache_dir(cache_root: Path, pdf_path: Path, dpi: int) -> Path:
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return cache_root / f"dpi-{dpi}" / digest

# ---------- 抑制器 ----------

def normalize_text(t: str) -> str:
    t = re.sub(r"<(td|tr|th)[^>]*>", r"<\1>", t)
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"\s+", " ", t)
    return t


def tail_repeat_block(t: str, min_l: int, gap: int, need: int) -> str | None:
    """文末是否存在 need 次"链式相邻"的重复块:相邻两次出现的结束位置相差 ≤ L+gap。

    实现: b = t[-L:](末尾这次出现本身计入);自 end=len(t) 起向前找上一次出现,其结束位置
    必须落在 (end-L-gap, end) 开区间内(rfind 上界取 end-1,排除末尾这次自身),链式回溯
    need-1 次。周期 P ≤ L 的严格循环(计数器/数列/单元循环)与周期 ≤ L+gap 的 ABAB 交替
    均在 L=2048 档命中。
    """
    for L in (2048, 1024, 512, 256, 128, 64):
        if L < min_l or len(t) < L:
            continue
        b = t[-L:]
        e, ok = len(t), True
        for _ in range(need - 1):
            p = t.rfind(b, max(0, e - 2 * L - gap), e - 1)
            if p < 0:
                ok = False
                break
            e = p + L
        if ok:
            return b
    return None


def make_suppressed_processor_class(state: dict, cfg: dict):
    import importlib
    mm = sys.modules[next(m for m in list(sys.modules) if m.split(".")[-1] == "modeling_unlimitedocr")]
    base_cls = mm.SlidingWindowNoRepeatNgramProcessor
    tok = state["tok"]

    class RepeatStopSuppressed(base_cls):
        def __call__(self, input_ids, scores):
            scores = super().__call__(input_ids, scores)
            seq = input_ids[0].tolist()
            if state["prompt_len"] is None:
                state["prompt_len"] = len(seq)
            gen_n = len(seq) - state["prompt_len"]
            if (not state["tripped"] and gen_n >= cfg["min_gen"]
                    and gen_n - state["last_n"] >= cfg["check_every"]):
                state["last_n"] = gen_n
                text = tok.decode(seq[state["prompt_len"]:], skip_special_tokens=True)
                block = tail_repeat_block(normalize_text(text), cfg["min_block"], cfg["gap"], cfg["need"])
                if block:
                    state["tripped"] = True
                    state["info"] = {"trip_gen_tokens": gen_n, "block_chars": len(block),
                                     "block_head": block[:80]}
            if state["tripped"]:
                s = scores.clone()
                s[:, :] = float("-inf")
                s[:, state["eos_id"]] = 0.0
                return s
            return scores

    mm.SlidingWindowNoRepeatNgramProcessor = RepeatStopSuppressed
    return base_cls

# ---------- sweep(与评测脚本 P3 同逻辑,单文档) ----------

def sweep_one(doc: str, pred_md: Path, work: Path) -> str | None:
    gt_d, pd_d = work / f"gt_{doc}", work / f"pred_{doc}"
    subprocess.run(["rm", "-rf", str(gt_d), str(pd_d)], check=True)
    gt_d.mkdir(parents=True); pd_d.mkdir(parents=True)
    if not (GT / f"{doc}.md").is_file():
        return "MISSING_GT"
    subprocess.run(["cp", str(GT / f"{doc}.md"), str(gt_d)], check=True)
    subprocess.run(["cp", str(pred_md), str(pd_d)], check=True)
    rc = subprocess.run(
        ["timeout", "600", str(VENV_OMNI), "run_md2md.py", "--gt_dir", str(gt_d),
         "--pred_dir", str(pd_d), "--save_name", f"{SWEEP_PREFIX_NEW}{doc}"],
        cwd=str(OMNI), capture_output=True, text=True).returncode
    subprocess.run(["rm", "-rf", str(gt_d), str(pd_d)], check=True)
    return None if rc == 0 else f"rc{rc}"

# ---------- 汇总对照 ----------

def pooled_for(prefix: str, suffix: str, docs: set[str]) -> tuple[float | None, int]:
    vals = []
    for p in OMNI_RESULT.glob(f"{prefix}*{suffix}"):
        doc = p.name[len(prefix):-len(suffix)]
        if doc not in docs:
            continue
        j = json.load(open(p))
        if suffix == "_table_per_table_TEDS.json":
            vv = [x.get("TEDS") for x in j.values() if isinstance(x, dict) and x.get("TEDS") is not None]
        else:
            vv = []
            for v in j.values():
                vv += [x for x in (v if isinstance(v, list) else (v.values() if isinstance(v, dict) else [v]))
                       if isinstance(x, (int, float))]
        vals += vv
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)

def main() -> None:
    global UOCR_ROOT, BASE_MODEL, GT, PDF_ROOT, OMNI, OMNI_RESULT, VENV_OMNI, PRED_DIRS

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--uocr-root",
        type=Path,
        default=os.environ.get("UOCR_ROOT", "/home/jovyan/hyx/uocr-ms-swift-title-mask"),
    )
    ap.add_argument(
        "--base-model",
        type=Path,
        default=os.environ.get("MODEL_PATH", "/home/jovyan/hyx/models/Unlimited-OCR"),
    )
    ap.add_argument(
        "--gt-root",
        type=Path,
        default=os.environ.get(
            "EVAL_GT_ROOT", "/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/groundtruth"
        ),
    )
    ap.add_argument(
        "--pdf-root",
        type=Path,
        default=os.environ.get(
            "EVAL_PDF_ROOT", "/home/jovyan/hyx/pdf2text-badcases/evaluation_datasets/pdf/source"
        ),
    )
    ap.add_argument(
        "--omnidocbench-root",
        type=Path,
        default=os.environ.get(
            "OMNIDOCBENCH_ROOT",
            "/home/jovyan/hyx/pdf2text-auto-label-eval/OmniDocBench_v1.5",
        ),
    )
    ap.add_argument(
        "--omnidocbench-python",
        type=Path,
        default=os.environ.get("OMNIDOCBENCH_PYTHON"),
    )
    ap.add_argument("--pmc-pred-dir", type=Path)
    ap.add_argument("--mix-pred-dir", type=Path)
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--min-gen", type=int, default=256)
    ap.add_argument("--check-every", type=int, default=64)
    ap.add_argument("--min-block", type=int, default=256,
                    help="最小重复块长;64/128 会误伤合法连号标题与重复短结构(冒烟#2 实测)")
    ap.add_argument("--gap", type=int, default=32)
    ap.add_argument("--need", type=int, default=3)
    ap.add_argument("--extra-docs", type=str, default="",
                    help="附加健康对照文档名(逗号分隔),用于检测误触发")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 篇(冒烟)")
    args = ap.parse_args()

    UOCR_ROOT = args.uocr_root.resolve()
    BASE_MODEL = args.base_model.resolve()
    GT = args.gt_root.resolve()
    PDF_ROOT = args.pdf_root.resolve()
    OMNI = args.omnidocbench_root.resolve()
    OMNI_RESULT = OMNI / "result"
    VENV_OMNI = (
        args.omnidocbench_python.resolve()
        if args.omnidocbench_python
        else UOCR_ROOT / "evaluation/tooling/omni-eval-venv/bin/python"
    )
    PRED_DIRS = {
        "pmc16k": (
            args.pmc_pred_dir.resolve()
            if args.pmc_pred_dir
            else UOCR_ROOT
            / "evaluation/pmc-fullce-16k_all129/pred_clean/pmc-fullce-16k"
        ),
        "mix": (
            args.mix_pred_dir.resolve()
            if args.mix_pred_dir
            else UOCR_ROOT
            / "evaluation/pmc-readoc-spage-16k_all129/pred_clean/pmc-readoc-spage-16k"
        ),
    }
    cfg = {"min_gen": args.min_gen, "check_every": args.check_every,
           "min_block": args.min_block, "gap": args.gap, "need": args.need}
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "pred_raw").mkdir(exist_ok=True)

    # 1) 失效文档并集(两轮四规则重算)
    sick = {r: detect_sick(d) for r, d in PRED_DIRS.items() if d.is_dir()}
    union = sorted({**sick.get("pmc16k", {}), **sick.get("mix", {})})
    extra = [x.strip() for x in args.extra_docs.split(",") if x.strip()]
    if args.limit:
        union = union[: args.limit]
    union = union + [d for d in extra if d not in union]
    print(f"[bypass] union sick docs: {len(union) - len(extra)} + controls {len(extra)}")
    for d in union:
        tag = ("pmc" if d in sick.get("pmc16k", {}) else "") + ("+mix" if d in sick.get("mix", {}) else "")
        print(f"  - {d} [{tag or 'mix'}] {sick.get('mix', {}).get(d) or sick.get('pmc16k', {}).get(d)}")

    # 2) 载模型 + 打补丁
    sys.path.insert(0, str(SCRIPTS))
    import serve_unlimited_ocr as srv
    service = srv.OCRService(BASE_MODEL, {srv.FULL_CE_MODEL_NAME: args.ckpt.resolve(),
                                          srv.TITLE_WEIGHTED_MODEL_NAME: args.ckpt.resolve()}, "cuda:0")
    state = {"tok": service.tokenizer, "eos_id": service.tokenizer.eos_token_id,
             "prompt_len": None, "last_n": 0, "tripped": False, "info": {}}
    orig_cls = make_suppressed_processor_class(state, cfg)
    print(f"[bypass] model loaded; suppressor armed (cfg={cfg})")

    # 3) 逐篇推理(与主协议同路径)
    results = {}
    for i, doc in enumerate(union, 1):
        pdf = PDF_ROOT / f"{doc}.pdf"
        if not pdf.is_file():
            results[doc] = {"error": "pdf missing"}
            continue
        imgs = render_pdf(pdf, page_cache_dir(out / "page-cache", pdf, 144), 144)
        state.update(prompt_len=None, last_n=0, tripped=False, info={})
        req = srv.InferRequest(model=srv.FULL_CE_MODEL_NAME,
                               image_paths=[str(p) for p in imgs],
                               max_length=args.max_length)
        r = service.infer(req)
        (out / "pred_raw" / f"{doc}.md").write_text(r["text"], encoding="utf-8")
        results[doc] = {"pages": len(imgs), "gen_tokens": r.get("output_tokens"),
                        "tripped": state["tripped"], **state["info"]}
        print(f"[bypass] {i}/{len(union)} {doc}: pages={len(imgs)} tripped={state['tripped']} "
              f"tokens={r.get('output_tokens')} {state['info'].get('block_chars', '')}")
    (out / "bypass_infer_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    tripped = [d for d, v in results.items() if v.get("tripped")]
    print(f"[bypass] inference done; tripped {len(tripped)}/{len(results)}: {tripped}")

    # 4) 剥壳 + check
    rc = subprocess.run([sys.executable, str(STRIP), "--in-dir", str(out / "pred_raw"),
                         "--out-dir", str(out / "pred_clean")]).returncode
    if rc != 0:
        print("[bypass] conservative strip 拒绝落盘:先人工复核 pred_raw 畸形件(坑14),"
              "对确认件用 --aggressive 后重跑 score 阶段。")
        return

    # 5) 逐篇 sweep
    work = Path("/tmp/rswp5"); work.mkdir(exist_ok=True)
    sweep_rc = {}
    for doc in union:
        f = out / "pred_clean" / f"{doc}.md"
        if f.is_file():
            sweep_rc[doc] = sweep_one(doc, f, work)
            print(f"[bypass] sweep {doc}: {'ok' if sweep_rc[doc] is None else sweep_rc[doc]}")

    # 6) 复发检测 + 新旧对照
    sick_new = detect_sick(out / "pred_clean")
    docs = set(union)
    report = {"config": cfg, "ckpt": str(args.ckpt), "docs": union,
              "infer": results, "sweep_rc": sweep_rc,
              "sick_before": {r: s for r, s in sick.items()},
              "sick_after": sick_new,
              "compare": {}}
    for suffix, agg in [("_table_per_table_TEDS.json", "raw"),
                        ("_text_block_per_page_edit.json", "inv"),
                        ("_reading_order_per_page_edit.json", "inv")]:
        row = {}
        for route, pref in SWEEP_PREFIX_OLD.items():
            m, n = pooled_for(pref, suffix, docs)
            row[route] = {"mean": m, "n": n}
        m, n = pooled_for(SWEEP_PREFIX_NEW, suffix, docs)
        row["bypass"] = {"mean": m, "n": n}
        report["compare"][suffix] = row
    (out / "bypass_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"[bypass] report: {out / 'bypass_report.json'}")
    print(f"[bypass] sick before(union)={len(union)} after={len(sick_new)}: {sick_new}")


if __name__ == "__main__":
    main()
