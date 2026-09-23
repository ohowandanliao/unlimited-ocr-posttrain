#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_pmc_singlepage_jsonl.py — 从 PMC 文档池（32k − 16k）每篇抽 1 页构建单页训练样本

背景：pmc-fullce-16k 几乎全是多页文档级样本（prompt "<image>Multi page merge."），
单页形态（prompt "<image>document parsing."，1 张页图）严重欠训练（评测 129 篇中 109 篇单页）。
本脚本从"32k 用了但 16k 没用"的文档池（与 16k train/val/test 零重叠，无泄漏）每篇取 1 页
（优先 page_idx=0，不合格退 page_idx=1），用生产同款清洗逻辑 gen_readoc_md_v2.py@v2.2.1
的 json_to_markdown() 对"只含该页 pdf_info 的迷你 middle.json"产出页级目标文本。

行契约：逐字段镜像 pmc-fullce 行 + ms_swift_title_mask.data_contract.validate_training_row 硬门：
  channel=title_reviewed；prompt=<image>document parsing.（恰好 1 图）；assistant 目标为
  canonical 换行形态；meta.split ∈ {train,validation,test} 且与文件名一致；
  title_review_status ∈ {SILVER_ACCEPTED,HUMAN_ACCEPTED} + 全套 title_* 字段；
  title_target_sha256 = sha256(目标)；title_heading_count = CommonMark ATX 标题数且 > 0。
  因此合格判据除指令给的 ①字符数 ②有效行数 ③含表格块 外，追加 ④≥1 个 ATX 标题（硬门）。

页合格判据（对每个候选页）：
  P0 页图存在（os.path.isfile）
  P1 该页 para_blocks 不含表格类块（type 含 "table"，如 "table"；试点一律排除，防跨页截断表）
  P2 200 ≤ 目标字符数 ≤ 12000（16k 窗口安全余量；本行仅 1 图）
  P3 非空文本行 ≥ 3
  P4 ATX 标题数 ≥ 1（validate_training_row 硬门，先在此挡掉）

用法：
  python3 build_pmc_singlepage_jsonl.py \
    --data-root <pmc-fullce-data> --source-root <middle-json-root> \
    --generator-script <gen_readoc_md_v2.py> --image-root <pmc-image-root> \
    --tokenizer <Unlimited-OCR-model> --out-dir <new-dir> [--limit 100]

各路径也可分别通过 PMC_DATA_ROOT、PMC_MIDDLE_ROOT、PMC_MARKDOWN_GENERATOR、
PMC_IMAGE_ROOT 和 MODEL_PATH 提供。输出目录已存在时拒绝运行。

输出（--out-dir 内）：train.jsonl、build_stats.json、example_p{25,50,90}_*.md
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import re
import tempfile
import time
from collections import Counter
from pathlib import Path

UOCR_BUNDLE = str(Path(__file__).resolve().parents[2])
DATA_ROOT = DS32 = DS16 = SRC = GEN_SCRIPT = IMG_ROOT = TOK_PATH = ""

MIN_CHARS, MAX_CHARS = 200, 12000
MIN_LINES = 3
CANDIDATE_PAGES = (0, 1)          # 每篇最多试 2 页
VISUAL_PER_PAGE = 273
BUDGETS = {"16k": 16384, "32k": 32768}
BUCKETS = [(4096, "le_4k"), (8192, "le_8k"), (12288, "le_12k"), (16384, "le_16k"),
           (24576, "le_24k"), (32768, "le_32k")]
PROMPT_SINGLE = "document parsing."
USER_CONTENT = "<image>" + PROMPT_SINGLE
CHANNEL = "title_reviewed"
RULESET = "gen_readoc_md_v2.py@v2.2.1"
REVIEW_VERSION = "pmc-v2.2.1-silver"
REVIEWER = "gen_readoc_md_v2.py@v2.2.1"
REVIEWED_AT = "2026-09-17T00:00:00+08:00"
BUILD_DATE = "2026-09-17"

sys.path.insert(0, UOCR_BUNDLE)
from ms_swift_title_mask.core import canonicalize_target, heading_count, sha256_text  # noqa: E402


def load_generator():
    """import 生产清洗脚本（v2.2.1），复用其 json_to_markdown()；默认全局即生产行为
    （MERGE_HYPHEN=False 盲保留连字符、USE_VOCAB=False，与全量 18400 篇构建一致）。"""
    spec = importlib.util.spec_from_file_location("gen_readoc_md_v2", GEN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jsonl_map(root):
    """读 train/validation/test 三文件 → {doc_id: row}（含来源文件名）。"""
    out, dup = {}, 0
    for name in ("train", "validation", "test"):
        p = os.path.join(root, f"{name}.jsonl")
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                did = r["meta"]["doc_id"]
                if did in out:
                    dup += 1
                    continue
                r["_src_file"] = name
                out[did] = r
    return out, dup


def middle_sha(doc_id):
    h = hashlib.sha256()
    with open(os.path.join(SRC, doc_id, "middle.json"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket_of(total):
    for cap, name in BUCKETS:
        if total <= cap:
            return name
    return "gt_32k"


def main():
    global DATA_ROOT, DS32, DS16, SRC, GEN_SCRIPT, IMG_ROOT, TOK_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="最多产出 N 个合格文档行（0=全部）")
    ap.add_argument(
        "--data-root",
        default=os.environ.get("PMC_DATA_ROOT", "/home/jovyan/hyx/dataset/pmc-fullce-data"),
    )
    ap.add_argument(
        "--source-root",
        default=os.environ.get(
            "PMC_MIDDLE_ROOT", "/home/jovyan/hyx/20260905_pmc_gt_doc_pack_v01/documents_old"
        ),
    )
    ap.add_argument(
        "--generator-script",
        default=os.environ.get(
            "PMC_MARKDOWN_GENERATOR",
            "/home/jovyan/hyx/dataset/pmc_doc_pack_v01_rendered_final/gen_readoc_md_v2.py",
        ),
    )
    ap.add_argument(
        "--image-root",
        default=os.environ.get(
            "PMC_IMAGE_ROOT", "/home/jovyan/hyx/dataset/pmc_doc_pack_v01_images/pmc"
        ),
    )
    ap.add_argument(
        "--tokenizer",
        default=os.environ.get("MODEL_PATH", "/home/jovyan/hyx/models/Unlimited-OCR"),
    )
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    DATA_ROOT = os.path.abspath(args.data_root)
    DS32 = os.path.join(DATA_ROOT, "pmc-fullce-32k")
    DS16 = os.path.join(DATA_ROOT, "pmc-fullce-16k")
    SRC = os.path.abspath(args.source_root)
    GEN_SCRIPT = os.path.abspath(args.generator_script)
    IMG_ROOT = os.path.abspath(args.image_root)
    TOK_PATH = os.path.abspath(args.tokenizer)
    args.out_dir = os.path.abspath(args.out_dir or os.path.join(DATA_ROOT, "pmc-spage-pilot"))

    for label, path, predicate in (
        ("32K dataset", DS32, os.path.isdir),
        ("16K dataset", DS16, os.path.isdir),
        ("middle.json root", SRC, os.path.isdir),
        ("Markdown generator", GEN_SCRIPT, os.path.isfile),
        ("image root", IMG_ROOT, os.path.isdir),
        ("tokenizer", TOK_PATH, os.path.isdir),
    ):
        if not predicate(path):
            ap.error(f"{label} does not exist: {path}")

    t0 = time.time()
    if os.path.exists(args.out_dir):
        ap.error(f"output directory already exists: {args.out_dir}")
    os.makedirs(args.out_dir)
    tmpdir = tempfile.mkdtemp(prefix=".tmp_mini_middle_", dir=args.out_dir)

    print("loading generator (production v2.2.1)...", flush=True)
    gen = load_generator()
    print("loading datasets for pool definition...", flush=True)
    d32, dup32 = load_jsonl_map(DS32)
    d16, _ = load_jsonl_map(DS16)
    pool = sorted(set(d32) - set(d16))
    print(f"pool: |32k|={len(d32)} |16k|={len(d16)} pool={len(pool)} dup32={dup32}", flush=True)

    rows, type_enum = [], Counter()
    rej_pages = Counter()          # 页级失败原因直方图
    docs_rejected = 0
    recovered_p1 = 0               # page0 失败但 page1 合格
    scanned = 0
    stop = False

    for doc_id in pool:
        if args.limit and len(rows) >= args.limit:
            stop = True
            break
        scanned += 1
        base = d32[doc_id]
        mid_path = os.path.join(SRC, doc_id, "middle.json")
        if not os.path.isfile(mid_path):
            rej_pages["no_middle_json"] += 1
            docs_rejected += 1
            continue
        try:
            with open(mid_path, encoding="utf-8") as f:
                j = json.load(f)
        except Exception as e:
            rej_pages[f"middle_load_error:{type(e).__name__}"] += 1
            docs_rejected += 1
            continue
        pages = j.get("pdf_info", []) if isinstance(j, dict) else []
        by_idx = {p.get("page_idx"): p for p in pages if isinstance(p, dict)}
        src_doc_pages = len(pages)

        picked = None                 # (page_idx, target, heading_count)
        attempts = []
        for k in CANDIDATE_PAGES:
            page = by_idx.get(k, pages[k] if k < len(pages) else None)
            if page is None:
                attempts.append(f"no_page_p{k}")
                rej_pages[f"no_page_p{k}"] += 1
                continue
            img = os.path.join(IMG_ROOT, doc_id, f"page_{k:04d}.png")
            if not os.path.isfile(img):
                attempts.append(f"img_missing_p{k}")
                rej_pages[f"img_missing_p{k}"] += 1
                continue
            pblocks = page.get("para_blocks", []) or []
            for b in pblocks:
                type_enum[str(b.get("type"))] += 1
            if any("table" in str(b.get("type", "")).lower() for b in pblocks):
                attempts.append(f"table_block_p{k}")
                rej_pages[f"table_block_p{k}"] += 1
                continue
            # 迷你 middle.json：只含该页 pdf_info → 单页清洗（无跨页并段、独立选主标题）
            mini = {"_backend": j.get("_backend"), "_version_name": j.get("_version_name"),
                    "pdf_info": [page]}
            tmp_mid = os.path.join(tmpdir, f"{doc_id}_p{k}.json")
            with open(tmp_mid, "w", encoding="utf-8") as f:
                json.dump(mini, f, ensure_ascii=False)
            try:
                content = gen.json_to_markdown(tmp_mid)
            except Exception as e:
                attempts.append(f"clean_error_p{k}:{type(e).__name__}")
                rej_pages[f"clean_error_p{k}:{type(e).__name__}"] += 1
                continue
            finally:
                if os.path.exists(tmp_mid):
                    os.remove(tmp_mid)
            try:
                target = canonicalize_target(content)
            except Exception:
                attempts.append(f"empty_target_p{k}")
                rej_pages[f"empty_target_p{k}"] += 1
                continue
            nc = len(target)
            if nc < MIN_CHARS:
                attempts.append(f"too_short_p{k}({nc})")
                rej_pages[f"too_short_p{k}"] += 1
                continue
            if nc > MAX_CHARS:
                attempts.append(f"too_long_p{k}({nc})")
                rej_pages[f"too_long_p{k}"] += 1
                continue
            nlines = sum(1 for ln in target.split("\n") if ln.strip())
            if nlines < MIN_LINES:
                attempts.append(f"few_lines_p{k}({nlines})")
                rej_pages[f"few_lines_p{k}"] += 1
                continue
            try:
                h = heading_count(target)
            except Exception as e:
                attempts.append(f"heading_parse_error_p{k}")
                rej_pages[f"heading_parse_error_p{k}"] += 1
                continue
            if h <= 0:
                attempts.append(f"no_heading_p{k}")
                rej_pages[f"no_heading_p{k}"] += 1
                continue
            # 硬门：页目标首个 ATX 必须等于生产 document.md 首个 ATX（保证单页样本标题正确）
            _gt = base["meta"].get("gt_path")
            _doc_head = None
            if _gt and os.path.isfile(_gt):
                for _ln in open(_gt, encoding="utf-8"):
                    _m = re.match(r"^(#{1,6})\s+(.+)", _ln.strip())
                    if _m:
                        _doc_head = _m.group(2).strip()
                        break
            _page_head = None
            for _ln in target.splitlines():
                _m = re.match(r"^(#{1,6})\s+(.+)", _ln.strip())
                if _m:
                    _page_head = _m.group(2).strip()
                    break
            if not _doc_head or _page_head != _doc_head:
                attempts.append(f"title_mismatch_p{k}")
                rej_pages[f"title_mismatch_p{k}"] += 1
                continue
            picked = (k, target, h, src_doc_pages)
            break

        if picked is None:
            docs_rejected += 1
            continue
        if attempts:
            recovered_p1 += 1    # page0 被拒、page1 合格

        k, target, h, src_doc_pages = picked
        meta = dict(base["meta"])          # 继承 32k 原行 meta（可能来自 val/test）
        meta.update({
            "split": "train",              # 强制覆盖：页样本进我们 train，防泄漏约定
            "sample_form": "single_page",
            "n_pages": 1,
            "target_chars": len(target),
            "visual_tokens": VISUAL_PER_PAGE,
            "cleaning_version": RULESET,
            "build_date": base["meta"].get("build_date"),
            "target_sha256": sha256_text(target),
            "title_review_status": "SILVER_ACCEPTED",
            "title_review_id": f"pmc_{doc_id}:{REVIEW_VERSION}",
            "title_review_version": REVIEW_VERSION,
            "title_reviewer": REVIEWER,
            "title_reviewed_at": REVIEWED_AT,
            "title_ruleset_version": RULESET,
            "title_source_sha256": middle_sha(doc_id),
            "title_target_sha256": sha256_text(target),
            "title_heading_count": h,
        })
        rows.append({
            "id": f"pmc_{doc_id}_p{k:04d}",
            "channel": CHANNEL,
            "messages": [
                {"role": "user", "content": USER_CONTENT},
                {"role": "assistant", "content": target},
            ],
            "images": [os.path.join(IMG_ROOT, doc_id, f"page_{k:04d}.png")],
            "meta": meta,
        })
        if len(rows) % 25 == 0:
            print(f"  [{len(rows)}/{args.limit or '*'}] scanned={scanned} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

    shutil.rmtree(tmpdir, ignore_errors=True)
    if not rows:
        print("no qualified rows; abort (nothing written)")
        return

    # ---- token 口径与 build_pmc_fullce_jsonl.py 完全一致 ----
    print("tokenizing targets (Unlimited-OCR tokenizer)...", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)
    pt_text = len(tok(PROMPT_SINGLE, add_special_tokens=False)["input_ids"])
    enc = tok([r["messages"][1]["content"] for r in rows], add_special_tokens=False)["input_ids"]
    for r, ids in zip(rows, enc):
        m = r["meta"]
        m["target_tokens"] = len(ids)
        m["prompt_tokens"] = VISUAL_PER_PAGE * 1 + pt_text
        total = m["prompt_tokens"] + m["target_tokens"]
        m["total_tokens"] = total
        m["bucket"] = bucket_of(total)
        m["fits_16k"] = total <= BUDGETS["16k"]
        m["fits_32k"] = total <= BUDGETS["32k"]
        m["row_sha256"] = hashlib.sha256(
            json.dumps(r["messages"], ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    # ---- 写 train.jsonl ----
    out_jsonl = os.path.join(args.out_dir, "train.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 示例 md：目标长度 p25/p50/p90 附近各 1 行 ----
    chars = sorted(r["meta"]["target_chars"] for r in rows)

    def pct(q):
        i = min(len(chars) - 1, max(0, round(q * (len(chars) - 1))))
        return chars[i]

    examples = {}
    for q in (0.25, 0.5, 0.90):
        want = pct(q)
        r = min(rows, key=lambda r: abs(r["meta"]["target_chars"] - want))
        name = f"example_p{int(q*100)}_{r['id']}.md"
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            f.write(r["messages"][1]["content"])
        examples[f"p{int(q*100)}"] = {"id": r["id"], "src_page": r["id"].rsplit("_p", 1)[-1],
                                      "target_chars": r["meta"]["target_chars"], "file": name}

    # ---- 统计 ----
    imgs_ok = all(os.path.isfile(p) for r in rows for p in r["images"])
    img_missing_rows = [r["id"] for r in rows for p in r["images"] if not os.path.isfile(p)]
    ttoks = sorted(m["target_tokens"] for m in (r["meta"] for r in rows))
    tots = sorted(m["total_tokens"] for m in (r["meta"] for r in rows))
    stats = {
        "pool_size": len(pool), "docs_scanned": scanned, "rows_written": len(rows),
        "docs_qualified": len(rows), "docs_rejected": docs_rejected,
        "stopped_by_limit": stop,
        "page1_recovered_after_page0_reject": recovered_p1,
        "page_reject_reasons": dict(sorted(rej_pages.items())),
        "para_block_type_enum_seen": dict(type_enum.most_common()),
        "target_chars_p10_p50_p90": [pct(0.10), pct(0.50), pct(0.90)],
        "target_chars_min_max": [chars[0], chars[-1]],
        "target_tokens_p50_max": [ttoks[len(ttoks) // 2], ttoks[-1]],
        "total_tokens_p50_max": [tots[len(tots) // 2], tots[-1]],
        "fits_16k_true": sum(1 for r in rows if r["meta"]["fits_16k"]),
        "image_paths_all_exist": imgs_ok, "image_missing_rows": img_missing_rows,
        "prompt": USER_CONTENT, "channel": CHANNEL,
        "cleaning_script": GEN_SCRIPT, "cleaning_call": "json_to_markdown(mini_middle.json) "
        "with pdf_info=[单页], MERGE_HYPHEN=False/USE_VOCAB=False (生产默认)",
        "tokenizer": TOK_PATH, "visual_tokens_per_page": VISUAL_PER_PAGE,
        "ruleset": RULESET, "build_date": BUILD_DATE,
        "examples": examples,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "build_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"DONE -> {out_jsonl} ({len(rows)} rows) elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
