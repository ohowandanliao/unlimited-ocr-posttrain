#!/usr/bin/env python3
"""Exact tokenizer length pre-scan for reviewed title-training JSONL."""
from __future__ import annotations
import argparse, json, sys, tempfile
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from ms_swift_title_mask.core import TitleMaskError, sha256_text
from ms_swift_title_mask.data_contract import file_sha256, publish_directory, read_jsonl, validate_training_row
PINNED_MS_SWIFT_COMMIT = "1a1ba3ee86488af323ef9b64ca3d34edee90ab11"
VISUAL_TOKENS_PER_IMAGE = 273  # Pinned unlimited_ocr v1/no-crop, image_size=1024.
SCHEMA_VERSION = "uocr-length-buckets-v3"
BUCKETS = ((4096, "le_4k"), (8192, "le_8k"), (16384, "le_16k"), (24576, "le_24k"), (32768, "le_32k"))
LEGACY_POOLS = frozenset({"readoc_full", "pmc_full", "pmc_single"})
def bucket_for(length):
    for limit, name in BUCKETS:
        if length <= limit: return name
    return "overflow"
def _ids(tokenizer, text):
    result = tokenizer(text, add_special_tokens=False)
    ids = result["input_ids"] if isinstance(result, dict) else result.input_ids
    if not isinstance(ids, (list, tuple)) or (ids and isinstance(ids[0], (list, tuple))): raise TitleMaskError("invalid tokenizer ids")
    return list(ids)
def validate_tokenizer(tokenizer):
    if not getattr(tokenizer, "is_fast", False): raise TitleMaskError("a fast tokenizer is required")
    if len(_ids(tokenizer, "<image>")) != 1: raise TitleMaskError("<image> must be exactly one token")
    if not isinstance(getattr(tokenizer, "bos_token_id", None), int) or not isinstance(getattr(tokenizer, "eos_token_id", None), int):
        raise TitleMaskError("tokenizer must provide integer bos/eos ids")
def count_row_tokens(row, tokenizer, *, visual_tokens_per_image=VISUAL_TOKENS_PER_IMAGE):
    messages, images = row.get("messages"), row.get("images")
    if not isinstance(messages, list) or len(messages) != 2 or not isinstance(images, list) or not images: raise TitleMaskError(f"{row.get('id')}: invalid row")
    prompt, target = messages[0].get("content"), messages[1].get("content")
    if not isinstance(prompt, str) or not isinstance(target, str) or prompt.count("<image>") != 1: raise TitleMaskError(f"{row.get('id')}: invalid prompt")
    parts = prompt.split("<image>")
    prompt_tokens = len(_ids(tokenizer, parts[0])) + len(_ids(tokenizer, parts[1])) + len(images) * visual_tokens_per_image
    target_tokens = len(_ids(tokenizer, target)); total = 2 + prompt_tokens + target_tokens
    meta = row.get("meta", {})
    recipe_pool = meta.get("recipe_mix_pool")
    legacy_pool = meta.get("mix_pool")
    pool = recipe_pool if isinstance(recipe_pool, str) and recipe_pool else legacy_pool
    return {"id": row.get("id"), "source": meta.get("source"), "doc_id": meta.get("doc_id"),
            "split": meta.get("split"), "mix_pool": pool, "images": len(images),
            "visual_tokens": len(images) * visual_tokens_per_image, "prompt_tokens": prompt_tokens, "target_tokens": target_tokens,
            "total_tokens": total, "bucket": bucket_for(total), "target_sha256": sha256_text(target)}
def scan_rows(rows, tokenizer, max_length):
    if not isinstance(max_length, int) or max_length <= 0: raise TitleMaskError("max_length must be positive")
    validate_tokenizer(tokenizer); out = []
    for row in rows:
        audit = count_row_tokens(row, tokenizer); audit["fits"] = audit["total_tokens"] <= max_length; out.append((row, audit))
    return out
def _stats(entries):
    vals = sorted(int(x["total_tokens"]) for x in entries); targets = sorted(int(x["target_tokens"]) for x in entries)
    def s(v):
        if not v: return {"min":0,"p50":0,"p90":0,"p95":0,"p99":0,"max":0,"mean":0,"sum":0}
        q=lambda p: v[min(len(v)-1,int((len(v)-1)*p))]
        return {"min":v[0],"p50":q(.5),"p90":q(.9),"p95":q(.95),"p99":q(.99),"max":v[-1],"mean":sum(v)/len(v),"sum":sum(v)}
    return {"rows":len(entries),"images":sum(x["images"] for x in entries),"target_tokens":s(targets),"total_tokens":s(vals),"buckets":dict(sorted(Counter(x["bucket"] for x in entries).items()))}
def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n")
def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input-dir",type=Path,required=True); p.add_argument("--model",required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--max-length",type=int,default=32768); p.add_argument("--dry-run",action="store_true"); p.add_argument("--overwrite",action="store_true"); return p.parse_args()
def _tokenizer_info(t):
    files={}; base=getattr(t,"name_or_path",None)
    if base and Path(base).is_dir():
        for n in ("tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.json","merges.txt"):
            if (Path(base)/n).is_file(): files[n]=file_sha256(Path(base)/n)
    return {"class":t.__class__.__name__,"name_or_path":base,"bos_token_id":t.bos_token_id,"eos_token_id":t.eos_token_id,"image_token_id":_ids(t, "<image>")[0],"files_sha256":files}

def build_outputs(input_dir, output_dir, tokenizer, max_length=32768, *, dry_run=False, overwrite=False, model="injected"):
    input_dir = Path(input_dir).resolve()
    inputs = {s: input_dir / f"{s}.jsonl" for s in ("train", "validation", "test")}
    validate_tokenizer(tokenizer)
    scanned = {}
    for split, path in inputs.items():
        items = []
        for _, row in read_jsonl(path):
            validate_training_row(row, expected_split=split, check_images=False)
            audit = count_row_tokens(row, tokenizer)
            pool = audit["mix_pool"]
            if not isinstance(pool, str) or not pool:
                raise TitleMaskError(
                    f"{row.get('id')}: meta.recipe_mix_pool or meta.mix_pool must be a non-empty string"
                )
            if audit["target_sha256"] != row["meta"].get("title_target_sha256"):
                raise TitleMaskError(f"{row.get('id')}: target SHA mismatch")
            audit["fits"] = audit["total_tokens"] <= max_length
            items.append((row, audit))
        scanned[split] = items
    pool_names = sorted({a["mix_pool"] for items in scanned.values() for _, a in items})
    report = {"schema_version": SCHEMA_VERSION, "model": model, "max_length": max_length,
              "bucket_definitions": {name: limit for limit, name in BUCKETS} | {"overflow": ">32768"},
              "length_contract": {
                  "tested_ms_swift_commit": PINNED_MS_SWIFT_COMMIT,
                  "template": "unlimited_ocr",
                  "deepseek_version": "v1",
                  "crop_mode": False,
                  "image_size": 1024,
                  "base_size": 1024,
                  "visual_tokens_per_image": VISUAL_TOKENS_PER_IMAGE,
                  "formula": "BOS + tokenize(prompt text around expanded image placeholders) + tokenize(response) + EOS",
                  "scope": "Pinned unlimited_ocr v1 no-crop 1024 contract; not a universal model constant.",
                  "pool_field_priority": ["meta.recipe_mix_pool", "meta.mix_pool"],
              },
              "visual_tokens_per_image": VISUAL_TOKENS_PER_IMAGE,
              "tokenizer": _tokenizer_info(tokenizer),
              "input_sha256": {split: file_sha256(path) for split, path in inputs.items()},
              "splits": {}}
    manifests = []
    for split, items in scanned.items():
        audits = [dict(a, input_split=split) for _, a in items]
        manifests.extend(audits)
        report["splits"][split] = dict(_stats(audits),
            fit=sum(a["fits"] for a in audits), overflow=sum(not a["fits"] for a in audits),
            pools={p: _stats([a for a in audits if a["mix_pool"] == p]) for p in pool_names},
            sources={s: _stats([a for a in audits if a["source"] == s])
                     for s in sorted({a["source"] for a in audits})})
    all_audits = [a for items in scanned.values() for _, a in items]
    report["overall"] = dict(_stats(all_audits),
        fit=sum(a["fits"] for a in all_audits), overflow=sum(not a["fits"] for a in all_audits),
        pools={p: _stats([a for a in all_audits if a["mix_pool"] == p]) for p in pool_names},
        sources={s: _stats([a for a in all_audits if a["source"] == s])
                 for s in sorted({a["source"] for a in all_audits})})
    if dry_run:
        report["dry_run"] = True
        return report
    output_dir = Path(output_dir).resolve()
    if output_dir == input_dir or output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
        raise TitleMaskError("output and input must be separate non-nested paths")
    if output_dir.exists() and not overwrite:
        raise TitleMaskError(f"output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=output_dir.parent))
    try:
        for name in ("fit", "overflow", "buckets", "smoke"):
            (temp / name).mkdir()
        for split, items in scanned.items():
            fits = [r for r, a in items if a["fits"]]
            overflow = [r for r, a in items if not a["fits"]]
            if not fits:
                raise TitleMaskError(f"{split} has no fitting rows")
            _write(temp / "fit" / f"{split}.jsonl", fits)
            _write(temp / "overflow" / f"{split}.jsonl", overflow)
            for _, name in BUCKETS + ((10**18, "overflow"),):
                _write(temp / "buckets" / f"{name}_{split}.jsonl",
                       [r for r, a in items if a["bucket"] == name])
            best = max((pair for pair in items if pair[1]["fits"]),
                       key=lambda pair: pair[1]["total_tokens"])[0]
            _write(temp / "smoke" / f"{split}.jsonl", [best])
        _write(temp / "length_manifest.jsonl", manifests)
        report["dry_run"] = False
        (temp / "report.json").write_text(json.dumps(report, ensure_ascii=False,
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
        publish_directory(temp, output_dir, overwrite)
    except Exception:
        import shutil
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return report
def main():
    a=parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as e: raise TitleMaskError("transformers is required") from e
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True, trust_remote_code=True)
    report = build_outputs(a.input_dir, a.output_dir, tokenizer, max_length=a.max_length,
                           dry_run=a.dry_run, overwrite=a.overwrite, model=a.model)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
if __name__=="__main__":
    try: main()
    except (TitleMaskError,OSError,ValueError) as e: print(f"ERROR: {e}",file=sys.stderr); raise SystemExit(2)
