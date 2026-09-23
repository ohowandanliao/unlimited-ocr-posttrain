#!/usr/bin/env python3
"""Build the pmc-readoc-spage-16k mixed training jsonl.

Composition (v3, 2026-09-17 拍板):
  1. PMC full-document rows  : pmc-fullce-16k/train.jsonl            (9,284)
  2. PMC single-page rows    : pmc-spage-16k/train.jsonl             (~2,000, page-split, prompt=document parsing.)
  3. READoc rows             : data/readoc-view-16k/train.jsonl ×1   (1,706, no title-prior lineage)

validation/test are symlinks to the PMC 16k split files so eval_loss stays
comparable with the pmc-fullce-16k reference run. Recipe gate:
validate_training_recipe.py --recipe pmc_readoc_mix.

All input and output locations are explicit command-line arguments. The builder
refuses to reuse an existing output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pmc-dir",
        type=Path,
        default=os.environ.get(
            "PMC_FULLCE_DIR",
            "/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-fullce-16k",
        ),
    )
    ap.add_argument(
        "--spage-dir",
        type=Path,
        default=os.environ.get(
            "PMC_SPAGE_DIR",
            "/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-spage-16k",
        ),
    )
    ap.add_argument(
        "--readoc-train",
        type=Path,
        default=os.environ.get(
            "READOC_TRAIN_JSONL",
            "/home/jovyan/hyx/uocr-ms-swift-title-mask/data/readoc-view-16k/train.jsonl",
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=os.environ.get(
            "PMC_READOC_MIX_OUT",
            "/home/jovyan/hyx/dataset/pmc-fullce-data/pmc-readoc-spage-16k",
        ),
    )
    ap.add_argument("--skip-image-check", action="store_true")
    args = ap.parse_args()

    pmc_dir = args.pmc_dir.resolve()
    spage_dir = args.spage_dir.resolve()
    readoc_train = args.readoc_train.resolve()
    out = args.out_dir.resolve()
    if out.exists():
        print(f"ABORT: {out} already exists (never overwrite datasets)", file=sys.stderr)
        return 1
    parts = [
        ("pmc_full", pmc_dir / "train.jsonl"),
        ("pmc_spage", spage_dir / "train.jsonl"),
        ("readoc", readoc_train),
    ]
    required_inputs = [path for _, path in parts]
    required_inputs.extend([pmc_dir / "validation.jsonl", pmc_dir / "test.jsonl"])
    for p in required_inputs:
        if not p.is_file():
            print(f"ABORT: missing input {p}", file=sys.stderr)
            return 1

    out.mkdir(parents=True)
    seen_ids: set[str] = set()
    missing_imgs: list[tuple[str, str]] = []
    stats: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    prompts: Counter[str] = Counter()
    allowed_meta: set[str] | None = None   # 以 PMC 行 meta 键集为基准，跨源归一（HF datasets 要求同文件 schema 一致）
    dropped_meta: Counter[str] = Counter()

    total_lines = 0
    with open(out / "train.jsonl", "w", encoding="utf-8") as w:
        for name, path in parts:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)  # validate JSON before writing
                    if allowed_meta is None:
                        allowed_meta = set(row["meta"].keys())
                    extra = set(row["meta"].keys()) - allowed_meta
                    modified = False
                    if extra:
                        for k in extra:
                            dropped_meta[k] += 1
                        row["meta"] = {k: v for k, v in row["meta"].items() if k in allowed_meta}
                        modified = True
                    rid = row["id"]
                    if rid in seen_ids:
                        print(f"ABORT: duplicate id {rid} from {name}", file=sys.stderr)
                        return 1
                    seen_ids.add(rid)
                    if not args.skip_image_check:
                        for im in row["images"]:
                            if not os.path.isfile(im):
                                missing_imgs.append((rid, im))
                    stats[name] += 1
                    chars[name] += len(row["messages"][1]["content"])
                    prompts[row["messages"][0]["content"]] += 1
                    w.write((json.dumps(row, ensure_ascii=False) + "\n") if modified else (line if line.endswith("\n") else line + "\n"))
                    total_lines += 1

    os.symlink(pmc_dir / "validation.jsonl", out / "validation.jsonl")
    os.symlink(pmc_dir / "test.jsonl", out / "test.jsonl")

    total_chars = sum(chars.values())
    print(f"written {out/'train.jsonl'}: {total_lines} rows")
    print(f"rows by source: {dict(stats)}")
    print(f"prompts: {dict(prompts)}")
    print(f"chars by source: {dict(chars)}")
    print(
        "family char share: pmc={:.4f} readoc={:.4f}".format(
            (chars["pmc_full"] + chars["pmc_spage"]) / total_chars,
            chars["readoc"] / total_chars,
        )
    )
    print(f"single-page prompt rows: {prompts['<image>document parsing.']}")
    if missing_imgs:
        print(f"ABORT: {len(missing_imgs)} missing images, e.g. {missing_imgs[:3]}", file=sys.stderr)
        return 1
    print("image existence: all OK" if not args.skip_image_check else "image check skipped")
    print(f"meta keys normalized to PMC schema; dropped fields: {dict(dropped_meta)} (messages/images untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
