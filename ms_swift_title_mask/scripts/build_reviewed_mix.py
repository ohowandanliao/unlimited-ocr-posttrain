#!/usr/bin/env python3
"""Merge accepted READoc-full, PMC-full, and paired PMC-single rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

_BUNDLE_PARENT = Path(__file__).resolve().parents[2]
if str(_BUNDLE_PARENT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_PARENT))

from ms_swift_title_mask.core import ALLOWED_SPLITS, TitleMaskError  # noqa: E402
from ms_swift_title_mask.data_contract import (  # noqa: E402
    dump_jsonl,
    file_sha256,
    publish_directory,
    read_jsonl,
    require_nonempty_string,
    validate_training_row,
)


MIX_VERSION = "silver-readoc-full-pmc-full-single-natural-v1"
POOL_FAMILIES = {"readoc_full": "readoc", "pmc_full": "pmc", "pmc_single": "pmc"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readoc-dir", type=Path, required=True)
    parser.add_argument("--pmc-full-dir", type=Path, required=True)
    parser.add_argument("--pmc-single-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="uocr-reviewed-mix-20260823-v2")
    parser.add_argument("--skip-image-existence", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_key(seed: str, *parts: str) -> str:
    return hashlib.sha256(":".join((seed, *parts)).encode("utf-8")).hexdigest()


def document_key(row, family: str):
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise TitleMaskError(f"{row.get('id')}: meta must be an object")
    source = require_nonempty_string(meta.get("source"), f"{row.get('id')}.meta.source")
    doc_id = require_nonempty_string(meta.get("doc_id"), f"{row.get('id')}.meta.doc_id")
    return family, source, doc_id


def load_pool(root: Path, pool_name: str, check_images: bool):
    family = POOL_FAMILIES[pool_name]
    by_split = {}
    paths = {}
    for split in sorted(ALLOWED_SPLITS):
        path = (root / f"{split}.jsonl").resolve()
        paths[split] = path
        rows = []
        seen_documents = set()
        for line_number, row in read_jsonl(path):
            try:
                summary = validate_training_row(
                    row,
                    expected_split=split,
                    check_images=check_images,
                )
                key = document_key(row, family)
            except TitleMaskError as exc:
                raise TitleMaskError(f"{path}:{line_number}: {exc}") from exc
            if key in seen_documents:
                raise TitleMaskError(
                    f"{path}:{line_number}: {pool_name} contains more than one row for document {key}"
                )
            seen_documents.add(key)
            sample_form = row["meta"].get("sample_form")
            if pool_name in {"readoc_full", "pmc_full"} and sample_form != "full_document":
                raise TitleMaskError(f"{path}:{line_number}: {pool_name} requires sample_form=full_document")
            if pool_name == "pmc_single" and (
                sample_form != "strict_single_page" or summary["images"] != 1
            ):
                raise TitleMaskError(
                    f"{path}:{line_number}: pmc_single requires strict_single_page with one image"
                )
            rows.append((row, summary, key))
        if not rows:
            raise TitleMaskError(f"{pool_name} {split} pool is empty: {path}")
        by_split[split] = rows
    return by_split, paths


def audit_global_identity(pools):
    seen_ids = {}
    document_splits = {}
    for pool_name, by_split in pools.items():
        for split, entries in by_split.items():
            for row, _summary, key in entries:
                row_id = row["id"]
                location = f"{pool_name}/{split}"
                if row_id in seen_ids:
                    raise TitleMaskError(
                        f"duplicate row id {row_id!r} in {seen_ids[row_id]} and {location}"
                    )
                seen_ids[row_id] = location
                previous_split = document_splits.setdefault(key, split)
                if previous_split != split:
                    raise TitleMaskError(
                        f"document leakage: {key} appears in both {previous_split} and {split}"
                    )
    for split in sorted(ALLOWED_SPLITS):
        full_docs = {key for _row, _summary, key in pools["pmc_full"][split]}
        single_docs = {key for _row, _summary, key in pools["pmc_single"][split]}
        if full_docs != single_docs:
            mismatch = sorted(full_docs ^ single_docs)
            raise TitleMaskError(
                f"PMC full/single document mismatch in {split}: {mismatch[:10]}"
            )


def annotate(row, pool_name):
    output = copy.deepcopy(row)
    output["meta"]["mix_pool"] = pool_name
    output["meta"]["mix_version"] = MIX_VERSION
    return output


def build_split(pools, split, seed):
    combined = []
    pool_stats = {}
    for pool_name in sorted(pools):
        entries = pools[pool_name][split]
        pool_stats[pool_name] = {
            "rows": len(entries),
            "images": sum(summary["images"] for _row, summary, _key in entries),
            "target_chars": sum(summary["target_chars"] for _row, summary, _key in entries),
        }
        combined.extend(annotate(row, pool_name) for row, _summary, _key in entries)

    combined.sort(key=lambda row: stable_key(seed, split, str(row["id"])))
    total_rows = len(combined)
    total_images = sum(stats["images"] for stats in pool_stats.values())
    total_chars = sum(stats["target_chars"] for stats in pool_stats.values())
    for stats in pool_stats.values():
        stats["row_share"] = stats["rows"] / total_rows
        stats["image_share"] = stats["images"] / total_images if total_images else 0.0
        stats["target_char_share"] = stats["target_chars"] / total_chars if total_chars else 0.0
    return combined, {
        "rows": total_rows,
        "images": total_images,
        "target_chars": total_chars,
        "pools": pool_stats,
    }


def main():
    args = parse_args()
    check_images = not args.skip_image_existence
    readoc, readoc_paths = load_pool(args.readoc_dir.resolve(), "readoc_full", check_images)
    pmc_full, pmc_full_paths = load_pool(args.pmc_full_dir.resolve(), "pmc_full", check_images)
    pmc_single, pmc_single_paths = load_pool(args.pmc_single_dir.resolve(), "pmc_single", check_images)
    pools = {
        "readoc_full": readoc,
        "pmc_full": pmc_full,
        "pmc_single": pmc_single,
    }
    audit_global_identity(pools)

    output_dir = args.output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    try:
        split_reports = {}
        for split in sorted(ALLOWED_SPLITS):
            rows, split_report = build_split(pools, split, args.seed)
            dump_jsonl(temp_dir / f"{split}.jsonl", rows)
            split_reports[split] = split_report
        input_paths = {
            "readoc_full": readoc_paths,
            "pmc_full": pmc_full_paths,
            "pmc_single": pmc_single_paths,
        }
        report = {
            "schema_version": "uocr-reviewed-mix-report-v2",
            "mix_version": MIX_VERSION,
            "sampling": "natural_union_one_readoc_full_plus_one_pmc_full_and_single_per_document",
            "seed": args.seed,
            "input_sha256": {
                pool: {split: file_sha256(path) for split, path in paths.items()}
                for pool, paths in input_paths.items()
            },
            "splits": split_reports,
        }
        (temp_dir / "mix_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_directory(temp_dir, output_dir, args.overwrite)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (TitleMaskError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
