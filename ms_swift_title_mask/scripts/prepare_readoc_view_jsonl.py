#!/usr/bin/env python3
"""Normalize legacy READoc view JSONL into the current traceable schema."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms_swift_title_mask.core import (  # noqa: E402
    MULTI_PAGE_PROMPT,
    SILVER_ACCEPTED,
    TitleMaskError,
    canonicalize_target,
    heading_count,
    is_mineru_heading_prior_prompt,
    sha256_text,
)
from ms_swift_title_mask.data_contract import (  # noqa: E402
    dump_jsonl,
    file_sha256,
    publish_directory,
    read_jsonl,
    require_nonempty_string,
    resolve_images,
    validate_training_row,
)


RULESET_VERSION = "readoc-view-atx-v1"
REVIEWED_AT = "2026-08-26T00:00:00+08:00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--skip-image-existence", action="store_true")
    parser.add_argument(
        "--allow-title-prior",
        action="store_true",
        help="accept the strict MinerU heading-prior prompt and preserve it",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _input_row_sha(row: dict) -> str:
    return sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def normalize_row(
    row: dict,
    *,
    split: str,
    path: Path,
    line_number: int,
    image_root: Path | None,
    check_images: bool,
    allow_title_prior: bool = False,
) -> dict:
    row_id = require_nonempty_string(row.get("id"), f"{path}:{line_number}.id")
    source = require_nonempty_string(row.get("source"), f"{path}:{line_number}.source")
    if not source.startswith("READoc-"):
        raise TitleMaskError(f"{path}:{line_number}: expected a READoc source, got {source!r}")
    prompt = require_nonempty_string(row.get("prompt"), f"{path}:{line_number}.prompt")
    if prompt == MULTI_PAGE_PROMPT:
        input_prompt_variant = "no_prior"
    elif is_mineru_heading_prior_prompt(prompt):
        if not allow_title_prior:
            raise TitleMaskError(
                f"{path}:{line_number}: MinerU heading-prior input requires --allow-title-prior"
            )
        source_meta = row.get("meta")
        if not isinstance(source_meta, dict) or source_meta.get("prior") != "mineru_heading":
            raise TitleMaskError(
                f"{path}:{line_number}: MinerU heading-prior input requires "
                "meta.prior='mineru_heading'"
            )
        input_prompt_variant = "mineru_heading_prior"
    else:
        raise TitleMaskError(
            f"{path}:{line_number}: unsupported readoc view prompt; expected {MULTI_PAGE_PROMPT!r} "
            "or the strict MinerU heading-prior format"
        )
    if row.get("mode") != "multi_base" or row.get("task") != "multi_page_merge":
        raise TitleMaskError(f"{path}:{line_number}: expected multi_base/multi_page_merge view rows")
    if row.get("target_format") != "markdown":
        raise TitleMaskError(f"{path}:{line_number}: target_format must be markdown")

    source_meta = row.get("meta")
    if not isinstance(source_meta, dict):
        raise TitleMaskError(f"{path}:{line_number}: meta must be an object")
    doc_id = require_nonempty_string(source_meta.get("doc_id"), f"{path}:{line_number}.meta.doc_id")
    raw_target = require_nonempty_string(row.get("target"), f"{path}:{line_number}.target")
    target = canonicalize_target(raw_target)
    headings = heading_count(target)
    if headings <= 0:
        raise TitleMaskError(f"{path}:{line_number}: target has no CommonMark ATX heading")

    images = resolve_images(
        row.get("images"),
        image_root=image_root,
        check_exists=check_images,
    )
    if len(images) < 2:
        raise TitleMaskError(f"{path}:{line_number}: multi-page view must contain at least two images")
    n_pages = source_meta.get("n_pages", len(images))
    if not isinstance(n_pages, int) or isinstance(n_pages, bool) or n_pages != len(images):
        raise TitleMaskError(
            f"{path}:{line_number}: meta.n_pages must equal image count ({len(images)})"
        )

    input_sha = _input_row_sha(row)
    meta = dict(source_meta)
    meta.update(
        {
            "source": source,
            "doc_id": doc_id,
            "sample_form": "full_document",
            "page_indices": list(range(n_pages)),
            "n_pages": n_pages,
            "split": split,
            "recipe_mix_pool": "readoc_view",
            "source_view_schema": "legacy_readoc_view_v1",
            "input_prompt_variant": input_prompt_variant,
            "source_view_input_sha256": input_sha,
            "title_review_status": SILVER_ACCEPTED,
            "title_review_id": f"{row_id}:{RULESET_VERSION}",
            "title_review_version": RULESET_VERSION,
            "title_reviewer": "readoc-view-atx-normalizer",
            "title_reviewed_at": REVIEWED_AT,
            "title_target_sha256": sha256_text(target),
            "title_heading_count": headings,
            "title_ruleset_version": RULESET_VERSION,
            "title_source_sha256": input_sha,
            "source_target_sha256": sha256_text(raw_target),
        }
    )
    normalized = {
        "id": row_id,
        "channel": "title_reviewed",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "images": images,
        "meta": meta,
    }
    validate_training_row(normalized, expected_split=split, check_images=check_images)
    return normalized


def _write_rows(path: Path, rows: list[dict]) -> None:
    dump_jsonl(path, rows)


def main() -> int:
    args = parse_args()
    input_paths = {"train": args.train.resolve(), "validation": args.validation.resolve()}
    if args.test is not None:
        input_paths["test"] = args.test.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir in input_paths.values():
        raise TitleMaskError("output directory cannot be an input JSONL path")
    if output_dir.exists() and not args.overwrite:
        raise TitleMaskError(f"output already exists; pass --overwrite explicitly: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    seen_ids: set[str] = set()
    seen_documents: dict[tuple[str, str], str] = {}
    split_counts = Counter()
    headings = 0
    images = 0
    target_chars = 0
    prompt_variants = Counter()
    rows_by_split: dict[str, list[dict]] = {}
    try:
        for split, input_path in input_paths.items():
            rows: list[dict] = []
            for line_number, row in read_jsonl(input_path):
                normalized = normalize_row(
                    row,
                    split=split,
                    path=input_path,
                    line_number=line_number,
                    image_root=args.image_root,
                    check_images=not args.skip_image_existence,
                    allow_title_prior=args.allow_title_prior,
                )
                row_id = normalized["id"]
                if row_id in seen_ids:
                    raise TitleMaskError(f"{input_path}:{line_number}: duplicate row id {row_id!r}")
                seen_ids.add(row_id)
                meta = normalized["meta"]
                document_key = (meta["source"], meta["doc_id"])
                previous_split = seen_documents.setdefault(document_key, split)
                if previous_split != split:
                    raise TitleMaskError(
                        f"document leakage: {document_key} appears in {previous_split} and {split}"
                    )
                rows.append(normalized)
                split_counts[split] += 1
                headings += meta["title_heading_count"]
                images += len(normalized["images"])
                target_chars += len(normalized["messages"][1]["content"])
                prompt_variants[meta["input_prompt_variant"]] += 1
            if not rows:
                raise TitleMaskError(f"empty input is not trainable: {input_path}")
            rows_by_split[split] = rows

        for split, rows in rows_by_split.items():
            _write_rows(temp_dir / f"{split}.jsonl", rows)
        report = {
            "schema_version": "uocr-readoc-view-ablation-v1",
            "ruleset_version": RULESET_VERSION,
            "title_status": SILVER_ACCEPTED,
            "title_status_note": "deterministic CommonMark heading parse; not human review",
            "input_jsonl": {split: str(path) for split, path in input_paths.items()},
            "input_sha256": {split: file_sha256(path) for split, path in input_paths.items()},
            "output_counts": dict(sorted(split_counts.items())),
            "rows": len(seen_ids),
            "documents": len(seen_documents),
            "headings": headings,
            "images": images,
            "target_chars": target_chars,
            "target_rewrite": "canonicalize_target only; no content repair",
            "input_prompt_variants": dict(sorted(prompt_variants.items())),
        }
        (temp_dir / "build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_directory(temp_dir, output_dir, args.overwrite)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TitleMaskError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
