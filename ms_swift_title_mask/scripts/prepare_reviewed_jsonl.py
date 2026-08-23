#!/usr/bin/env python3
"""Build ms-swift JSONL from explicit human- or rules-accepted decisions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

_BUNDLE_PARENT = Path(__file__).resolve().parents[2]
if str(_BUNDLE_PARENT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_PARENT))

from ms_swift_title_mask.core import (  # noqa: E402
    ACCEPTED_TITLE_STATUSES,
    ALLOWED_SPLITS,
    TitleMaskError,
)
from ms_swift_title_mask.data_contract import (  # noqa: E402
    build_reviewed_row,
    dump_jsonl,
    file_sha256,
    publish_directory,
    read_jsonl,
    require_nonempty_string,
    validate_review_record,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--skip-image-existence", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.source_jsonl.resolve()
    review_path = args.review_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir in (source_path, review_path) or output_dir.is_relative_to(source_path) or output_dir.is_relative_to(review_path):
        raise TitleMaskError("output directory cannot be inside an input file path")

    reviews = {}
    review_ids = set()
    accepted_review_ids = set()
    status_counts = Counter()
    for line_number, raw_review in read_jsonl(review_path):
        review = validate_review_record(raw_review)
        row_id = review["id"]
        if row_id in reviews:
            raise TitleMaskError(f"{review_path}:{line_number}: duplicate review id {row_id!r}")
        if review.get("review_id") in review_ids:
            raise TitleMaskError(f"{review_path}:{line_number}: duplicate review_id {review['review_id']!r}")
        reviews[row_id] = review
        status_counts[review["status"]] += 1
        if review["status"] in ACCEPTED_TITLE_STATUSES:
            review_ids.add(review["review_id"])
            accepted_review_ids.add(row_id)

    if not accepted_review_ids:
        raise TitleMaskError("review manifest contains 0 accepted rows; refusing to build training data")

    rows_by_split = {split: [] for split in ALLOWED_SPLITS}
    source_ids = set()
    source_count = 0
    for line_number, source in read_jsonl(source_path):
        source_count += 1
        row_id = require_nonempty_string(source.get("id"), f"{source_path}:{line_number}.id")
        if row_id in source_ids:
            raise TitleMaskError(f"{source_path}:{line_number}: duplicate source id {row_id!r}")
        source_ids.add(row_id)
        review = reviews.get(row_id)
        if review is None or review["status"] not in ACCEPTED_TITLE_STATUSES:
            continue
        built = build_reviewed_row(
            source,
            review,
            image_root=args.image_root,
            check_images=not args.skip_image_existence,
        )
        rows_by_split[review["split"]].append(built)

    missing_sources = sorted(accepted_review_ids - source_ids)
    if missing_sources:
        raise TitleMaskError(f"accepted reviews missing from source JSONL: {missing_sources[:10]}")
    empty_splits = sorted(split for split, rows in rows_by_split.items() if not rows)
    if empty_splits:
        raise TitleMaskError(f"accepted output has empty required splits: {empty_splits}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    try:
        output_counts = {}
        for split in sorted(ALLOWED_SPLITS):
            output_counts[split] = dump_jsonl(temp_dir / f"{split}.jsonl", rows_by_split[split])
        report = {
            "schema_version": "uocr-reviewed-title-v1",
            "source_jsonl": str(source_path),
            "source_sha256": file_sha256(source_path),
            "review_manifest": str(review_path),
            "review_manifest_sha256": file_sha256(review_path),
            "source_rows": source_count,
            "review_status_counts": dict(sorted(status_counts.items())),
            "output_counts": output_counts,
            "accepted_total": sum(output_counts.values()),
        }
        (temp_dir / "build_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_directory(temp_dir, output_dir, args.overwrite)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TitleMaskError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
