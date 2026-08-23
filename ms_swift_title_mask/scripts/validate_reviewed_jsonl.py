#!/usr/bin/env python3
"""Validate reviewed-title training JSONL before ms-swift sees it."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_BUNDLE_PARENT = Path(__file__).resolve().parents[2]
if str(_BUNDLE_PARENT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_PARENT))

from ms_swift_title_mask.core import ALLOWED_SPLITS, TitleMaskError  # noqa: E402
from ms_swift_title_mask.data_contract import read_jsonl, validate_training_row  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="+")
    parser.add_argument("--skip-image-existence", action="store_true")
    return parser.parse_args()


def inferred_split(path: Path):
    return path.stem if path.stem in ALLOWED_SPLITS else None


def main() -> int:
    args = parse_args()
    seen_ids = set()
    counts = Counter()
    headings = 0
    images = 0
    chars = 0
    for path_arg in args.jsonl:
        path = path_arg.resolve()
        file_count = 0
        for line_number, row in read_jsonl(path):
            try:
                summary = validate_training_row(
                    row,
                    expected_split=inferred_split(path),
                    check_images=not args.skip_image_existence,
                )
            except TitleMaskError as exc:
                raise TitleMaskError(f"{path}:{line_number}: {exc}") from exc
            if summary["id"] in seen_ids:
                raise TitleMaskError(f"{path}:{line_number}: duplicate id across inputs: {summary['id']}")
            seen_ids.add(summary["id"])
            counts[summary["split"]] += 1
            headings += summary["headings"]
            images += summary["images"]
            chars += summary["target_chars"]
            file_count += 1
        if file_count == 0:
            raise TitleMaskError(f"empty JSONL is not trainable: {path}")

    result = {
        "rows": len(seen_ids),
        "split_counts": dict(sorted(counts.items())),
        "headings": headings,
        "images": images,
        "target_chars": chars,
        "image_existence_checked": not args.skip_image_existence,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TitleMaskError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
