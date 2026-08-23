#!/usr/bin/env python3
"""Freeze READoc rule-cleaned SILVER_CANDIDATE documents into ms-swift JSONL."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

_BUNDLE_PARENT = Path(__file__).resolve().parents[2]
if str(_BUNDLE_PARENT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_PARENT))

from ms_swift_title_mask.core import (  # noqa: E402
    ALLOWED_SPLITS,
    MULTI_PAGE_PROMPT,
    SILVER_ACCEPTED,
    SINGLE_PAGE_PROMPT,
    TitleMaskError,
    canonicalize_target,
    heading_count,
    sha256_text,
)
from ms_swift_title_mask.data_contract import (  # noqa: E402
    file_sha256,
    publish_directory,
    read_jsonl,
    validate_training_row,
)
from ms_swift_title_mask.pmc_silver import stable_split  # noqa: E402


REVIEWED_AT = "2026-08-23T00:00:00+08:00"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", default="uocr-readoc-document-split-20260823-v1")
    parser.add_argument("--skip-image-existence", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path):
    rows = {}
    for line_number, row in read_jsonl(path):
        source = row.get("source")
        doc_id = row.get("doc_id")
        if not isinstance(source, str) or not isinstance(doc_id, str):
            raise TitleMaskError(f"{path}:{line_number}: invalid READoc identity")
        key = (source, doc_id)
        if key in rows:
            raise TitleMaskError(f"{path}:{line_number}: duplicate READoc document {key}")
        rows[key] = row
    return rows


def _assert_separate_paths(gt_root: Path, output_dir: Path):
    if output_dir == gt_root or output_dir.is_relative_to(gt_root) or gt_root.is_relative_to(output_dir):
        raise TitleMaskError("output and READoc GT roots must be separate non-nested paths")


def _image_paths(image_root: Path, source: str, doc_id: str, page_count: int, check_exists: bool):
    paths = [
        str((image_root / "readoc" / source / doc_id / f"page_{page_idx:04d}.png").resolve())
        for page_idx in range(page_count)
    ]
    if check_exists:
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise TitleMaskError(f"READoc {source}/{doc_id}: missing rendered image {missing[0]}")
    return paths


def _write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    args = parse_args()
    gt_root = args.gt_root.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    _assert_separate_paths(gt_root, output_dir)

    manifest_path = gt_root / "document_manifest.jsonl"
    summary_path = gt_root / "summary.json"
    validation_path = gt_root / "validation.json"
    manifests = _load_manifest(manifest_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("valid") is not True:
        raise TitleMaskError("READoc title GT validation is not valid")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    split_counts = Counter()
    source_counts = Counter()
    excluded_counts = Counter()
    pages_total = 0
    headings_total = 0
    try:
        (temp_dir / "readoc_full").mkdir(parents=True)
        (temp_dir / "ground_truth" / "arxiv").mkdir(parents=True)
        (temp_dir / "ground_truth" / "github").mkdir(parents=True)
        with ExitStack() as stack:
            handles = {
                split: stack.enter_context(
                    (temp_dir / "readoc_full" / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n")
                )
                for split in sorted(ALLOWED_SPLITS)
            }
            frozen_manifest = stack.enter_context(
                (temp_dir / "silver_manifest.jsonl").open("w", encoding="utf-8", newline="\n")
            )
            for (source, doc_id), manifest in sorted(manifests.items()):
                profile = manifest.get("title_profile_status")
                if profile != "SILVER_CANDIDATE":
                    excluded_counts[str(profile)] += 1
                    continue
                relative_gt = manifest.get("derived_gt")
                if not isinstance(relative_gt, str):
                    raise TitleMaskError(f"READoc {source}/{doc_id}: missing derived_gt")
                gt_path = (gt_root / relative_gt).resolve()
                if not gt_path.is_relative_to(gt_root):
                    raise TitleMaskError(f"READoc {source}/{doc_id}: GT path escapes root")
                if file_sha256(gt_path) != manifest.get("derived_sha256"):
                    raise TitleMaskError(f"READoc {source}/{doc_id}: derived GT SHA mismatch")
                target = canonicalize_target(gt_path.read_text(encoding="utf-8"))
                target_headings = heading_count(target)
                if target_headings <= 0:
                    raise TitleMaskError(f"READoc {source}/{doc_id}: silver target has no ATX heading")
                page_count = manifest.get("pdf_pages")
                if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count <= 0:
                    raise TitleMaskError(f"READoc {source}/{doc_id}: invalid PDF page count")
                images = _image_paths(
                    image_root,
                    source,
                    doc_id,
                    page_count,
                    not args.skip_image_existence,
                )
                source_name = f"READoc-{source}"
                split = stable_split(args.split_seed, source_name, doc_id)
                row_id = f"readoc_{source}_{doc_id}__full"
                ruleset = manifest.get("ruleset_version")
                source_sha = manifest.get("source_sha256")
                if not isinstance(ruleset, str) or not isinstance(source_sha, str) or len(source_sha) != 64:
                    raise TitleMaskError(f"READoc {source}/{doc_id}: invalid silver provenance")
                row = {
                    "id": row_id,
                    "channel": "title_reviewed",
                    "messages": [
                        {"role": "user", "content": SINGLE_PAGE_PROMPT if page_count == 1 else MULTI_PAGE_PROMPT},
                        {"role": "assistant", "content": target},
                    ],
                    "images": images,
                    "meta": {
                        "source": source_name,
                        "doc_id": doc_id,
                        "sample_form": "full_document",
                        "page_indices": list(range(page_count)),
                        "n_pages": page_count,
                        "split": split,
                        "title_review_status": SILVER_ACCEPTED,
                        "title_review_id": f"{row_id}:{ruleset}",
                        "title_review_version": ruleset,
                        "title_reviewer": "deterministic-silver-rules",
                        "title_reviewed_at": REVIEWED_AT,
                        "title_target_sha256": sha256_text(target),
                        "title_heading_count": target_headings,
                        "title_ruleset_version": ruleset,
                        "title_source_sha256": source_sha,
                        "derived_gt_sha256": manifest["derived_sha256"],
                        "source_pdf_sha256": manifest.get("pdf_sha256"),
                        "accepted_change_count": manifest.get("accepted_change_count", 0),
                    },
                }
                validate_training_row(row, expected_split=split, check_images=not args.skip_image_existence)
                _write_row(handles[split], row)
                split_counts[split] += 1
                source_counts[source] += 1
                pages_total += page_count
                headings_total += target_headings
                shutil.copyfile(gt_path, temp_dir / "ground_truth" / source / f"{doc_id}.md")
                _write_row(
                    frozen_manifest,
                    {
                        "schema_version": "readoc-title-silver-manifest-v1",
                        "ruleset_version": ruleset,
                        "id": row_id,
                        "source": source_name,
                        "doc_id": doc_id,
                        "split": split,
                        "source_sha256": source_sha,
                        "derived_sha256": manifest["derived_sha256"],
                        "pdf_sha256": manifest.get("pdf_sha256"),
                        "pdf_pages": page_count,
                        "title_heading_count": target_headings,
                    },
                )

        empty = sorted(split for split in ALLOWED_SPLITS if split_counts[split] == 0)
        if empty:
            raise TitleMaskError(f"READoc silver output has empty splits: {empty}")
        report = {
            "schema_version": "readoc-title-silver-build-report-v1",
            "input_root": str(gt_root),
            "image_root": str(image_root),
            "input_sha256": {
                "document_manifest": file_sha256(manifest_path),
                "summary": file_sha256(summary_path),
                "validation": file_sha256(validation_path),
            },
            "accepted_profile": "SILVER_CANDIDATE",
            "documents": sum(split_counts.values()),
            "pages": pages_total,
            "headings": headings_total,
            "source_counts": dict(sorted(source_counts.items())),
            "split_counts": dict(sorted(split_counts.items())),
            "excluded_counts": dict(sorted(excluded_counts.items())),
            "image_existence_checked": not args.skip_image_existence,
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


if __name__ == "__main__":
    try:
        main()
    except (TitleMaskError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
