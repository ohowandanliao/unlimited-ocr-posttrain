#!/usr/bin/env python3
"""Reproduce the historical PMC title-silver JSONL snapshot.

Title acceptance is not full-content acceptance: this builder does not consume
the PMC content quarantine and its output must not be used by current recipes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
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
from ms_swift_title_mask.pmc_silver import (  # noqa: E402
    PMC_SILVER_RULESET_VERSION,
    render_pmc_document,
    stable_page_choice,
    stable_split,
)


REVIEWED_AT = "2026-08-23T00:00:00+08:00"
SOURCE_NAME = "PMC-v26-synthetic"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", default="uocr-pmc-document-split-20260823-v1")
    parser.add_argument("--single-seed", default="uocr-pmc-single-page-20260823-v1")
    parser.add_argument("--skip-image-existence", action="store_true")
    parser.add_argument("--skip-pdf-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_unique_by_id(path: Path, field: str):
    result = {}
    for line_number, row in read_jsonl(path):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise TitleMaskError(f"{path}:{line_number}: missing {field}")
        if value in result:
            raise TitleMaskError(f"{path}:{line_number}: duplicate {field} {value!r}")
        result[value] = row
    return result


def _load_candidates(path: Path):
    result = defaultdict(list)
    seen_ids = set()
    for line_number, row in read_jsonl(path):
        candidate_id = row.get("candidate_id")
        doc_id = row.get("doc_id")
        if not isinstance(candidate_id, str) or not isinstance(doc_id, str):
            raise TitleMaskError(f"{path}:{line_number}: invalid candidate identity")
        if candidate_id in seen_ids:
            raise TitleMaskError(f"{path}:{line_number}: duplicate candidate_id {candidate_id!r}")
        seen_ids.add(candidate_id)
        result[doc_id].append(row)
    return result, len(seen_ids)


def _assert_separate_paths(input_root: Path, audit_root: Path, output_dir: Path):
    for source in (input_root, audit_root):
        if output_dir == source or output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
            raise TitleMaskError("output, raw input, and audit roots must be separate non-nested paths")


def _row(
    *,
    row_id: str,
    doc_id: str,
    split: str,
    target: str,
    images: list[str],
    sample_form: str,
    page_indices: list[int],
    source_sha256: str,
    pdf_sha256: str,
    operation_counts: Counter,
    reason_counts: Counter,
):
    target = canonicalize_target(target)
    prompt = SINGLE_PAGE_PROMPT if len(images) == 1 else MULTI_PAGE_PROMPT
    target_sha = sha256_text(target)
    return {
        "id": row_id,
        "channel": "title_reviewed",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "images": images,
        "meta": {
            "source": SOURCE_NAME,
            "doc_id": doc_id,
            "sample_form": sample_form,
            "page_indices": page_indices,
            "n_pages": len(images),
            "split": split,
            "title_review_status": SILVER_ACCEPTED,
            "title_review_id": f"{row_id}:{PMC_SILVER_RULESET_VERSION}",
            "title_review_version": PMC_SILVER_RULESET_VERSION,
            "title_reviewer": "deterministic-silver-rules",
            "title_reviewed_at": REVIEWED_AT,
            "title_target_sha256": target_sha,
            "title_heading_count": heading_count(target),
            "title_ruleset_version": PMC_SILVER_RULESET_VERSION,
            "title_source_sha256": source_sha256,
            "source_pdf_sha256": pdf_sha256,
            "silver_operation_counts": dict(sorted(operation_counts.items())),
            "silver_reason_counts": dict(sorted(reason_counts.items())),
        },
    }


def _image_paths(image_root: Path, doc_id: str, page_indices: list[int], check_exists: bool):
    paths = [str((image_root / "pmc" / doc_id / f"page_{page_idx:04d}.png").resolve()) for page_idx in page_indices]
    if check_exists:
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise TitleMaskError(f"{doc_id}: missing rendered page image: {missing[0]}")
    return paths


def _write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    audit_root = args.audit_root.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    _assert_separate_paths(input_root, audit_root, output_dir)

    manifest_path = audit_root / "document_manifest.jsonl"
    candidates_path = audit_root / "title_candidates.jsonl"
    manifests = _load_unique_by_id(manifest_path, "doc_id")
    candidates_by_doc, candidate_count = _load_candidates(candidates_path)
    if set(manifests) != set(candidates_by_doc):
        missing = sorted(set(manifests) ^ set(candidates_by_doc))
        raise TitleMaskError(f"PMC manifest/candidate document mismatch: {missing[:10]}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    split_counts = {"pmc_full": Counter(), "pmc_single": Counter()}
    total_operations = Counter()
    total_reasons = Counter()
    total_pages = 0
    try:
        for pool in ("pmc_full", "pmc_single"):
            (temp_dir / pool).mkdir(parents=True)
        (temp_dir / "ground_truth" / "full").mkdir(parents=True)
        (temp_dir / "ground_truth" / "single").mkdir(parents=True)

        with ExitStack() as stack:
            handles = {
                (pool, split): stack.enter_context((temp_dir / pool / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n"))
                for pool in ("pmc_full", "pmc_single")
                for split in sorted(ALLOWED_SPLITS)
            }
            decisions_handle = stack.enter_context(
                (temp_dir / "silver_decisions.jsonl").open("w", encoding="utf-8", newline="\n")
            )

            for doc_id in sorted(manifests):
                manifest = manifests[doc_id]
                middle_path = (input_root / manifest["source_middle_json"]).resolve()
                pdf_path = (input_root / manifest["source_pdf"]).resolve()
                if not middle_path.is_relative_to(input_root) or not pdf_path.is_relative_to(input_root):
                    raise TitleMaskError(f"{doc_id}: source path escapes input root")
                source_sha = file_sha256(middle_path)
                if source_sha != manifest.get("source_sha256"):
                    raise TitleMaskError(f"{doc_id}: middle.json SHA mismatch")
                pdf_sha = manifest.get("pdf_sha256")
                if not isinstance(pdf_sha, str) or len(pdf_sha) != 64:
                    raise TitleMaskError(f"{doc_id}: invalid PDF SHA in audit manifest")
                if not args.skip_pdf_hash and file_sha256(pdf_path) != pdf_sha:
                    raise TitleMaskError(f"{doc_id}: PDF SHA mismatch")
                for candidate in candidates_by_doc[doc_id]:
                    if candidate.get("source_sha256") != source_sha:
                        raise TitleMaskError(f"{doc_id}: candidate source SHA mismatch")

                with middle_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                rendered_pages, decisions = render_pmc_document(payload, candidates_by_doc[doc_id])
                if len(rendered_pages) != manifest.get("pages"):
                    raise TitleMaskError(f"{doc_id}: rendered page count mismatch")
                full_target = canonicalize_target("\n\n".join(page for page in rendered_pages if page.strip()))
                if heading_count(full_target) <= 0:
                    raise TitleMaskError(f"{doc_id}: full target contains no ATX heading")

                split = stable_split(args.split_seed, SOURCE_NAME, doc_id)
                page_indices = list(range(len(rendered_pages)))
                full_images = _image_paths(image_root, doc_id, page_indices, not args.skip_image_existence)
                operation_counts = Counter(decision["operation"] for decision in decisions)
                reason_counts = Counter(reason for decision in decisions for reason in decision["reason_codes"])
                total_operations.update(operation_counts)
                total_reasons.update(reason_counts)
                total_pages += len(rendered_pages)

                full_id = f"{doc_id}__full"
                full_row = _row(
                    row_id=full_id,
                    doc_id=doc_id,
                    split=split,
                    target=full_target,
                    images=full_images,
                    sample_form="full_document",
                    page_indices=page_indices,
                    source_sha256=source_sha,
                    pdf_sha256=pdf_sha,
                    operation_counts=operation_counts,
                    reason_counts=reason_counts,
                )
                validate_training_row(full_row, expected_split=split, check_images=not args.skip_image_existence)
                _write_row(handles[("pmc_full", split)], full_row)
                split_counts["pmc_full"][split] += 1
                (temp_dir / "ground_truth" / "full" / f"{doc_id}.md").write_text(full_target, encoding="utf-8")

                single_page = stable_page_choice(args.single_seed, doc_id, rendered_pages, heading_count)
                single_target = rendered_pages[single_page]
                single_id = f"{doc_id}__p{single_page:04d}"
                single_images = _image_paths(image_root, doc_id, [single_page], not args.skip_image_existence)
                single_row = _row(
                    row_id=single_id,
                    doc_id=doc_id,
                    split=split,
                    target=single_target,
                    images=single_images,
                    sample_form="strict_single_page",
                    page_indices=[single_page],
                    source_sha256=source_sha,
                    pdf_sha256=pdf_sha,
                    operation_counts=operation_counts,
                    reason_counts=reason_counts,
                )
                validate_training_row(single_row, expected_split=split, check_images=not args.skip_image_existence)
                _write_row(handles[("pmc_single", split)], single_row)
                split_counts["pmc_single"][split] += 1
                (temp_dir / "ground_truth" / "single" / f"{doc_id}__p{single_page:04d}.md").write_text(
                    single_target,
                    encoding="utf-8",
                )

                for decision in decisions:
                    record = {
                        "schema_version": "pmc-title-silver-decision-v1",
                        "ruleset_version": PMC_SILVER_RULESET_VERSION,
                        "doc_id": doc_id,
                        "source_sha256": source_sha,
                        **decision,
                    }
                    _write_row(decisions_handle, record)

        for pool, counts in split_counts.items():
            empty = sorted(split for split in ALLOWED_SPLITS if counts[split] == 0)
            if empty:
                raise TitleMaskError(f"{pool} has empty splits: {empty}")
        report = {
            "schema_version": "pmc-title-silver-build-report-v1",
            "ruleset_version": PMC_SILVER_RULESET_VERSION,
            "source_name": SOURCE_NAME,
            "input_root": str(input_root),
            "audit_root": str(audit_root),
            "image_root": str(image_root),
            "input_sha256": {
                "document_manifest": file_sha256(manifest_path),
                "title_candidates": file_sha256(candidates_path),
            },
            "documents": len(manifests),
            "pages": total_pages,
            "candidates": candidate_count,
            "split_counts": {pool: dict(sorted(counts.items())) for pool, counts in split_counts.items()},
            "operation_counts": dict(sorted(total_operations.items())),
            "reason_counts": dict(sorted(total_reasons.items())),
            "single_page_policy": "one deterministic heading-bearing page per PMC document",
            "pdf_sha_checked": not args.skip_pdf_hash,
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
