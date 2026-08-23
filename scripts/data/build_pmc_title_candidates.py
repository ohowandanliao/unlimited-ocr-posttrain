#!/usr/bin/env python3
"""Build deterministic PMC title candidates and a review queue from middle.json."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

from pmc_title_rules import (
    RULESET_VERSION,
    SCHEMA_VERSION,
    MiddleJsonError,
    analyze_document,
    sha256_file,
)


MANIFEST_SCHEMA_VERSION = "pmc-title-document-manifest-v1"
SUMMARY_SCHEMA_VERSION = "pmc-title-build-summary-v1"


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _discover_documents(input_root: Path, doc_ids: list[str] | None) -> list[Path]:
    documents_root = input_root / "documents"
    if not documents_root.is_dir():
        raise FileNotFoundError(f"missing documents directory: {documents_root}")
    if doc_ids:
        duplicates = sorted(doc_id for doc_id, count in Counter(doc_ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate --doc-id values: {', '.join(duplicates)}")
        paths = [documents_root / doc_id / "middle.json" for doc_id in sorted(doc_ids)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing requested middle.json: " + ", ".join(missing))
        return paths
    return sorted(documents_root.glob("*/middle.json"), key=lambda path: path.parent.name)


def _pdf_page_count(path: Path) -> int:
    try:
        import pymupdf
    except ImportError:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise MiddleJsonError(
                "PDF validation requires pymupdf (preferred) or pypdf"
            ) from exc
        try:
            page_count = len(PdfReader(str(path), strict=True).pages)
        except Exception as exc:
            raise MiddleJsonError(f"unreadable PDF: {exc}") from exc
    else:
        try:
            with pymupdf.open(path) as document:
                page_count = document.page_count
        except Exception as exc:
            raise MiddleJsonError(f"unreadable PDF: {exc}") from exc
    if page_count <= 0:
        raise MiddleJsonError("PDF has no pages")
    return page_count


class ReviewShardWriter:
    def __init__(self, directory: Path, shard_size: int) -> None:
        if shard_size <= 0:
            raise ValueError("review shard size must be positive")
        self.directory = directory
        self.shard_size = shard_size
        self.count = 0
        self.paths: list[str] = []
        self._handle: TextIO | None = None

    def write(self, record: dict[str, Any]) -> None:
        if self.count % self.shard_size == 0:
            self.close_handle()
            shard_index = self.count // self.shard_size
            path = self.directory / f"shard_{shard_index:05d}.jsonl"
            self._handle = path.open("w", encoding="utf-8")
            self.paths.append(f"review_shards/{path.name}")
        assert self._handle is not None
        self._handle.write(_json_line(record))
        self.count += 1

    def close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        self.close_handle()


def _replace_output(temp_output: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists (use --overwrite): {output_root}")
        backup = output_root.with_name(output_root.name + ".previous")
        if backup.exists():
            raise FileExistsError(f"refusing to replace existing backup: {backup}")
        output_root.rename(backup)
        try:
            os.replace(temp_output, output_root)
        except BaseException:
            backup.rename(output_root)
            raise
        shutil.rmtree(backup)
        return
    os.replace(temp_output, output_root)


def build_dataset(
    *,
    input_root: Path,
    output_root: Path,
    doc_ids: list[str] | None = None,
    limit: int | None = None,
    review_shard_size: int = 500,
    include_promotions: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if (
        output_root == input_root
        or input_root in output_root.parents
        or output_root in input_root.parents
    ):
        raise ValueError("input_root and output_root must be disjoint sibling trees")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    middle_paths = _discover_documents(input_root, doc_ids)
    if limit is not None:
        middle_paths = middle_paths[:limit]
    if not middle_paths:
        raise ValueError("no middle.json files selected")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    (temp_output / "review_shards").mkdir()

    totals: Counter[str] = Counter()
    review_reasons: Counter[str] = Counter()
    content_flags: Counter[str] = Counter()
    content_flag_occurrences: Counter[str] = Counter()
    text_quality_flags: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    candidate_ids: set[str] = set()
    shard_writer = ReviewShardWriter(temp_output / "review_shards", review_shard_size)

    try:
        with (
            (temp_output / "document_manifest.jsonl").open("w", encoding="utf-8") as manifest_out,
            (temp_output / "title_candidates.jsonl").open("w", encoding="utf-8") as candidates_out,
            (temp_output / "review_queue.jsonl").open("w", encoding="utf-8") as review_out,
            (temp_output / "text_review_queue.jsonl").open("w", encoding="utf-8") as text_review_out,
            (temp_output / "content_quarantine.jsonl").open("w", encoding="utf-8") as content_out,
        ):
            for middle_path in middle_paths:
                doc_id = middle_path.parent.name
                source_rel = middle_path.relative_to(input_root).as_posix()
                pdf_path = middle_path.parent / "document.pdf"
                totals["documents"] += 1
                try:
                    source_hash = sha256_file(middle_path)
                    payload = json.loads(middle_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise MiddleJsonError("middle.json root must be an object")
                    if not pdf_path.is_file():
                        raise MiddleJsonError("document.pdf is missing")
                    pdf_pages = _pdf_page_count(pdf_path)
                    result = analyze_document(
                        doc_id=doc_id,
                        payload=payload,
                        source_middle_json=source_rel,
                        source_sha256=source_hash,
                        document_dir=middle_path.parent,
                        include_promotions=include_promotions,
                    )
                    if pdf_pages != result["pages"]:
                        raise MiddleJsonError(
                            f"PDF page count {pdf_pages} != middle.json page count {result['pages']}"
                        )
                    pdf_hash = sha256_file(pdf_path)
                except (OSError, json.JSONDecodeError, MiddleJsonError, ValueError) as exc:
                    totals["failed_documents"] += 1
                    manifest_out.write(
                        _json_line(
                            {
                                "schema_version": MANIFEST_SCHEMA_VERSION,
                                "ruleset_version": RULESET_VERSION,
                                "doc_id": doc_id,
                                "source_middle_json": source_rel,
                                "status": "ERROR",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                    )
                    content_out.write(
                        _json_line(
                            {
                                "schema_version": "pmc-content-quarantine-v1",
                                "ruleset_version": RULESET_VERSION,
                                "doc_id": doc_id,
                                "source_middle_json": source_rel,
                                "flags": ["STRUCTURE_ERROR"],
                                "flag_counts": {"STRUCTURE_ERROR": 1},
                                "examples": [{"code": "STRUCTURE_ERROR", "detail": str(exc)}],
                            }
                        )
                    )
                    content_flags["STRUCTURE_ERROR"] += 1
                    continue

                doc_candidate_count = len(result["candidates"])
                doc_review_count = len(result["review_records"])
                auto_count = doc_candidate_count - doc_review_count
                totals["pages"] += result["pages"]
                totals["source_titles"] += result["source_title_count"]
                totals["candidates"] += doc_candidate_count
                totals["review_required"] += doc_review_count
                totals["auto_silver"] += auto_count
                totals["text_review_required"] += len(result["text_review_records"])
                if result["content_flags"]:
                    totals["documents_with_content_flags"] += 1

                for candidate in result["candidates"]:
                    candidate_id = candidate["candidate_id"]
                    if candidate_id in candidate_ids:
                        raise RuntimeError(f"duplicate candidate_id: {candidate_id}")
                    candidate_ids.add(candidate_id)
                    candidates_out.write(_json_line(candidate))
                    operations[candidate["operation"]] += 1
                    statuses[candidate["review_status"]] += 1
                    level = candidate.get("input_level")
                    if candidate["input_type"] == "title":
                        levels[str(level)] += 1
                    text_quality_flags.update(candidate["text_quality_flags"])

                for review in result["review_records"]:
                    review_out.write(_json_line(review))
                    shard_writer.write(review)
                    review_reasons.update(review["reason_codes"])

                for text_review in result["text_review_records"]:
                    text_review_out.write(_json_line(text_review))

                if result["content_flags"]:
                    content_out.write(
                        _json_line(
                            {
                                "schema_version": "pmc-content-quarantine-v1",
                                "ruleset_version": RULESET_VERSION,
                                "doc_id": doc_id,
                                "source_middle_json": source_rel,
                                "source_sha256": source_hash,
                                "flags": result["content_flags"],
                                "flag_counts": result["content_flag_counts"],
                                "examples": result["content_examples"],
                            }
                        )
                    )
                    content_flags.update(result["content_flags"])
                    content_flag_occurrences.update(result["content_flag_counts"])

                manifest_out.write(
                    _json_line(
                        {
                            "schema_version": MANIFEST_SCHEMA_VERSION,
                            "ruleset_version": RULESET_VERSION,
                            "doc_id": doc_id,
                            "source_middle_json": source_rel,
                            "source_sha256": source_hash,
                            "source_pdf": pdf_path.relative_to(input_root).as_posix(),
                            "pdf_sha256": pdf_hash,
                            "backend": payload.get("_backend"),
                            "source_version": payload.get("_version_name"),
                            "pages": result["pages"],
                            "pdf_pages": pdf_pages,
                            "source_title_count": result["source_title_count"],
                            "candidate_count": doc_candidate_count,
                            "auto_silver_count": auto_count,
                            "review_required_count": doc_review_count,
                            "text_review_required_count": len(result["text_review_records"]),
                            "content_flags": result["content_flags"],
                            "status": "OK",
                        }
                    )
                )

        shard_writer.close()
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "candidate_schema_version": SCHEMA_VERSION,
            "ruleset_version": RULESET_VERSION,
            "input_layout": "documents/<doc_id>/{middle.json,document.pdf}",
            "configuration": {
                "include_promotions": include_promotions,
                "review_shard_size": review_shard_size,
                "selected_doc_ids": sorted(doc_ids) if doc_ids else None,
                "limit": limit,
            },
            "totals": dict(sorted(totals.items())),
            "title_input_levels": dict(sorted(levels.items(), key=lambda item: item[0])),
            "operations": dict(sorted(operations.items())),
            "review_statuses": dict(sorted(statuses.items())),
            "review_reason_counts": dict(sorted(review_reasons.items())),
            "documents_by_content_flag": dict(sorted(content_flags.items())),
            "content_flag_occurrences": dict(sorted(content_flag_occurrences.items())),
            "candidate_text_quality_flag_counts": dict(sorted(text_quality_flags.items())),
            "review_shards": shard_writer.paths,
        }
        _write_json(temp_output / "summary.json", summary)
        _replace_output(temp_output, output_root, overwrite)
        if totals["failed_documents"]:
            raise RuntimeError(
                f"built output with {totals['failed_documents']} failed document(s); see manifest"
            )
        return summary
    except BaseException:
        shard_writer.close()
        if temp_output.exists():
            shutil.rmtree(temp_output)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--doc-id", action="append", dest="doc_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--review-shard-size", type=int, default=500)
    parser.add_argument("--no-promotion-candidates", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_dataset(
            input_root=args.input_root,
            output_root=args.output_root,
            doc_ids=args.doc_ids,
            limit=args.limit,
            review_shard_size=args.review_shard_size,
            include_promotions=not args.no_promotion_candidates,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
