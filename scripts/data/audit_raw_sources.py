#!/usr/bin/env python3
"""Audit READoc and PMC source data without extracting or copying training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import tempfile
import warnings
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

from markdown_it import MarkdownIt
from pypdf import PdfReader


SCHEMA_VERSION = "uocr-raw-source-audit-v1"
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^\n)]*\)")
_FORMULA_RE = re.compile(r"(?<!\\)\$|\\(?:begin|end)\{|\\\[|\\\]")


class AuditError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readoc-root", type=Path, required=True)
    parser.add_argument("--pmc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-pdf-profile",
        action="store_true",
        help="skip PDF page/geometry parsing; hashes and source mappings are still checked",
    )
    parser.add_argument(
        "--strict-pdf-probe",
        action="store_true",
        help="also try pypdf strict=True and record failures/warnings",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _warning_text(caught: Iterable[warnings.WarningMessage]) -> list[str]:
    return sorted({f"{item.category.__name__}: {item.message}" for item in caught})


def pdf_profile(data: bytes, *, strict_probe: bool) -> dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader = PdfReader(BytesIO(data), strict=False)
        page_sizes = Counter()
        rotations = Counter()
        for page in reader.pages:
            width = round(float(page.mediabox.width), 2)
            height = round(float(page.mediabox.height), 2)
            page_sizes[f"{width:g}x{height:g}"] += 1
            rotations[str(int(page.get("/Rotate", 0) or 0) % 360)] += 1
        result: dict[str, Any] = {
            "pages": len(reader.pages),
            "page_sizes": dict(sorted(page_sizes.items())),
            "rotations": dict(sorted(rotations.items())),
            "parser_warnings": _warning_text(caught),
        }

    if strict_probe:
        try:
            with warnings.catch_warnings(record=True) as strict_caught:
                warnings.simplefilter("always")
                strict_reader = PdfReader(BytesIO(data), strict=True)
                strict_pages = len(strict_reader.pages)
            result["strict_probe"] = {
                "ok": True,
                "pages": strict_pages,
                "warnings": _warning_text(strict_caught),
            }
        except Exception as exc:  # pypdf raises several parser-specific exception types
            result["strict_probe"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result


def markdown_profile(text: str) -> dict[str, Any]:
    if "\x00" in text:
        raise AuditError("Markdown contains NUL")
    headings = sum(
        token.type == "heading_open"
        for token in MarkdownIt("commonmark").parse(text)
    )
    lines = text.splitlines()
    return {
        "utf8_bytes": len(text.encode("utf-8")),
        "chars": len(text),
        "lines": len(lines),
        "headings": headings,
        "fence_markers": sum(bool(_FENCE_RE.match(line)) for line in lines),
        "formula_markers": len(_FORMULA_RE.findall(text)),
        "markdown_table_rows": sum(bool(_TABLE_ROW_RE.match(line)) for line in lines),
        "markdown_images": len(_MARKDOWN_IMAGE_RE.findall(text)),
        "has_cjk": bool(_CJK_RE.search(text)),
        "ends_with_newline": text.endswith("\n"),
    }


def _safe_member_ids(archive: zipfile.ZipFile, source: str) -> tuple[dict[str, str], list[str]]:
    prefix = f"{source}/pdf/"
    members: dict[str, str] = {}
    unsafe = []
    for info in archive.infolist():
        path = Path(info.filename)
        if path.is_absolute() or ".." in path.parts:
            unsafe.append(info.filename)
        if info.is_dir() or not info.filename.lower().endswith(".pdf"):
            continue
        if not info.filename.startswith(prefix):
            continue
        doc_id = Path(info.filename).stem
        if doc_id in members:
            raise AuditError(f"duplicate READoc PDF id in {archive.filename}: {doc_id}")
        members[doc_id] = info.filename
    return members, unsafe


def audit_readoc(root: Path, *, profile_pdfs: bool, strict_probe: bool) -> tuple[list[dict], dict]:
    records = []
    source_summaries = {}
    for source in ("arxiv", "github"):
        gt_dir = root / "ground_truth" / source
        archive_path = root / "archives" / f"{source}.zip"
        if not gt_dir.is_dir() or not archive_path.is_file():
            raise AuditError(f"READoc source layout is incomplete for {source}")
        gt_paths = {path.stem: path for path in gt_dir.glob("*.md")}
        with zipfile.ZipFile(archive_path) as archive:
            pdf_members, unsafe_members = _safe_member_ids(archive, source)
            missing_pdf = sorted(set(gt_paths) - set(pdf_members))
            missing_gt = sorted(set(pdf_members) - set(gt_paths))
            if missing_pdf or missing_gt:
                raise AuditError(
                    f"READoc {source} PDF/GT mismatch: missing_pdf={missing_pdf[:5]} "
                    f"missing_gt={missing_gt[:5]}"
                )
            if unsafe_members:
                raise AuditError(f"READoc {source} archive has unsafe members: {unsafe_members[:5]}")

            pages = []
            strict_failures = 0
            for doc_id in sorted(gt_paths):
                gt_path = gt_paths[doc_id]
                target = gt_path.read_text(encoding="utf-8")
                pdf_member = pdf_members[doc_id]
                pdf_data = archive.read(pdf_member)
                record: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "source": f"READoc-{source}",
                    "doc_id": doc_id,
                    "ground_truth": str(gt_path.relative_to(root)),
                    "ground_truth_sha256": sha256_bytes(target.encode("utf-8")),
                    "pdf_archive": str(archive_path.relative_to(root)),
                    "pdf_member": pdf_member,
                    "pdf_sha256": sha256_bytes(pdf_data),
                    "target": markdown_profile(target),
                }
                if profile_pdfs:
                    record["pdf"] = pdf_profile(pdf_data, strict_probe=strict_probe)
                    pages.append(record["pdf"]["pages"])
                    if not record["pdf"].get("strict_probe", {"ok": True})["ok"]:
                        strict_failures += 1
                records.append(record)

        source_records = [row for row in records if row["source"] == f"READoc-{source}"]
        source_summaries[source] = {
            "documents": len(source_records),
            "archive_sha256": sha256_file(archive_path),
            "target_chars": distribution(row["target"]["chars"] for row in source_records),
            "target_lines": distribution(row["target"]["lines"] for row in source_records),
            "documents_without_headings": sum(row["target"]["headings"] == 0 for row in source_records),
            "documents_without_final_newline": sum(
                not row["target"]["ends_with_newline"] for row in source_records
            ),
            "documents_with_cjk": sum(row["target"]["has_cjk"] for row in source_records),
            "pdf_pages": distribution(pages) if pages else None,
            "strict_probe_failures": strict_failures if strict_probe else None,
        }
    return records, {"documents": len(records), "sources": source_summaries}


def _iter_blocks(blocks: object) -> Iterable[Mapping[str, Any]]:
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        yield block
        yield from _iter_blocks(block.get("blocks"))


def _iter_spans(block: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for line in block.get("lines") or []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans") or []:
            if isinstance(span, dict):
                yield span


def pmc_middle_profile(payload: Mapping[str, Any], document_root: Path) -> dict[str, Any]:
    pages = payload.get("pdf_info")
    if isinstance(pages, dict):
        pages = pages.get("pdf_info")
    if not isinstance(pages, list):
        raise AuditError("PMC middle.json has no pdf_info list")
    block_types = Counter()
    image_refs = []
    empty_pages = []
    page_sizes = Counter()
    for position, page in enumerate(pages):
        if not isinstance(page, dict) or page.get("page_idx") != position:
            raise AuditError(f"PMC page_idx is not contiguous at {position}")
        size = page.get("page_size")
        if isinstance(size, list) and len(size) == 2:
            page_sizes[f"{size[0]}x{size[1]}"] += 1
        para_blocks = page.get("para_blocks") or []
        if not para_blocks:
            empty_pages.append(position)
        for block in _iter_blocks(para_blocks):
            block_types[str(block.get("type") or "<missing>")] += 1
            for span in _iter_spans(block):
                image_path = span.get("image_path")
                if isinstance(image_path, str) and image_path:
                    image_refs.append(image_path)
    unique_refs = sorted(set(image_refs))
    missing = [ref for ref in unique_refs if not (document_root / ref).is_file()]
    return {
        "pages": len(pages),
        "backend": payload.get("_backend"),
        "version": payload.get("_version_name"),
        "page_sizes": dict(sorted(page_sizes.items())),
        "block_types": dict(sorted(block_types.items())),
        "empty_page_indices": empty_pages,
        "image_asset_references": len(image_refs),
        "unique_image_assets": len(unique_refs),
        "missing_unique_image_assets": len(missing),
    }


def audit_pmc(root: Path, *, profile_pdfs: bool, strict_probe: bool) -> tuple[list[dict], dict]:
    documents_root = root / "documents"
    if not documents_root.is_dir():
        raise AuditError("PMC root must contain documents/<doc_id>/{middle.json,document.pdf}")
    records = []
    total_blocks = Counter()
    pages = []
    strict_failures = 0
    for document_root in sorted(path for path in documents_root.iterdir() if path.is_dir()):
        middle_path = document_root / "middle.json"
        pdf_path = document_root / "document.pdf"
        if not middle_path.is_file() or not pdf_path.is_file():
            raise AuditError(f"PMC document is incomplete: {document_root.name}")
        middle_data = middle_path.read_bytes()
        payload = json.loads(middle_data)
        middle = pmc_middle_profile(payload, document_root)
        total_blocks.update(middle["block_types"])
        pdf_data = pdf_path.read_bytes()
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source": "PMC-v26-synthetic",
            "doc_id": document_root.name,
            "middle_json": str(middle_path.relative_to(root)),
            "middle_json_sha256": sha256_bytes(middle_data),
            "pdf": str(pdf_path.relative_to(root)),
            "pdf_sha256": sha256_bytes(pdf_data),
            "middle": middle,
        }
        if profile_pdfs:
            record["pdf_profile"] = pdf_profile(pdf_data, strict_probe=strict_probe)
            if record["pdf_profile"]["pages"] != middle["pages"]:
                raise AuditError(
                    f"{document_root.name}: PDF pages {record['pdf_profile']['pages']} "
                    f"!= middle pages {middle['pages']}"
                )
            if not record["pdf_profile"].get("strict_probe", {"ok": True})["ok"]:
                strict_failures += 1
        pages.append(middle["pages"])
        records.append(record)
    return records, {
        "documents": len(records),
        "pages": sum(pages),
        "pages_per_document": distribution(pages),
        "block_types": dict(sorted(total_blocks.items())),
        "empty_pages": sum(len(row["middle"]["empty_page_indices"]) for row in records),
        "documents_with_empty_pages": sum(bool(row["middle"]["empty_page_indices"]) for row in records),
        "image_asset_references": sum(row["middle"]["image_asset_references"] for row in records),
        "unique_image_assets_per_document_sum": sum(row["middle"]["unique_image_assets"] for row in records),
        "missing_unique_image_assets_per_document_sum": sum(
            row["middle"]["missing_unique_image_assets"] for row in records
        ),
        "strict_probe_failures": strict_failures if strict_probe else None,
    }


def distribution(values: Iterable[int]) -> dict[str, int | float]:
    materialized = list(values)
    if not materialized:
        return {"count": 0}
    return {
        "count": len(materialized),
        "min": min(materialized),
        "median": statistics.median(materialized),
        "max": max(materialized),
        "sum": sum(materialized),
    }


def _assert_output_is_separate(output_dir: Path, sources: Iterable[Path]) -> None:
    if output_dir.exists():
        raise AuditError(f"output directory already exists: {output_dir}")
    for source in sources:
        if output_dir == source or output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
            raise AuditError("output directory and source roots must be separate and non-nested")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    readoc_root = args.readoc_root.resolve()
    pmc_root = args.pmc_root.resolve()
    output_dir = args.output_dir.resolve()
    _assert_output_is_separate(output_dir, (readoc_root, pmc_root))
    if not readoc_root.is_dir() or not pmc_root.is_dir():
        raise AuditError("both source roots must exist")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    try:
        profile_pdfs = not args.skip_pdf_profile
        readoc_records, readoc_summary = audit_readoc(
            readoc_root,
            profile_pdfs=profile_pdfs,
            strict_probe=args.strict_pdf_probe,
        )
        pmc_records, pmc_summary = audit_pmc(
            pmc_root,
            profile_pdfs=profile_pdfs,
            strict_probe=args.strict_pdf_probe,
        )
        write_jsonl(temp_dir / "readoc_documents.jsonl", readoc_records)
        write_jsonl(temp_dir / "pmc_documents.jsonl", pmc_records)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "readoc_root": str(readoc_root),
            "pmc_root": str(pmc_root),
            "pdf_profiled": profile_pdfs,
            "strict_pdf_probe": args.strict_pdf_probe,
            "readoc": readoc_summary,
            "pmc": pmc_summary,
        }
        (temp_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"ERROR: {exc}")
