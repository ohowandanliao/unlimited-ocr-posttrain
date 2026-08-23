#!/usr/bin/env python3
"""Materialize PDF pages referenced by mixed reviewed JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fitz():
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyMuPDF is required; install it with: pip install PyMuPDF") from exc
    return fitz


def _safe_target(value: str, output_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"image target must be absolute: {value}")
    root = output_root.resolve()
    candidate = path.resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError(f"image target escapes output root: {value}")
    return candidate


ALLOWED_SOURCES = {"READoc-arxiv", "READoc-github", "PMC-v26-synthetic"}
ALLOWED_SAMPLE_FORMS = {"full_document", "strict_single_page"}
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_target(source: str, doc_id: str, page: int, output_root: Path) -> Path:
    if source == "PMC-v26-synthetic":
        return output_root / "pmc" / doc_id / f"page_{page:04d}.png"
    family = source.removeprefix("READoc-")
    return output_root / "readoc" / family / doc_id / f"page_{page:04d}.png"


def _error_category(message: str) -> str:
    lowered = message.lower()
    if "sha" in lowered:
        return "sha_mismatch"
    if "path" in lowered or "archive" in lowered or "member" in lowered or "escapes output root" in lowered:
        return "source_path"
    if "page" in lowered or "pages" in lowered:
        return "page_mismatch"
    if "missing" in lowered or "not found" in lowered or "no such file" in lowered:
        return "missing"
    return "contract"


def load_tasks(
    jsonl_paths: list[Path],
    output_root: Path,
    manifest: dict | None = None,
    validation_errors: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    tasks = {}
    target_owners = {}
    input_sha256 = {}
    for jsonl in jsonl_paths:
        path = jsonl.resolve()
        input_sha256[str(path)] = sha256_file(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                    meta = row.get("meta")
                    if not isinstance(meta, dict):
                        raise ValueError(f"{path}:{line_number}: missing meta")
                    source, doc_id = meta.get("source"), meta.get("doc_id")
                    pages, images = meta.get("page_indices"), row.get("images")
                    if not isinstance(source, str) or source not in ALLOWED_SOURCES or not isinstance(doc_id, str) or not doc_id:
                        raise ValueError(f"{path}:{line_number}: unsupported source {source!r}")
                    if not isinstance(pages, list) or not pages or not isinstance(images, list) or len(pages) != len(images):
                        raise ValueError(f"{path}:{line_number}: page_indices/images mismatch")
                    sample_form = meta.get("sample_form")
                    if sample_form not in ALLOWED_SAMPLE_FORMS:
                        raise ValueError(f"{path}:{line_number}: invalid sample_form {sample_form!r}")
                    n_pages = meta.get("n_pages")
                    if not isinstance(n_pages, int) or isinstance(n_pages, bool) or n_pages != len(pages):
                        raise ValueError(f"{path}:{line_number}: n_pages/page_indices mismatch")
                    if len(set(pages)) != len(pages) or pages != sorted(pages):
                        raise ValueError(f"{path}:{line_number}: page_indices must be unique and ordered")
                    if sample_form == "strict_single_page" and len(pages) != 1:
                        raise ValueError(f"{path}:{line_number}: strict_single_page must reference one page")
                    key = (source, doc_id)
                    manifest_record = (manifest or {}).get(key)
                    if not manifest_record:
                        raise ValueError(f"manifest entry missing for {key}")
                    manifest_sha = manifest_record.get("pdf_sha256")
                    row_sha = meta.get("source_pdf_sha256")
                    if not isinstance(row_sha, str) or row_sha != manifest_sha:
                        raise ValueError(f"row/manifest PDF SHA mismatch for {key}")
                    pdf_pages = manifest_record.get("pdf_pages", manifest_record.get("pages"))
                    if not isinstance(pdf_pages, int) or isinstance(pdf_pages, bool) or pdf_pages <= 0:
                        raise ValueError(f"manifest page count missing for {key}")
                    if any(not isinstance(page, int) or isinstance(page, bool) or page < 0 or page >= pdf_pages for page in pages):
                        raise ValueError(f"page index out of manifest range for {key}: {pages}")
                    if sample_form == "full_document" and pages != list(range(pdf_pages)):
                        raise ValueError(f"full page_indices do not equal manifest pdf_pages for {key}")
                    split = meta.get("split")
                    if not isinstance(split, str) or not split:
                        raise ValueError(f"{path}:{line_number}: invalid split")
                    task = tasks.setdefault(key, {"source": source, "doc_id": doc_id, "pages": {}, "rows": [], "splits": set(), "pdf_pages": pdf_pages})
                    if task["pdf_pages"] != pdf_pages:
                        raise ValueError(f"conflicting manifest page count for {key}")
                    task["splits"].add(split)
                    task["pdf_sha256"] = row_sha
                    for page, image in zip(pages, images):
                        target = _safe_target(image, output_root)
                        canonical = _canonical_target(source, doc_id, page, output_root).resolve()
                        if target != canonical:
                            raise ValueError(f"image target is not canonical for {key} page {page}: {image}")
                        owner = (key, page)
                        previous = target_owners.get(target)
                        if previous is not None and previous != owner:
                            raise ValueError(f"conflicting image target {target}: {previous} vs {owner}")
                        target_owners[target] = owner
                        previous_page = task["pages"].get(page)
                        if previous_page is not None and previous_page != target:
                            raise ValueError(f"conflicting target for {key} page {page}")
                        task["pages"][page] = target
                    task["rows"].append(str(path))
                except Exception as exc:
                    if validation_errors is None:
                        raise
                    validation_errors.append({"category": _error_category(str(exc)), "error": str(exc)})
    return list(tasks.values()), input_sha256


def _readoc_pdf(task: dict, readoc_root: Path, manifest: dict) -> bytes:
    record = manifest.get((task["source"], task["doc_id"]))
    if record is None:
        raise ValueError(f"READoc manifest entry missing for {task['source']}/{task['doc_id']}")
    archive = Path(record["source_pdf_archive"])
    if archive.is_absolute() or ".." in archive.parts:
        raise ValueError("READoc archive path escapes readoc root")
    archive_path = (readoc_root / archive).resolve()
    if not archive_path.is_relative_to(readoc_root.resolve()):
        raise ValueError("READoc archive path escapes readoc root")
    with zipfile.ZipFile(archive_path) as bundle:
        member = record.get("source_pdf_member")
        if not isinstance(member, str) or member.startswith("/") or ".." in Path(member).parts:
            raise ValueError("invalid READoc PDF member path")
        with bundle.open(member) as handle:
            return handle.read()


def _pdf_bytes(task: dict, readoc_root: Path | None, pmc_root: Path | None, manifest: dict) -> bytes:
    if task["source"].startswith("READoc-"):
        if readoc_root is None:
            raise ValueError("--readoc-root is required for READoc rows")
        data = _readoc_pdf(task, readoc_root, manifest)
    else:
        if pmc_root is None:
            raise ValueError("--pmc-root is required for PMC rows")
        record = manifest.get((task["source"], task["doc_id"]))
        if record is None:
            raise FileNotFoundError(f"PMC manifest entry missing for {task['source']}/{task['doc_id']}")
        relative = Path(record["source_pdf"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("PMC source_pdf path escapes pmc root")
        pdf = (pmc_root / relative).resolve()
        if not pdf.is_relative_to(pmc_root.resolve()):
            raise ValueError("PMC PDF path escapes pmc root")
        data = pdf.read_bytes()
    if sha256_bytes(data) != task["pdf_sha256"]:
        raise ValueError(f"PDF SHA256 mismatch for {task['source']}/{task['doc_id']}")
    return data


def materialize_task(task: dict, *, readoc_root: Path | None, pmc_root: Path | None, manifest: dict, dpi: int, overwrite: bool) -> dict:
    fitz = _fitz()
    data = _pdf_bytes(task, readoc_root, pmc_root, manifest)
    pdf = fitz.open(stream=data, filetype="pdf")
    try:
        if len(pdf) != task["pdf_pages"]:
            raise ValueError(
                f"page mismatch: manifest declares {task['pdf_pages']} pages, PDF has {len(pdf)}"
            )
        pages = sorted(task["pages"])
        for page_number in pages:
            if page_number >= len(pdf):
                raise ValueError(f"page mismatch: page index {page_number} out of range (PDF has {len(pdf)} pages)")
        for page_number in pages:
            target = task["pages"][page_number]
            if target.exists() and not overwrite:
                raise FileExistsError(f"refusing to overwrite {target}; pass --overwrite")
            target.parent.mkdir(parents=True, exist_ok=True)
            pix = pdf[page_number].get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
            fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            try:
                os.close(fd)
                pix.save(temp_name, output="png")
                if target.exists() and not overwrite:
                    raise FileExistsError(f"refusing to overwrite {target}; pass --overwrite")
                os.replace(temp_name, target)
            finally:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
    finally:
        pdf.close()
    return {"docs": 1, "pages": len(task["pages"]), "error": None}


def run_task(task: dict, *, readoc_root: Path | None, pmc_root: Path | None, manifest: dict, dpi: int, overwrite: bool, verify_only: bool = False) -> dict:
    try:
        if verify_only:
            data = _pdf_bytes(task, readoc_root, pmc_root, manifest)
            fitz = _fitz()
            pdf = fitz.open(stream=data, filetype="pdf")
            try:
                if len(pdf) != task["pdf_pages"]:
                    raise ValueError(
                        f"page mismatch: manifest declares {task['pdf_pages']} pages, PDF has {len(pdf)}"
                    )
                for page in task["pages"]:
                    if page >= len(pdf):
                        raise ValueError(f"page mismatch: page index {page} out of range (PDF has {len(pdf)} pages)")
            finally:
                pdf.close()
            return {"docs": 1, "pages": len(task["pages"]), "rendered": False, "error": None}
        result = materialize_task(task, readoc_root=readoc_root, pmc_root=pmc_root, manifest=manifest, dpi=dpi, overwrite=overwrite)
        result["rendered"] = True
        return result
    except Exception as exc:
        return {"docs": 0, "pages": 0, "error": f"{task['source']}/{task['doc_id']}: {exc}"}


def read_manifest(path: Path, *, kind: str) -> dict:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if kind == "readoc":
                source = {"arxiv": "READoc-arxiv", "github": "READoc-github"}.get(record.get("source"))
            else:
                source = "PMC-v26-synthetic" if record.get("doc_id") else None
            if source is None or not isinstance(record.get("doc_id"), str):
                raise ValueError("manifest source/doc_id is invalid")
            fields = ("source_pdf_archive", "source_pdf_member", "pdf_sha256") if kind == "readoc" else ("source_pdf", "pdf_sha256")
            for field in fields:
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(f"manifest entry missing {field}")
            if not HEX_SHA256.fullmatch(record["pdf_sha256"]):
                raise ValueError(f"manifest entry has invalid pdf_sha256 for {record['doc_id']}")
            page_count = record.get("pdf_pages", record.get("pages"))
            if not isinstance(page_count, int) or isinstance(page_count, bool) or page_count <= 0:
                raise ValueError(f"manifest entry missing positive pdf_pages/pages for {record['doc_id']}")
            key = (source, record["doc_id"])
            if key in result:
                raise ValueError(f"duplicate READoc manifest key: {key}")
            normalized = dict(record); normalized["source"] = source; normalized["pdf_pages"] = page_count
            result[key] = normalized
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--readoc-root", type=Path)
    parser.add_argument("--readoc-manifest", type=Path)
    parser.add_argument("--pmc-root", type=Path)
    parser.add_argument("--pmc-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-sources", action="store_true", help="verify source SHA/page ranges without rendering")
    parser.add_argument("--report-path", type=Path, help="write the report here; omitted reports are stdout-only")
    args = parser.parse_args(argv)
    if args.dpi <= 0 or args.workers <= 0:
        parser.error("--dpi and --workers must be positive")
    output_root = args.output_root.resolve()
    try:
        manifest = {}
        if args.readoc_manifest:
            manifest.update(read_manifest(args.readoc_manifest.resolve(), kind="readoc"))
        if args.pmc_manifest:
            manifest.update(read_manifest(args.pmc_manifest.resolve(), kind="pmc"))
        validation_errors = []
        tasks, input_sha256 = load_tasks([p.resolve() for p in args.jsonl], output_root, manifest, validation_errors)
    except Exception as exc:
        report = {"schema_version": "pdf-page-materialization-report-v1", "input_sha256": {}, "dpi": args.dpi, "docs": 0, "pages": 0, "source_split": {}, "verified_source_docs": 0, "verified_source_pages": 0, "rendered_docs": 0, "rendered_pages": 0, "failures": 1, "errors": [str(exc)], "error_counts": {"contract": 1}, "dry_run": args.dry_run, "verify_sources": args.verify_sources}
        _write_report(report, args.report_path, allow_write=not args.dry_run)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 1
    stats = Counter(); failures = list(validation_errors)
    error_counts = Counter(item["category"] for item in validation_errors)
    stats["docs"] = len(tasks); stats["pages"] = sum(len(t["pages"]) for t in tasks)
    source_split = Counter()
    for task in tasks:
        for split in task["splits"]:
            source_split[(task["source"], split)] += 1
    verified_docs = verified_pages = rendered_docs = rendered_pages = 0
    if not args.dry_run:
        fn = lambda task: run_task(task, readoc_root=args.readoc_root.resolve() if args.readoc_root else None, pmc_root=args.pmc_root.resolve() if args.pmc_root else None, manifest=manifest, dpi=args.dpi, overwrite=args.overwrite, verify_only=args.verify_sources)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for task, result in zip(tasks, pool.map(fn, tasks)):
                if result.get("error"):
                    category = _error_category(result["error"])
                    error_counts[category] += 1
                    failures.append({"category": category, "error": result["error"]})
                else:
                    verified_docs += 1; verified_pages += result["pages"]
                    if result.get("rendered"):
                        rendered_docs += 1; rendered_pages += result["pages"]
    report = {"schema_version": "pdf-page-materialization-report-v1", "input_sha256": input_sha256, "dpi": args.dpi, "docs": stats["docs"], "pages": stats["pages"], "source_split": {f"{source}/{split}": docs for (source, split), docs in sorted(source_split.items())}, "verified_source_docs": verified_docs, "verified_source_pages": verified_pages, "rendered_docs": rendered_docs, "rendered_pages": rendered_pages, "failures": len(failures), "errors": failures, "error_counts": dict(sorted(error_counts.items())), "dry_run": args.dry_run, "verify_sources": args.verify_sources}
    _write_report(report, args.report_path, allow_write=not args.dry_run)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1 if failures else 0


def _write_report(report: dict, report_path: Path | None, *, allow_write: bool) -> None:
    if report_path is None or not allow_write:
        return
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{report_path.name}.", suffix=".tmp", dir=report_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, report_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
