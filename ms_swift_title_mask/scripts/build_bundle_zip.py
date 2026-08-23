#!/usr/bin/env python3
"""Validate and reproducibly package the ms_swift_title_mask bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_assets"
SPLITS = ("train", "validation", "test")
EXPECTED_SPLIT_ROWS = {"train": 4073, "validation": 223, "test": 228}
FORBIDDEN_SUFFIXES = (".pdf", ".png", ".tgz", ".tar.gz", ".pyc")
EXPECTED_POOL_ROWS = {"readoc_full": 1552, "pmc_full": 1486, "pmc_single": 1486}
EXPECTED_MIX_VERSION = "silver-readoc-full-pmc-full-single-natural-v1"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSONL {path}:{line_number}: {exc}")


def validate_jsonl(path: Path, expected: int | None = None) -> list[dict]:
    items = [item for _line, item in rows(path)]
    if expected is not None and len(items) != expected:
        fail(f"{path}: expected {expected} rows, found {len(items)}")
    return items


def canonical_target(text: str) -> str:
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        fail("assistant/GT content must be a non-empty string without NUL")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.rstrip("\n") + "\n"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_unique(path: Path, key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict]:
    result = {}
    for line_number, row in rows(path):
        key = tuple(row.get(field) for field in key_fields)
        if any(not isinstance(value, str) or not value for value in key):
            fail(f"{path}:{line_number}: invalid manifest key {key_fields}: {key}")
        if key in result:
            fail(f"{path}:{line_number}: duplicate manifest key {key}")
        result[key] = row
    return result


def assistant_target(row: dict, location: str) -> tuple[str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        fail(f"{location}: expected exactly one user and one assistant message")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        fail(f"{location}: invalid message roles")
    prompt, target = messages[0].get("content"), messages[1].get("content")
    if not isinstance(prompt, str) or not isinstance(target, str):
        fail(f"{location}: message content must be strings")
    if target != canonical_target(target):
        fail(f"{location}: assistant target is not newline-canonical")
    return prompt, target


def expected_image_suffixes(source: str, doc_id: str, page_indices: list[int]) -> list[Path]:
    if source == "PMC-v26-synthetic":
        base = Path("pmc") / doc_id
    elif source in {"READoc-arxiv", "READoc-github"}:
        base = Path("readoc") / source.removeprefix("READoc-") / doc_id
    else:
        fail(f"unsupported training source: {source!r}")
    return [base / f"page_{page:04d}.png" for page in page_indices]


def validate_source_row(
    row: dict,
    *,
    pool: str,
    split: str,
    location: str,
    readoc_manifest: dict,
    readoc_silver: dict,
    pmc_manifest: dict,
    expected_gt: set[Path],
) -> str:
    row_id = row.get("id")
    if not isinstance(row_id, str) or not row_id:
        fail(f"{location}: missing row id")
    if row.get("channel") != "title_reviewed":
        fail(f"{location}: invalid channel")
    meta = row.get("meta")
    if not isinstance(meta, dict):
        fail(f"{location}: missing meta")
    source, doc_id = meta.get("source"), meta.get("doc_id")
    if not isinstance(source, str) or not isinstance(doc_id, str) or not doc_id:
        fail(f"{location}: invalid source/doc_id")
    if meta.get("split") != split:
        fail(f"{location}: meta split does not match {split}")

    prompt, target = assistant_target(row, location)
    if meta.get("title_target_sha256") != text_sha256(target):
        fail(f"{location}: title_target_sha256 mismatch")
    pdf_sha = meta.get("source_pdf_sha256")
    if not isinstance(pdf_sha, str) or not HEX_SHA256.fullmatch(pdf_sha):
        fail(f"{location}: invalid source_pdf_sha256")

    page_indices = meta.get("page_indices")
    images = row.get("images")
    if (
        not isinstance(page_indices, list)
        or not page_indices
        or any(not isinstance(page, int) or isinstance(page, bool) or page < 0 for page in page_indices)
        or len(set(page_indices)) != len(page_indices)
        or page_indices != sorted(page_indices)
    ):
        fail(f"{location}: page_indices must be non-empty, unique, ordered non-negative integers")
    if meta.get("n_pages") != len(page_indices) or not isinstance(images, list) or len(images) != len(page_indices):
        fail(f"{location}: n_pages/page_indices/images mismatch")
    expected_suffixes = expected_image_suffixes(source, doc_id, page_indices)
    for image, suffix in zip(images, expected_suffixes):
        image_path = Path(image) if isinstance(image, str) else Path()
        if (
            not isinstance(image, str)
            or not image_path.is_absolute()
            or len(image_path.parts) < len(suffix.parts)
            or image_path.parts[-len(suffix.parts):] != suffix.parts
        ):
            fail(f"{location}: image path does not match its source/doc/page suffix")
    expected_prompt = "<image>document parsing." if len(images) == 1 else "<image>Multi page merge."
    if prompt != expected_prompt:
        fail(f"{location}: prompt does not match page count")

    if source.startswith("READoc-"):
        if pool != "readoc_full" or meta.get("sample_form") != "full_document":
            fail(f"{location}: READoc row must be readoc_full/full_document")
        source_short = source.removeprefix("READoc-")
        manifest = readoc_manifest.get((source_short, doc_id))
        silver = readoc_silver.get((row_id,))
        if manifest is None or silver is None:
            fail(f"{location}: READoc source/silver manifest entry missing")
        gt_path = DATA / "readoc_silver_v1" / "ground_truth" / source_short / f"{doc_id}.md"
        raw_gt_sha = sha256(gt_path) if gt_path.is_file() else ""
        if manifest.get("derived_gt") != f"ground_truth/{source_short}/{doc_id}.md":
            fail(f"{location}: READoc derived_gt path mismatch")
        if raw_gt_sha != manifest.get("derived_sha256") or raw_gt_sha != meta.get("derived_gt_sha256"):
            fail(f"{location}: READoc raw GT SHA mismatch")
        expected_silver = {
            "source": source,
            "doc_id": doc_id,
            "split": split,
            "pdf_sha256": pdf_sha,
            "pdf_pages": manifest.get("pdf_pages"),
            "derived_sha256": raw_gt_sha,
        }
        for field, expected in expected_silver.items():
            if silver.get(field) != expected:
                fail(f"{location}: READoc silver manifest {field} mismatch")
    else:
        manifest = pmc_manifest.get((doc_id,))
        if manifest is None:
            fail(f"{location}: PMC source manifest entry missing")
        if pool == "pmc_full" and meta.get("sample_form") == "full_document":
            gt_path = DATA / "pmc_silver_v2" / "ground_truth" / "full" / f"{doc_id}.md"
        elif pool == "pmc_single" and meta.get("sample_form") == "strict_single_page" and len(page_indices) == 1:
            gt_path = DATA / "pmc_silver_v2" / "ground_truth" / "single" / f"{doc_id}__p{page_indices[0]:04d}.md"
        else:
            fail(f"{location}: PMC pool/sample_form mismatch")

    pdf_pages = manifest.get("pdf_pages", manifest.get("pages"))
    if not isinstance(pdf_pages, int) or isinstance(pdf_pages, bool) or pdf_pages <= 0:
        fail(f"{location}: manifest PDF page count is invalid")
    if manifest.get("pdf_sha256") != pdf_sha:
        fail(f"{location}: row/source manifest PDF SHA mismatch")
    if meta.get("sample_form") == "full_document" and page_indices != list(range(pdf_pages)):
        fail(f"{location}: full-document pages do not cover the manifest PDF")
    if any(page >= pdf_pages for page in page_indices):
        fail(f"{location}: page index is outside the manifest PDF")

    if not gt_path.is_file():
        fail(f"{location}: standalone GT is missing: {gt_path.relative_to(DATA)}")
    gt_text = gt_path.read_text(encoding="utf-8")
    if canonical_target(gt_text) != target:
        fail(f"{location}: standalone GT does not match JSONL assistant target")
    if source == "PMC-v26-synthetic" and gt_text != target:
        fail(f"{location}: PMC standalone GT must byte-match the JSONL assistant target")
    expected_gt.add(gt_path.resolve())
    return row_id


def validate_contract() -> dict[str, int]:
    required = [
        DATA / "readoc_silver_v1/build_report.json",
        DATA / "readoc_silver_v1/silver_manifest.jsonl",
        DATA / "readoc_silver_v1/source_audit/document_manifest.jsonl",
        DATA / "readoc_silver_v1/source_audit/change_manifest.jsonl",
        DATA / "readoc_silver_v1/source_audit/validation.json",
        DATA / "readoc_silver_v1/source_audit/review_policy.json",
        DATA / "readoc_silver_v1/source_audit/summary.json",
        DATA / "pmc_silver_v2/build_report.json",
        DATA / "pmc_silver_v2/silver_decisions.jsonl",
        DATA / "pmc_silver_v2/source_audit/document_manifest.jsonl",
        DATA / "pmc_silver_v2/source_audit/title_candidates.jsonl",
        DATA / "final_mix_v1/mix_report.json",
    ]
    required.extend(
        DATA / "readoc_silver_v1/readoc_full" / f"{split}.jsonl" for split in SPLITS
    )
    for pool in ("pmc_full", "pmc_single"):
        required.extend(DATA / "pmc_silver_v2" / pool / f"{split}.jsonl" for split in SPLITS)
    required.extend(DATA / "final_mix_v1" / f"{split}.jsonl" for split in SPLITS)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        fail("missing required files: " + ", ".join(missing))

    for path in DATA.rglob("*"):
        if path.is_symlink():
            fail(f"symlink is not allowed: {path.relative_to(ROOT)}")
        if path.is_file() and ("__pycache__" in path.parts or path.name.endswith(FORBIDDEN_SUFFIXES)):
            fail(f"forbidden file in bundle: {path.relative_to(ROOT)}")

    readoc_manifest = load_unique(
        DATA / "readoc_silver_v1/source_audit/document_manifest.jsonl", ("source", "doc_id")
    )
    readoc_silver = load_unique(DATA / "readoc_silver_v1/silver_manifest.jsonl", ("id",))
    pmc_manifest = load_unique(DATA / "pmc_silver_v2/source_audit/document_manifest.jsonl", ("doc_id",))
    if len(readoc_silver) != EXPECTED_POOL_ROWS["readoc_full"]:
        fail(f"READoc silver manifest expected 1552 rows, found {len(readoc_silver)}")
    if len(pmc_manifest) != EXPECTED_POOL_ROWS["pmc_full"]:
        fail(f"PMC source manifest expected 1486 rows, found {len(pmc_manifest)}")

    source_rows: dict[str, dict] = {}
    source_pool: dict[str, str] = {}
    pool_rows: dict[str, list[dict]] = defaultdict(list)
    expected_gt: set[Path] = set()
    pool_roots = {
        "readoc_full": DATA / "readoc_silver_v1/readoc_full",
        "pmc_full": DATA / "pmc_silver_v2/pmc_full",
        "pmc_single": DATA / "pmc_silver_v2/pmc_single",
    }
    for pool, pool_root in pool_roots.items():
        for split in SPLITS:
            path = pool_root / f"{split}.jsonl"
            for line_number, row in rows(path):
                location = f"{path.relative_to(DATA)}:{line_number}"
                row_id = validate_source_row(
                    row,
                    pool=pool,
                    split=split,
                    location=location,
                    readoc_manifest=readoc_manifest,
                    readoc_silver=readoc_silver,
                    pmc_manifest=pmc_manifest,
                    expected_gt=expected_gt,
                )
                if row_id in source_rows:
                    fail(f"duplicate source row id: {row_id}")
                source_rows[row_id] = row
                source_pool[row_id] = pool
                pool_rows[pool].append(row)
        expected = EXPECTED_POOL_ROWS[pool]
        if len(pool_rows[pool]) != expected:
            fail(f"{pool} expected {expected} rows, found {len(pool_rows[pool])}")

    actual_gt = {
        path.resolve()
        for root in (DATA / "readoc_silver_v1/ground_truth", DATA / "pmc_silver_v2/ground_truth")
        for path in root.rglob("*.md")
    }
    if actual_gt != expected_gt:
        data_root = DATA.resolve()
        missing_gt = sorted(str(path.relative_to(data_root)) for path in expected_gt - actual_gt)
        extra_gt = sorted(str(path.relative_to(data_root)) for path in actual_gt - expected_gt)
        fail(f"standalone GT set mismatch; missing={missing_gt[:3]} extra={extra_gt[:3]}")

    def identity(row: dict) -> tuple[str, str, str]:
        meta = row.get("meta", {})
        return (str(meta.get("source", "")), str(meta.get("doc_id", "")), str(meta.get("split", "")))

    full_keys = {identity(row) for row in pool_rows["pmc_full"]}
    single_keys = {identity(row) for row in pool_rows["pmc_single"]}
    if full_keys != single_keys:
        fail("PMC full/single identities or splits do not match")

    mix_counts = {}
    pool_counts = Counter()
    mix_ids = set()
    for split, expected_count in EXPECTED_SPLIT_ROWS.items():
        path = DATA / "final_mix_v1" / f"{split}.jsonl"
        items = validate_jsonl(path, expected_count)
        mix_counts[split] = len(items)
        for line_number, item in enumerate(items, 1):
            row_id = item.get("id")
            pool = item.get("meta", {}).get("mix_pool")
            if row_id in mix_ids or row_id not in source_rows:
                fail(f"{path}:{line_number}: duplicate or unknown mixed row id {row_id!r}")
            if pool != source_pool[row_id]:
                fail(f"{path}:{line_number}: mixed row has incorrect pool {pool!r}")
            expected_row = json.loads(json.dumps(source_rows[row_id], ensure_ascii=False))
            expected_row["meta"]["mix_pool"] = pool
            expected_row["meta"]["mix_version"] = EXPECTED_MIX_VERSION
            if item != expected_row or item["meta"].get("split") != split:
                fail(f"{path}:{line_number}: mixed row differs from frozen source row {row_id}")
            mix_ids.add(row_id)
            pool_counts[pool] += 1
    if sum(mix_counts.values()) != len(source_rows) or mix_ids != set(source_rows):
        fail("final mix does not contain every frozen source row exactly once")
    if pool_counts != Counter(EXPECTED_POOL_ROWS):
        fail(f"unexpected final mix pool counts: {dict(pool_counts)}")
    return {
        "unique_pdf_docs": len(readoc_silver) + len(pmc_manifest),
        "standalone_gt": len(expected_gt),
        **mix_counts,
        **dict(pool_counts),
    }


def payload_files() -> list[Path]:
    result = []
    for path in DATA.rglob("*"):
        if not path.is_file() or path.name in {"STATUS_ZH.md", ".gitignore", "DATA_SHA256SUMS"}:
            continue
        if path.is_symlink():
            fail(f"symlink is not allowed: {path.relative_to(ROOT)}")
        result.append(path)
    return sorted(result, key=lambda path: path.relative_to(DATA).as_posix())


def write_checksums() -> Path:
    output = DATA / "DATA_SHA256SUMS"
    lines = [f"{sha256(path)}  {path.relative_to(DATA).as_posix()}" for path in payload_files()]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parse_checksums(content: str) -> dict[str, str]:
    result = {}
    for line_number, line in enumerate(content.splitlines(), 1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or not HEX_SHA256.fullmatch(parts[0]):
            fail(f"DATA_SHA256SUMS:{line_number}: invalid checksum line")
        digest, name = parts
        path = Path(name)
        if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            fail(f"DATA_SHA256SUMS:{line_number}: unsafe payload path {name!r}")
        if name in result:
            fail(f"DATA_SHA256SUMS:{line_number}: duplicate payload path {name!r}")
        result[name] = digest
    if not result:
        fail("DATA_SHA256SUMS is empty")
    return result


def verify_archive_checksums(archive: zipfile.ZipFile) -> int:
    prefix = "ms_swift_title_mask/data_assets/"
    checksum_entry = prefix + "DATA_SHA256SUMS"
    try:
        checksum_text = archive.read(checksum_entry).decode("utf-8")
    except (KeyError, UnicodeDecodeError) as exc:
        fail(f"ZIP checksum manifest is missing or invalid UTF-8: {exc}")
    expected = parse_checksums(checksum_text)
    excluded = {"STATUS_ZH.md", ".gitignore", "DATA_SHA256SUMS"}
    archived_payload = {
        name.removeprefix(prefix)
        for name in archive.namelist()
        if name.startswith(prefix)
        and not name.endswith("/")
        and name.removeprefix(prefix) not in excluded
    }
    if set(expected) != archived_payload:
        missing = sorted(set(expected) - archived_payload)
        extra = sorted(archived_payload - set(expected))
        fail(f"ZIP/checksum payload set mismatch; missing={missing[:3]} extra={extra[:3]}")
    for name, expected_digest in expected.items():
        actual = hashlib.sha256(archive.read(prefix + name)).hexdigest()
        if actual != expected_digest:
            fail(f"ZIP payload checksum mismatch: {name}")
    return len(expected)


def package(output: Path) -> tuple[str, int, int]:
    if output.exists():
        fail(f"refusing to overwrite existing output: {output}")
    if output.resolve().is_relative_to(ROOT):
        fail("ZIP output must be outside the bundle root")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            fail(f"symlink is not allowed in ZIP: {path.relative_to(ROOT)}")
    files = sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(FORBIDDEN_SUFFIXES)
        ),
        key=lambda p: p.relative_to(ROOT).as_posix(),
    )
    timestamp = (2026, 8, 23, 0, 0, 0)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(f"ms_swift_title_mask/{relative}", timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            fail("ZIP contains duplicate entry names")
        required = {
            "ms_swift_title_mask/data_assets/STATUS_ZH.md",
            "ms_swift_title_mask/data_assets/DATA_SHA256SUMS",
            "ms_swift_title_mask/data_assets/final_mix_v1/mix_report.json",
            "ms_swift_title_mask/data_assets/final_mix_v1/train.jsonl",
            "ms_swift_title_mask/data_assets/final_mix_v1/validation.jsonl",
            "ms_swift_title_mask/data_assets/final_mix_v1/test.jsonl",
            "ms_swift_title_mask/scripts/build_bundle_zip.py",
        }
        for pool in ("readoc_full", "pmc_full", "pmc_single"):
            prefix = "readoc_silver_v1/readoc_full" if pool == "readoc_full" else f"pmc_silver_v2/{pool}"
            required.update(f"ms_swift_title_mask/data_assets/{prefix}/{split}.jsonl" for split in SPLITS)
        missing = sorted(required - set(names))
        if missing:
            fail("ZIP is missing required entries: " + ", ".join(missing))
        if any(not name.startswith("ms_swift_title_mask/") for name in names):
            fail("ZIP root contains an entry outside ms_swift_title_mask/")
        if any(name.endswith(FORBIDDEN_SUFFIXES) or "__pycache__" in name for name in names):
            fail("ZIP contains a forbidden entry")
        verify_archive_checksums(archive)
    return sha256(output), output.stat().st_size, len(names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate only; do not write checksums or ZIP")
    parser.add_argument("--output", type=Path, help="reproducible ZIP destination (required without --check)")
    args = parser.parse_args()
    if not args.check and args.output is None:
        parser.error("--output is required unless --check is used")
    try:
        counts = validate_contract()
        if args.check:
            print(json.dumps({"check": "ok", "counts": counts}, sort_keys=True))
            return 0
        write_checksums()
        digest, size, files = package(args.output.resolve())
        print(json.dumps({"sha256": digest, "size": size, "files": files}, sort_keys=True))
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
