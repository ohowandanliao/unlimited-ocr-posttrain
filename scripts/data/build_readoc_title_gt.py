#!/usr/bin/env python3
"""Build a conservative, traceable READoc title-silver derivative."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, TextIO

from readoc_title_rules import (
    RULESET_VERSION,
    SCHEMA_VERSION,
    analyze_markdown,
    apply_line_decisions,
    compact_title,
    normalize_title,
    sha256_bytes,
    sha256_file,
    stable_id,
    validate_title_only_changes,
)


MANIFEST_SCHEMA_VERSION = "readoc-title-document-manifest-v1"
PDF_PROFILE_SCHEMA_VERSION = "readoc-pdf-profile-v1"
SUMMARY_SCHEMA_VERSION = "readoc-title-build-summary-v1"
DEFAULT_POLICY = Path(__file__).with_name("readoc_title_review_policy_v1.json")
DEFAULT_POLICY_SHA256 = "414b10d439035ce7cd3f3a14ce175c211c74995346e02780497946d391d8e0bb"
SOURCES = ("arxiv", "github")


def _json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_paths(input_root: Path, source: str) -> tuple[Path, Path]:
    return input_root / "ground_truth" / source, input_root / "archives" / f"{source}.zip"


def _archive_pdf_members(archive: zipfile.ZipFile, source: str) -> dict[str, str]:
    prefix = f"{source}/pdf/"
    members: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(prefix) or not info.filename.lower().endswith(".pdf"):
            continue
        doc_id = Path(info.filename).stem
        if doc_id in members:
            raise ValueError(f"duplicate PDF doc_id in {source}.zip: {doc_id}")
        members[doc_id] = info.filename
    return members


def _discover_documents(
    input_root: Path,
    selected_sources: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    documents: list[dict[str, Any]] = []
    archive_hashes: dict[str, str] = {}
    for source in selected_sources:
        gt_root, archive_path = _source_paths(input_root, source)
        if not gt_root.is_dir():
            raise FileNotFoundError(f"missing READoc ground truth directory: {gt_root}")
        if not archive_path.is_file():
            raise FileNotFoundError(f"missing READoc PDF archive: {archive_path}")
        gt_paths = {path.stem: path for path in gt_root.glob("*.md") if path.is_file()}
        with zipfile.ZipFile(archive_path) as archive:
            pdf_members = _archive_pdf_members(archive, source)
        missing_pdf = sorted(set(gt_paths) - set(pdf_members))
        missing_gt = sorted(set(pdf_members) - set(gt_paths))
        if missing_pdf or missing_gt:
            raise ValueError(
                f"{source} PDF/GT mismatch: missing_pdf={missing_pdf[:10]}, "
                f"missing_gt={missing_gt[:10]}"
            )
        archive_hashes[source] = sha256_file(archive_path)
        for doc_id in sorted(gt_paths):
            documents.append(
                {
                    "source": source,
                    "doc_id": doc_id,
                    "gt_path": gt_paths[doc_id],
                    "archive_path": archive_path,
                    "pdf_member": pdf_members[doc_id],
                }
            )
    return documents, archive_hashes


def _select_documents(
    documents: list[dict[str, Any]],
    doc_ids: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if doc_ids:
        counts = Counter(doc_ids)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate --doc-id values: {', '.join(duplicates)}")
        requested: set[tuple[str, str]] = set()
        for value in doc_ids:
            if "/" not in value:
                raise ValueError(f"--doc-id must be SOURCE/DOC_ID, got: {value}")
            source, doc_id = value.split("/", 1)
            if source not in SOURCES or not doc_id:
                raise ValueError(f"invalid --doc-id: {value}")
            requested.add((source, doc_id))
        available = {(item["source"], item["doc_id"]): item for item in documents}
        missing = sorted(f"{source}/{doc_id}" for source, doc_id in requested if (source, doc_id) not in available)
        if missing:
            raise FileNotFoundError("missing requested READoc documents: " + ", ".join(missing))
        documents = [available[key] for key in sorted(requested)]
    if limit is not None:
        documents = documents[:limit]
    if not documents:
        raise ValueError("no READoc documents selected")
    return documents


def _load_policy(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"default review policy SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("READoc review policy root must be an object")
    if policy.get("schema_version") != "readoc-title-review-policy-v1":
        raise ValueError(f"unsupported review policy schema: {policy.get('schema_version')}")
    if policy.get("ruleset_version") != RULESET_VERSION:
        raise ValueError(
            f"review policy ruleset {policy.get('ruleset_version')} != {RULESET_VERSION}"
        )
    policy.setdefault("page_evidence_overrides", [])
    for field in ("bulk_decisions", "document_decisions", "page_evidence_overrides"):
        if not isinstance(policy.get(field), list):
            raise ValueError(f"review policy {field} must be a list")
    policy["_artifact_sha256"] = actual_sha256
    return policy


def _safe_metadata(reader: Any) -> dict[str, str]:
    try:
        metadata = reader.metadata or {}
    except Exception:
        return {}
    result: dict[str, str] = {}
    for key in ("/Title", "/Author", "/Subject", "/Creator", "/Producer"):
        try:
            value = metadata.get(key)
        except Exception:
            continue
        if value is not None and str(value).strip():
            result[key.lstrip("/").casefold()] = " ".join(str(value).split())[:1000]
    return result


def _flatten_outline(reader: Any, items: Any, level: int = 1) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return records
    current_level = level
    for item in items:
        if isinstance(item, list):
            records.extend(_flatten_outline(reader, item, current_level + 1))
            continue
        title = getattr(item, "title", None)
        if title is None and isinstance(item, dict):
            title = item.get("/Title")
        if title is None:
            continue
        try:
            page = reader.get_destination_page_number(item) + 1
        except Exception:
            page = None
        records.append({"level": current_level, "text": str(title).strip(), "pdf_page": page})
    return records


def _first_page_profile(reader: Any) -> tuple[dict[str, Any], str]:
    fragments: list[dict[str, Any]] = []

    def visitor(text: str, cm: Any, tm: Any, font_dict: Any, font_size: Any) -> None:
        del cm, font_dict
        normalized = " ".join(str(text).split())
        if not normalized:
            return
        try:
            x = round(float(tm[4]), 2)
            y = round(float(tm[5]), 2)
            size = round(float(font_size), 2)
        except (IndexError, TypeError, ValueError):
            x = y = size = None
        fragments.append({"text": normalized[:500], "font_size": size, "x": x, "y": y})

    page = reader.pages[0]
    text = ""
    error = None
    try:
        text = page.extract_text(visitor_text=visitor) or ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
    unique: dict[tuple[str, float | None], dict[str, Any]] = {}
    for fragment in fragments:
        unique.setdefault((fragment["text"], fragment["font_size"]), fragment)
    largest = sorted(
        unique.values(),
        key=lambda item: (
            -(item["font_size"] if isinstance(item["font_size"], (int, float)) else -1),
            -len(item["text"]),
            item["text"],
        ),
    )[:40]
    profile = {
        "native_text_chars": len(compact_title(text)),
        "largest_text_fragments": largest,
    }
    if error:
        profile["extraction_warning"] = error
    return profile, text


def _following_body_anchor(content: str, heading_line: int) -> str:
    """Return a compact paragraph anchor after a heading for PDF page lookup."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in content.splitlines()[heading_line:]:
        stripped = line.strip()
        if stripped.startswith("#") and len(stripped) > 1:
            break
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))
    for paragraph in paragraphs:
        anchor = compact_title(paragraph)
        if len(anchor) >= 32:
            return anchor[:240]
    return ""


def _anchor_match_score(anchor: str, page_text: str) -> float:
    page = compact_title(page_text)
    if len(anchor) < 32 or not page:
        return 0.0
    prefix = anchor[: min(80, len(anchor))]
    if prefix in page:
        return 1.0
    width = min(20, max(12, len(anchor) // 8))
    sample = anchor[:240]
    shingles = {
        sample[index : index + width]
        for index in range(0, max(1, len(sample) - width + 1), max(6, width // 2))
        if len(sample[index : index + width]) == width
    }
    if not shingles:
        return 0.0
    return round(sum(shingle in page for shingle in shingles) / len(shingles), 6)


def _locate_body_anchors(
    reader: Any,
    anchors: dict[str, str],
    first_page_text: str,
) -> dict[str, dict[str, Any]]:
    if not anchors:
        return {}
    unresolved = set(anchors)
    best = {
        candidate_id: {"pdf_page": None, "score": 0.0}
        for candidate_id in anchors
    }
    for page_index in range(len(reader.pages)):
        if not unresolved:
            break
        if page_index == 0:
            page_text = first_page_text
        else:
            try:
                page_text = reader.pages[page_index].extract_text() or ""
            except Exception:
                page_text = ""
        for candidate_id in list(unresolved):
            score = _anchor_match_score(anchors[candidate_id], page_text)
            if score > best[candidate_id]["score"]:
                best[candidate_id] = {"pdf_page": page_index + 1, "score": score}
            if score >= 0.6:
                unresolved.remove(candidate_id)
    return {
        candidate_id: {
            "candidate_id": candidate_id,
            "anchor_sha256": sha256_bytes(anchor.encode("utf-8")),
            "pdf_page": best[candidate_id]["pdf_page"],
            "score": best[candidate_id]["score"],
            "matched": best[candidate_id]["score"] >= 0.6,
            "method": "following_paragraph_text_shingles",
        }
        for candidate_id, anchor in anchors.items()
    }


def _locate_heading_labels(
    reader: Any,
    labels: dict[str, str],
    first_page_text: str,
) -> dict[str, dict[str, Any]]:
    unresolved = {
        candidate_id: normalize_title(label)
        for candidate_id, label in labels.items()
        if normalize_title(label)
    }
    matches: dict[str, dict[str, Any]] = {}
    for page_index in range(len(reader.pages)):
        if not unresolved:
            break
        if page_index == 0:
            page_text = first_page_text
        else:
            try:
                page_text = reader.pages[page_index].extract_text() or ""
            except Exception:
                page_text = ""
        normalized_page = normalize_title(page_text)
        for candidate_id, label in list(unresolved.items()):
            if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", normalized_page):
                matches[candidate_id] = {
                    "candidate_id": candidate_id,
                    "pdf_page": page_index + 1,
                    "matched_text": label,
                    "method": "pdf_native_heading_label_exact",
                }
                del unresolved[candidate_id]
    return matches


def _pdf_profile(
    pdf_bytes: bytes,
    member: str,
    include_text_profile: bool,
    body_anchors: dict[str, str] | None = None,
    heading_labels: dict[str, str] | None = None,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for READoc PDF validation") from exc

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
        if reader.is_encrypted:
            decrypt_result = reader.decrypt("")
            if not decrypt_result:
                raise ValueError("encrypted PDF cannot be opened with an empty password")
        page_count = len(reader.pages)
    except Exception as exc:
        raise ValueError(f"unreadable PDF member {member}: {exc}") from exc
    if page_count <= 0:
        raise ValueError(f"PDF member has no pages: {member}")

    try:
        outline = _flatten_outline(reader, reader.outline)
    except Exception:
        outline = []
    profile: dict[str, Any] = {
        "schema_version": PDF_PROFILE_SCHEMA_VERSION,
        "pdf_member": member,
        "pdf_sha256": sha256_bytes(pdf_bytes),
        "pdf_bytes": len(pdf_bytes),
        "pages": page_count,
        "encrypted": bool(reader.is_encrypted),
        "metadata": _safe_metadata(reader),
        "outline_entries": outline[:1000],
        "outline_truncated": len(outline) > 1000,
    }
    first_page_text = ""
    anchor_matches: dict[str, dict[str, Any]] = {}
    label_matches: dict[str, dict[str, Any]] = {}
    if include_text_profile:
        first_page, first_page_text = _first_page_profile(reader)
        profile["first_page"] = first_page
        anchor_matches = _locate_body_anchors(reader, body_anchors or {}, first_page_text)
        label_matches = _locate_heading_labels(reader, heading_labels or {}, first_page_text)
        profile["title_body_anchor_matches"] = list(anchor_matches.values())
        profile["title_heading_label_matches"] = list(label_matches.values())
    return profile, first_page_text, anchor_matches, label_matches


def _candidate_pdf_evidence(
    candidates: list[dict[str, Any]],
    profile: dict[str, Any],
    first_page_text: str,
    body_anchor_matches: dict[str, dict[str, Any]],
    heading_label_matches: dict[str, dict[str, Any]],
) -> None:
    outline_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in profile.get("outline_entries", []):
        key = compact_title(str(entry.get("text", "")))
        if key:
            outline_index[key].append(entry)
    metadata_title = str(profile.get("metadata", {}).get("title", ""))
    metadata_key = compact_title(metadata_title)
    first_page_key = compact_title(first_page_text)
    large_fragments = profile.get("first_page", {}).get("largest_text_fragments", [])

    for candidate in candidates:
        label_match = heading_label_matches.get(candidate["candidate_id"])
        if label_match is not None:
            candidate["evidence"].append(
                {"type": "pdf_heading_label_match", **label_match}
            )
        body_match = body_anchor_matches.get(candidate["candidate_id"])
        if body_match is not None:
            candidate["evidence"].append(
                {
                    "type": "pdf_following_body_anchor_match"
                    if body_match["matched"]
                    else "pdf_following_body_anchor_unresolved",
                    **body_match,
                }
            )
        text = str(candidate.get("plain_text") or candidate.get("current_text") or "")
        key = compact_title(text)
        if not key:
            continue
        outline_matches = outline_index.get(key, [])
        if outline_matches:
            candidate["evidence"].append(
                {
                    "type": "pdf_outline_exact",
                    "matches": outline_matches[:5],
                }
            )
        if metadata_key and len(key) >= 8 and (key == metadata_key or key in metadata_key or metadata_key in key):
            candidate["evidence"].append(
                {"type": "pdf_metadata_title_match", "pdf_text": metadata_title}
            )
        if len(key) >= 8 and key in first_page_key:
            candidate["evidence"].append({"type": "pdf_first_page_text_match", "pdf_page": 1})
        fragment_matches = []
        for fragment in large_fragments:
            fragment_key = compact_title(str(fragment.get("text", "")))
            if fragment_key and len(key) >= 8 and (key == fragment_key or key in fragment_key or fragment_key in key):
                fragment_matches.append(fragment)
        if fragment_matches:
            candidate["evidence"].append(
                {"type": "pdf_first_page_large_text_match", "matches": fragment_matches[:5]}
            )


def _candidate_context(content: str, line: int, radius: int = 2) -> dict[str, Any]:
    lines = content.splitlines()
    start = max(1, line - radius)
    end = min(len(lines), line + radius)
    return {
        "start_line": start,
        "end_line": end,
        "lines": [
            {"line": number, "text": lines[number - 1][:1000]}
            for number in range(start, end + 1)
        ],
    }


def _candidate_pdf_page(candidate: dict[str, Any]) -> tuple[int | None, str | None]:
    for evidence in candidate["evidence"]:
        if evidence.get("type") == "pdf_following_body_anchor_match":
            return evidence.get("pdf_page"), "pdf_following_body_anchor_match"
    for evidence in candidate["evidence"]:
        if evidence.get("type") == "pdf_heading_label_match":
            return evidence.get("pdf_page"), "pdf_heading_label_match"
    for evidence in candidate["evidence"]:
        if evidence.get("type") in {
            "pdf_first_page_text_match",
            "pdf_first_page_large_text_match",
        }:
            return 1, evidence["type"]
    for evidence in candidate["evidence"]:
        if evidence.get("type") != "pdf_outline_exact":
            continue
        pages = [
            match.get("pdf_page")
            for match in evidence.get("matches", [])
            if isinstance(match.get("pdf_page"), int)
        ]
        if pages:
            return pages[0], "pdf_outline_exact"
    return None, None


def _accepted_decisions(
    *,
    source: str,
    doc_id: str,
    source_sha256: str,
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
    pdf_sha256: str,
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for rule in policy["bulk_decisions"]:
        if rule.get("decision") != "ACCEPT_SILVER" or rule.get("source") != source:
            continue
        for candidate in candidates:
            if (
                candidate.get("operation") == rule.get("operation")
                and candidate.get("current_level") == rule.get("current_level")
                and compact_title(str(candidate.get("plain_text", ""))) == compact_title(str(rule.get("normalized_text", "")))
                and candidate.get("source_kind") == rule.get("required_source_kind")
            ):
                pdf_page, pdf_page_evidence_type = _candidate_pdf_page(candidate)
                override = next(
                    (
                        item
                        for item in policy["page_evidence_overrides"]
                        if item.get("source") == source
                        and item.get("doc_id") == doc_id
                        and int(item.get("line", -1)) == int(candidate["line"])
                    ),
                    None,
                )
                if override is not None:
                    if override.get("source_sha256") != source_sha256:
                        raise ValueError(
                            f"page evidence source hash mismatch for {source}/{doc_id}"
                        )
                    if override.get("pdf_sha256") != pdf_sha256:
                        raise ValueError(
                            f"page evidence PDF hash mismatch for {source}/{doc_id}"
                        )
                    if (
                        not isinstance(override.get("pdf_page"), int)
                        or isinstance(override.get("pdf_page"), bool)
                        or int(override["pdf_page"]) < 1
                    ):
                        raise ValueError(
                            f"page evidence override has invalid PDF page for {source}/{doc_id}"
                        )
                    if (
                        not isinstance(override.get("evidence_method"), str)
                        or not override["evidence_method"].strip()
                    ):
                        raise ValueError(
                            f"page evidence override lacks evidence method for {source}/{doc_id}"
                        )
                    pdf_page = int(override["pdf_page"])
                    pdf_page_evidence_type = override["evidence_method"]
                    candidate["evidence"].append(
                        {
                            "type": "reviewed_pdf_page_override",
                            **override,
                        }
                    )
                accepted.append(
                    {
                        "change_id": stable_id("chg", rule["decision_id"], candidate["candidate_id"]),
                        "candidate_id": candidate["candidate_id"],
                        "decision_id": rule["decision_id"],
                        "decision": "ACCEPT_SILVER",
                        "review_scope": "versioned_bulk_policy",
                        "source": source,
                        "doc_id": doc_id,
                        "line": candidate["line"],
                        "operation": candidate["operation"],
                        "expected_source_line": candidate["source_line"],
                        "proposed_level": rule["proposed_level"],
                        "proposed_text": candidate["current_text"],
                        "reason": rule["reason"],
                        "evidence_documents": rule.get("evidence_documents", []),
                        "pdf_page": pdf_page,
                        "pdf_page_evidence_type": pdf_page_evidence_type,
                        "evidence": candidate["evidence"],
                    }
                )

    for decision in policy["document_decisions"]:
        if decision.get("decision") != "ACCEPT_SILVER":
            continue
        if decision.get("source") != source or decision.get("doc_id") != doc_id:
            continue
        if decision.get("source_sha256") != source_sha256:
            raise ValueError(
                f"review decision source hash mismatch for {source}/{doc_id}: "
                f"{decision.get('source_sha256')} != {source_sha256}"
            )
        if decision.get("pdf_sha256") != pdf_sha256:
            raise ValueError(
                f"review decision PDF hash mismatch for {source}/{doc_id}: "
                f"{decision.get('pdf_sha256')} != {pdf_sha256}"
            )
        if (
            not isinstance(decision.get("pdf_page"), int)
            or isinstance(decision.get("pdf_page"), bool)
            or int(decision["pdf_page"]) < 1
        ):
            raise ValueError(f"review decision has invalid PDF page for {source}/{doc_id}")
        if (
            not isinstance(decision.get("evidence_method"), str)
            or not decision["evidence_method"].strip()
        ):
            raise ValueError(f"review decision lacks evidence method for {source}/{doc_id}")
        record = dict(decision)
        matching = [
            candidate
            for candidate in candidates
            if int(candidate["line"]) == int(decision["line"])
            and (
                candidate["operation"] == decision["operation"]
                or decision["operation"] == "replace_heading"
            )
        ]
        if len(matching) != 1:
            raise ValueError(
                f"review decision must match exactly one candidate for {source}/{doc_id} "
                f"line {decision['line']}, got {len(matching)}"
            )
        record["candidate_id"] = matching[0]["candidate_id"]
        record["pdf_page_evidence_type"] = decision["evidence_method"]
        record["change_id"] = stable_id(
            "chg", decision["decision_id"], source_sha256, decision["line"]
        )
        record["review_scope"] = "document_pdf_review"
        accepted.append(record)
    return accepted


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
            path = self.directory / f"shard_{self.count // self.shard_size:05d}.jsonl"
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
    policy_path: Path = DEFAULT_POLICY,
    sources: list[str] | None = None,
    doc_ids: list[str] | None = None,
    limit: int | None = None,
    review_shard_size: int = 250,
    include_pdf_text_profile: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    policy_path = policy_path.resolve()
    if output_root == input_root or input_root in output_root.parents or output_root in input_root.parents:
        raise ValueError("input_root and output_root must be disjoint sibling trees")
    selected_sources = sources or list(SOURCES)
    if len(set(selected_sources)) != len(selected_sources) or any(source not in SOURCES for source in selected_sources):
        raise ValueError(f"sources must be unique values from {SOURCES}")
    expected_policy_sha = (
        DEFAULT_POLICY_SHA256 if policy_path == DEFAULT_POLICY.resolve() else None
    )
    policy = _load_policy(policy_path, expected_policy_sha)
    documents, archive_hashes_before = _discover_documents(input_root, selected_sources)
    documents = _select_documents(documents, doc_ids, limit)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    (temp_output / "ground_truth").mkdir()
    (temp_output / "review_shards").mkdir()

    totals: Counter[str] = Counter()
    source_risks: Counter[str] = Counter()
    derived_risks: Counter[str] = Counter()
    candidate_reasons: Counter[str] = Counter()
    candidate_statuses: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    changes_by_decision: Counter[str] = Counter()
    source_hashes_before: dict[tuple[str, str], str] = {}
    candidate_ids: set[str] = set()
    change_ids: set[str] = set()
    accepted_changes_missing_pdf_page: list[dict[str, Any]] = []
    accepted_changes_invalid_pdf_evidence: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    shard_writer = ReviewShardWriter(temp_output / "review_shards", review_shard_size)
    archives: dict[str, zipfile.ZipFile] = {}

    try:
        for source in selected_sources:
            archive_path = _source_paths(input_root, source)[1]
            archives[source] = zipfile.ZipFile(archive_path)
            (temp_output / "ground_truth" / source).mkdir()

        with (
            (temp_output / "document_manifest.jsonl").open("w", encoding="utf-8") as manifest_out,
            (temp_output / "pdf_profiles.jsonl").open("w", encoding="utf-8") as profiles_out,
            (temp_output / "heading_inventory.jsonl").open("w", encoding="utf-8") as inventory_out,
            (temp_output / "title_candidates.jsonl").open("w", encoding="utf-8") as candidates_out,
            (temp_output / "review_queue.jsonl").open("w", encoding="utf-8") as review_out,
            (temp_output / "change_manifest.jsonl").open("w", encoding="utf-8") as changes_out,
        ):
            for item in documents:
                source = item["source"]
                doc_id = item["doc_id"]
                gt_path: Path = item["gt_path"]
                totals["documents"] += 1
                raw_bytes = gt_path.read_bytes()
                source_sha = sha256_bytes(raw_bytes)
                source_hashes_before[(source, doc_id)] = source_sha
                try:
                    content = raw_bytes.decode("utf-8", errors="strict")
                    if not content.strip():
                        raise ValueError("ground-truth Markdown is empty")
                    analysis = analyze_markdown(
                        source=source,
                        doc_id=doc_id,
                        content=content,
                        source_sha256=source_sha,
                    )
                    body_anchors = {
                        candidate["candidate_id"]: anchor
                        for candidate in analysis["candidates"]
                        if candidate["operation"] == "change_level"
                        and (
                            anchor := _following_body_anchor(content, int(candidate["line"]))
                        )
                    }
                    heading_labels = {
                        candidate["candidate_id"]: str(candidate["plain_text"])
                        for candidate in analysis["candidates"]
                        if candidate["operation"] == "change_level"
                    }
                    pdf_bytes = archives[source].read(item["pdf_member"])
                    (
                        profile,
                        first_page_text,
                        body_anchor_matches,
                        heading_label_matches,
                    ) = _pdf_profile(
                        pdf_bytes,
                        item["pdf_member"],
                        include_pdf_text_profile,
                        body_anchors,
                        heading_labels,
                    )
                    _candidate_pdf_evidence(
                        analysis["candidates"],
                        profile,
                        first_page_text,
                        body_anchor_matches,
                        heading_label_matches,
                    )
                    decisions = _accepted_decisions(
                        source=source,
                        doc_id=doc_id,
                        source_sha256=source_sha,
                        candidates=analysis["candidates"],
                        policy=policy,
                        pdf_sha256=profile["pdf_sha256"],
                    )
                    target, applied = apply_line_decisions(content, decisions)
                    validation_errors = validate_title_only_changes(content, target, applied)
                    if validation_errors:
                        raise ValueError(f"title-only validation failed: {validation_errors[:3]}")
                    derived_analysis = analyze_markdown(
                        source=source,
                        doc_id=doc_id,
                        content=target,
                        source_sha256=sha256_bytes(target.encode("utf-8")),
                    )
                except Exception as exc:
                    totals["failed_documents"] += 1
                    record = {
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "ruleset_version": RULESET_VERSION,
                        "source": source,
                        "doc_id": doc_id,
                        "source_gt": gt_path.relative_to(input_root).as_posix(),
                        "source_sha256": source_sha,
                        "source_pdf_archive": item["archive_path"].relative_to(input_root).as_posix(),
                        "source_pdf_member": item["pdf_member"],
                        "status": "ERROR",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    manifest_records.append(record)
                    manifest_out.write(_json_line(record))
                    continue

                profile.update({"source": source, "doc_id": doc_id})
                profiles_out.write(_json_line(profile))
                accepted_by_candidate = {
                    decision["candidate_id"]: decision
                    for decision in applied
                    if decision.get("candidate_id")
                }
                for heading in analysis["headings"]:
                    inventory_out.write(
                        _json_line(
                            {
                                "schema_version": "readoc-heading-inventory-v1",
                                "ruleset_version": RULESET_VERSION,
                                "source": source,
                                "doc_id": doc_id,
                                "source_sha256": source_sha,
                                **heading,
                            }
                        )
                    )
                for candidate in analysis["candidates"]:
                    candidate_id = candidate["candidate_id"]
                    if candidate_id in candidate_ids:
                        raise RuntimeError(f"duplicate READoc candidate_id: {candidate_id}")
                    candidate_ids.add(candidate_id)
                    accepted_decision = accepted_by_candidate.get(candidate_id)
                    unresolved_reasons = set(candidate["reason_codes"])
                    if accepted_decision is not None:
                        if accepted_decision.get("review_scope") == "versioned_bulk_policy":
                            unresolved_reasons.discard("ARXIV_H6_ABSTRACT_CONVENTION")
                        else:
                            unresolved_reasons.clear()
                    if "NO_H1" not in derived_analysis["document_risks"]:
                        unresolved_reasons.discard("DOCUMENT_HAS_NO_H1")
                        unresolved_reasons.discard("MISSING_H1_FIRST_TEXT_CANDIDATE")
                    if "MULTIPLE_H1" not in derived_analysis["document_risks"]:
                        unresolved_reasons.discard("DOCUMENT_HAS_MULTIPLE_H1")
                    if "EMPTY_HEADING" not in derived_analysis["document_risks"]:
                        unresolved_reasons.discard("EMPTY_HEADING")
                    candidate["unresolved_reason_codes"] = sorted(unresolved_reasons)
                    if accepted_decision is not None and not unresolved_reasons:
                        candidate["review_status"] = "POLICY_ACCEPTED_SILVER"
                        candidate["label_grade"] = "silver_policy"
                    elif not unresolved_reasons and candidate["review_status"] == "REVIEW_REQUIRED":
                        candidate["review_status"] = "RESOLVED_BY_ACCEPTED_CHANGE"
                        candidate["label_grade"] = "silver_policy"
                    elif unresolved_reasons:
                        candidate["review_status"] = "REVIEW_REQUIRED"
                        candidate["label_grade"] = "review_candidate"
                    candidates_out.write(_json_line(candidate))
                    operations[candidate["operation"]] += 1
                    candidate_statuses[candidate["review_status"]] += 1
                    candidate_reasons.update(candidate["reason_codes"])
                    if candidate["review_status"] == "REVIEW_REQUIRED":
                        review = dict(candidate)
                        review["context"] = _candidate_context(content, int(candidate["line"]))
                        review_out.write(_json_line(review))
                        shard_writer.write(review)

                for decision in applied:
                    change_id = decision["change_id"]
                    if change_id in change_ids:
                        raise RuntimeError(f"duplicate READoc change_id: {change_id}")
                    change_ids.add(change_id)
                    decision["source_sha256"] = source_sha
                    decision["target_sha256"] = sha256_bytes(target.encode("utf-8"))
                    decision["pdf_sha256"] = profile["pdf_sha256"]
                    decision["schema_version"] = "readoc-title-change-v1"
                    decision["ruleset_version"] = RULESET_VERSION
                    decision["policy_schema_version"] = policy["schema_version"]
                    decision["policy_sha256"] = policy["_artifact_sha256"]
                    if include_pdf_text_profile and decision.get("pdf_page") is None:
                        accepted_changes_missing_pdf_page.append(
                            {
                                "change_id": change_id,
                                "source": source,
                                "doc_id": doc_id,
                                "line": decision["line"],
                            }
                        )
                    if include_pdf_text_profile and (
                        not isinstance(decision.get("pdf_page"), int)
                        or isinstance(decision.get("pdf_page"), bool)
                        or not 1 <= int(decision["pdf_page"]) <= int(profile["pages"])
                        or not isinstance(decision.get("pdf_page_evidence_type"), str)
                        or not decision["pdf_page_evidence_type"].strip()
                    ):
                        accepted_changes_invalid_pdf_evidence.append(
                            {
                                "change_id": change_id,
                                "source": source,
                                "doc_id": doc_id,
                                "line": decision["line"],
                                "pdf_page": decision.get("pdf_page"),
                                "pdf_pages": profile["pages"],
                                "pdf_page_evidence_type": decision.get(
                                    "pdf_page_evidence_type"
                                ),
                            }
                        )
                    changes_out.write(_json_line(decision))
                    changes_by_decision[decision["decision_id"]] += 1

                target_path = temp_output / "ground_truth" / source / f"{doc_id}.md"
                target_bytes = target.encode("utf-8")
                target_path.write_bytes(target_bytes)
                source_risks.update(analysis["document_risks"])
                derived_risks.update(derived_analysis["document_risks"])
                if analysis["document_risks"]:
                    totals["source_documents_with_title_risks"] += 1
                if derived_analysis["document_risks"]:
                    totals["derived_documents_review_required"] += 1
                    title_status = "REVIEW_REQUIRED"
                else:
                    totals["derived_documents_silver_candidates"] += 1
                    title_status = "SILVER_CANDIDATE"
                totals["source_headings"] += analysis["heading_count"]
                totals["derived_headings"] += derived_analysis["heading_count"]
                totals["pdf_pages"] += profile["pages"]
                totals["accepted_changes"] += len(applied)
                record = {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "ruleset_version": RULESET_VERSION,
                    "policy_schema_version": policy["schema_version"],
                    "source": source,
                    "doc_id": doc_id,
                    "source_gt": gt_path.relative_to(input_root).as_posix(),
                    "source_sha256": source_sha,
                    "derived_gt": target_path.relative_to(temp_output).as_posix(),
                    "derived_sha256": sha256_bytes(target_bytes),
                    "source_pdf_archive": item["archive_path"].relative_to(input_root).as_posix(),
                    "source_pdf_member": item["pdf_member"],
                    "pdf_sha256": profile["pdf_sha256"],
                    "pdf_pages": profile["pages"],
                    "source_heading_count": analysis["heading_count"],
                    "derived_heading_count": derived_analysis["heading_count"],
                    "source_title_risks": analysis["document_risks"],
                    "derived_title_risks": derived_analysis["document_risks"],
                    "accepted_change_count": len(applied),
                    "pdf_gt_pair_valid": True,
                    "full_ce_eligible": None,
                    "full_ce_status": "NOT_EVALUATED_BY_TITLE_PIPELINE",
                    "title_profile_status": title_status,
                    "title_weighted_eligible": None,
                    "title_weighted_silver_candidate": title_status == "SILVER_CANDIDATE",
                    "title_label_grade": "silver",
                    "title_only_scope_valid": True,
                    "status": "OK",
                }
                manifest_records.append(record)
                manifest_out.write(_json_line(record))

        shard_writer.close()
        for archive in archives.values():
            archive.close()
        archives.clear()

        source_mutations = []
        for item in documents:
            key = (item["source"], item["doc_id"])
            after = sha256_file(item["gt_path"])
            if after != source_hashes_before[key]:
                source_mutations.append(
                    {"source": key[0], "doc_id": key[1], "before": source_hashes_before[key], "after": after}
                )
        archive_hashes_after = {
            source: sha256_file(_source_paths(input_root, source)[1])
            for source in selected_sources
        }
        archive_mutations = {
            source: {"before": archive_hashes_before[source], "after": archive_hashes_after[source]}
            for source in selected_sources
            if archive_hashes_before[source] != archive_hashes_after[source]
        }
        expected_pairs = {(item["source"], item["doc_id"]) for item in documents}
        output_pairs = {
            (source, path.stem)
            for source in selected_sources
            for path in (temp_output / "ground_truth" / source).glob("*.md")
        }
        coverage_missing = sorted(f"{source}/{doc_id}" for source, doc_id in expected_pairs - output_pairs)
        coverage_extra = sorted(f"{source}/{doc_id}" for source, doc_id in output_pairs - expected_pairs)
        valid = (
            not totals["failed_documents"]
            and not source_mutations
            and not archive_mutations
            and not coverage_missing
            and not coverage_extra
            and (not include_pdf_text_profile or not accepted_changes_missing_pdf_page)
            and (not include_pdf_text_profile or not accepted_changes_invalid_pdf_evidence)
            and len(manifest_records) == len(documents)
        )
        validation = {
            "schema_version": "readoc-title-validation-v1",
            "ruleset_version": RULESET_VERSION,
            "valid": valid,
            "planned_documents": len(documents),
            "manifest_documents": len(manifest_records),
            "derived_documents": len(output_pairs),
            "source_mutations": source_mutations,
            "archive_mutations": archive_mutations,
            "coverage_missing": coverage_missing,
            "coverage_extra": coverage_extra,
            "accepted_changes_missing_pdf_page": accepted_changes_missing_pdf_page,
            "accepted_changes_invalid_pdf_evidence": accepted_changes_invalid_pdf_evidence,
            "title_only_scope_errors": sum(
                record.get("title_only_scope_valid") is False for record in manifest_records
            ),
        }
        _write_json(temp_output / "validation.json", validation)
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "candidate_schema_version": SCHEMA_VERSION,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "ruleset_version": RULESET_VERSION,
            "policy_schema_version": policy["schema_version"],
            "policy_sha256": policy["_artifact_sha256"],
            "builder_sha256": sha256_file(Path(__file__)),
            "rules_module_sha256": sha256_file(Path(__file__).with_name("readoc_title_rules.py")),
            "input_layout": "ground_truth/{arxiv,github}/*.md + archives/{arxiv,github}.zip",
            "output_layout": "ground_truth/{arxiv,github}/*.md plus audit JSONL",
            "configuration": {
                "sources": selected_sources,
                "selected_doc_ids": sorted(doc_ids) if doc_ids else None,
                "limit": limit,
                "review_shard_size": review_shard_size,
                "include_pdf_text_profile": include_pdf_text_profile,
                "policy_file": policy_path.name,
            },
            "archive_sha256": archive_hashes_before,
            "totals": dict(sorted(totals.items())),
            "source_document_risk_counts": dict(sorted(source_risks.items())),
            "derived_document_risk_counts": dict(sorted(derived_risks.items())),
            "candidate_operations": dict(sorted(operations.items())),
            "candidate_review_statuses": dict(sorted(candidate_statuses.items())),
            "candidate_reason_counts": dict(sorted(candidate_reasons.items())),
            "accepted_changes_by_decision": dict(sorted(changes_by_decision.items())),
            "review_shards": shard_writer.paths,
            "validation": {key: validation[key] for key in ("valid", "planned_documents", "derived_documents")},
            "label_note": "Derived hierarchy is conservative silver. SILVER_CANDIDATE is not final title-weighted eligibility; full-content and visual acceptance remain separate, and no model review is declared gold.",
        }
        _write_json(temp_output / "summary.json", summary)
        shutil.copy2(policy_path, temp_output / "review_policy.json")
        _replace_output(temp_output, output_root, overwrite)
        if not valid:
            raise RuntimeError(f"READoc title build failed validation; see {output_root / 'validation.json'}")
        return summary
    except BaseException:
        shard_writer.close()
        for archive in archives.values():
            archive.close()
        if temp_output.exists():
            shutil.rmtree(temp_output)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--source", action="append", choices=SOURCES, dest="sources")
    parser.add_argument("--doc-id", action="append", dest="doc_ids", help="SOURCE/DOC_ID")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--review-shard-size", type=int, default=250)
    parser.add_argument("--no-pdf-text-profile", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        summary = build_dataset(
            input_root=args.input_root,
            output_root=args.output_root,
            policy_path=args.policy,
            sources=args.sources,
            doc_ids=args.doc_ids,
            limit=args.limit,
            review_shard_size=args.review_shard_size,
            include_pdf_text_profile=not args.no_pdf_text_profile,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, json.JSONDecodeError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
