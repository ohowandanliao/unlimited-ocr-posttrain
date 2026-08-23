"""CommonMark-aware READoc title analysis and title-only transformations."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt


SCHEMA_VERSION = "readoc-title-candidates-v1"
RULESET_VERSION = "readoc-title-rules-2026-08-23-v1"

_ATX_LINE_RE = re.compile(
    r"^(?P<indent> {0,3})(?P<marks>#{1,6})(?P<separator>[ \t]+|(?=\r?$))"
    r"(?P<rest>.*?)(?P<newline>\r?\n?)$"
)
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+){0,5})(?:\.)?(?:\s+|(?=[A-Z]))(?P<body>\S.*)$"
)
_TRAILING_TITLE_PUNCTUATION_RE = re.compile(r"[\s.:：。]+$")
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(canonical_json_bytes(parts)).hexdigest()[:24]
    return f"{prefix}_{digest}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_title(value: str) -> str:
    """Normalize title text for evidence matching, never for output rewriting."""
    value = html.unescape(value).replace("\u00a0", " ")
    value = re.sub(r"[`*_~]", "", value)
    value = " ".join(value.split())
    value = _TRAILING_TITLE_PUNCTUATION_RE.sub("", value)
    return value.casefold()


def compact_title(value: str) -> str:
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", normalize_title(value))


def numbered_depth(value: str) -> int | None:
    match = _NUMBERED_HEADING_RE.match(" ".join(value.split()))
    if not match:
        return None
    parts = match.group("number").split(".")
    if any(int(part) > 99 for part in parts):
        return None
    return len(parts)


def _inline_plain_text(token: Any) -> str:
    children = getattr(token, "children", None) or []
    if not children:
        return token.content.strip()
    parts: list[str] = []
    for child in children:
        if child.type in {"text", "code_inline", "html_inline"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
        elif child.type == "image":
            parts.append(child.content)
    return " ".join("".join(parts).split())


def parse_headings(content: str) -> list[dict[str, Any]]:
    """Return CommonMark headings with exact source line locations."""
    parser = MarkdownIt("commonmark")
    tokens = parser.parse(content)
    lines = content.splitlines(keepends=True)
    headings: list[dict[str, Any]] = []
    for position, token in enumerate(tokens):
        if token.type != "heading_open" or not token.map:
            continue
        inline = tokens[position + 1] if position + 1 < len(tokens) else None
        if inline is None or inline.type != "inline":
            raw_text = ""
            plain_text = ""
        else:
            raw_text = inline.content.strip()
            plain_text = _inline_plain_text(inline)
        line_start, line_end = token.map
        source_line = lines[line_start] if line_start < len(lines) else ""
        atx_match = _ATX_LINE_RE.fullmatch(source_line) if line_end == line_start + 1 else None
        level = int(token.tag[1:])
        headings.append(
            {
                "index": len(headings) + 1,
                "line": line_start + 1,
                "line_end": line_end,
                "level": level,
                "text": raw_text,
                "plain_text": plain_text,
                "normalized_text": normalize_title(plain_text),
                "markup": token.markup,
                "source_kind": "atx_top_level" if atx_match else "setext_or_container",
                "editable_atx": bool(atx_match and len(atx_match.group("marks")) == level),
                "source_line": source_line.rstrip("\r\n"),
            }
        )
    return headings


def first_plain_line_candidate(content: str, heading_lines: set[int]) -> dict[str, Any] | None:
    lines = content.splitlines()
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        if line_number in heading_lines:
            return None
        if (
            len(stripped) > 320
            or stripped.startswith(("```", "~~~", "<!--", "<", "![", "[!", "|"))
            or re.match(r"^(?:[-+*]|\d+[.)])\s+", stripped)
        ):
            return None
        return {"line": line_number, "text": stripped, "source_line": line}
    return None


def _candidate_base(
    *,
    source: str,
    doc_id: str,
    source_sha256: str,
    line: int,
    operation: str,
    current_level: int | None,
    current_text: str,
) -> dict[str, Any]:
    candidate_id = stable_id(
        "cand",
        RULESET_VERSION,
        source,
        doc_id,
        source_sha256,
        line,
        operation,
        current_level,
        current_text,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "ruleset_version": RULESET_VERSION,
        "candidate_id": candidate_id,
        "source": source,
        "doc_id": doc_id,
        "source_sha256": source_sha256,
        "line": line,
        "operation": operation,
        "current_level": current_level,
        "current_text": current_text,
        "proposed_level": current_level,
        "proposed_text": current_text,
        "reason_codes": [],
        "evidence": [],
        "review_status": "AUTO_SILVER_RETAIN",
        "label_grade": "silver_source",
    }


def analyze_markdown(
    *,
    source: str,
    doc_id: str,
    content: str,
    source_sha256: str,
) -> dict[str, Any]:
    if source not in {"arxiv", "github"}:
        raise ValueError(f"unsupported READoc source: {source}")
    headings = parse_headings(content)
    h1_count = sum(heading["level"] == 1 for heading in headings)
    duplicate_counts = Counter(
        heading["normalized_text"]
        for heading in headings
        if heading["normalized_text"]
    )
    document_risks: set[str] = set()
    if not headings:
        document_risks.add("NO_HEADINGS")
    if h1_count == 0:
        document_risks.add("NO_H1")
    elif h1_count > 1:
        document_risks.add("MULTIPLE_H1")

    candidates: list[dict[str, Any]] = []
    previous_structural_level: int | None = None
    for heading_position, heading in enumerate(headings):
        candidate = _candidate_base(
            source=source,
            doc_id=doc_id,
            source_sha256=source_sha256,
            line=heading["line"],
            operation="retain",
            current_level=heading["level"],
            current_text=heading["text"],
        )
        candidate["heading_index"] = heading["index"]
        candidate["plain_text"] = heading["plain_text"]
        candidate["source_kind"] = heading["source_kind"]
        candidate["editable_atx"] = heading["editable_atx"]
        candidate["source_line"] = heading["source_line"]
        reasons: set[str] = set()

        is_arxiv_abstract_h6 = (
            source == "arxiv"
            and heading["level"] == 6
            and heading["normalized_text"] == "abstract"
        )
        if not heading["normalized_text"]:
            reasons.add("EMPTY_HEADING")
        if duplicate_counts[heading["normalized_text"]] > 1 and heading["normalized_text"]:
            reasons.add("DUPLICATE_HEADING_TEXT")
        if len(heading["plain_text"]) > 200 or len(_WORD_RE.findall(heading["plain_text"])) > 40:
            reasons.add("HEADING_TOO_LONG")

        depth = numbered_depth(heading["plain_text"])
        if depth is not None:
            expected_level = min(6, depth + (1 if h1_count == 1 else 0))
            candidate["evidence"].append(
                {
                    "type": "explicit_numbering",
                    "numbered_depth": depth,
                    "expected_level_if_semantic": expected_level,
                }
            )
            if heading["level"] != expected_level:
                candidate["evidence"].append(
                    {
                        "type": "number_level_conflict",
                        "current_level": heading["level"],
                        "expected_level_if_semantic": expected_level,
                    }
                )

        if is_arxiv_abstract_h6:
            reasons.add("ARXIV_H6_ABSTRACT_CONVENTION")
            candidate["operation"] = "change_level"
            candidate["proposed_level"] = 2
            candidate["review_status"] = "POLICY_ACCEPTED_SILVER"
            candidate["label_grade"] = "silver_policy"
            candidate["evidence"].append(
                {
                    "type": "dataset_format_convention",
                    "policy_id": "ARXIV_H6_ABSTRACT_TO_H2_V1",
                }
            )
        else:
            if (
                previous_structural_level is not None
                and heading["level"] > previous_structural_level + 1
            ):
                reasons.add("LEVEL_SKIP")
                candidate["evidence"].append(
                    {"type": "previous_structural_level", "level": previous_structural_level}
                )
            previous_structural_level = heading["level"]

        if h1_count > 1 and heading["level"] == 1:
            reasons.add("DOCUMENT_HAS_MULTIPLE_H1")
        if h1_count == 0 and heading_position == 0:
            reasons.add("DOCUMENT_HAS_NO_H1")
        candidate["reason_codes"] = sorted(reasons)
        unresolved = reasons - {"ARXIV_H6_ABSTRACT_CONVENTION"}
        if unresolved:
            candidate["review_status"] = "REVIEW_REQUIRED"
            candidate["label_grade"] = "review_candidate"
            document_risks.update(
                unresolved - {"DOCUMENT_HAS_NO_H1", "DOCUMENT_HAS_MULTIPLE_H1"}
            )
        candidates.append(candidate)

    heading_lines = {heading["line"] for heading in headings}
    if h1_count == 0:
        promotion = first_plain_line_candidate(content, heading_lines)
        if promotion is not None:
            candidate = _candidate_base(
                source=source,
                doc_id=doc_id,
                source_sha256=source_sha256,
                line=promotion["line"],
                operation="promote",
                current_level=None,
                current_text=promotion["text"],
            )
            candidate.update(
                {
                    "source_line": promotion["source_line"],
                    "proposed_level": 1,
                    "proposed_text": promotion["text"],
                    "reason_codes": ["MISSING_H1_FIRST_TEXT_CANDIDATE"],
                    "review_status": "REVIEW_REQUIRED",
                    "label_grade": "review_candidate",
                }
            )
            candidates.append(candidate)

    return {
        "headings": headings,
        "candidates": candidates,
        "document_risks": sorted(document_risks),
        "heading_count": len(headings),
        "h1_count": h1_count,
        "heading_levels": dict(sorted(Counter(h["level"] for h in headings).items())),
    }


def replace_atx_level(source_line: str, proposed_level: int) -> str:
    if not 1 <= proposed_level <= 6:
        raise ValueError(f"invalid proposed ATX level: {proposed_level}")
    newline = ""
    body = source_line
    if body.endswith("\r\n"):
        body, newline = body[:-2], "\r\n"
    elif body.endswith("\n"):
        body, newline = body[:-1], "\n"
    match = _ATX_LINE_RE.fullmatch(body)
    if not match:
        raise ValueError(f"line is not an editable top-level ATX heading: {body!r}")
    return (
        match.group("indent")
        + ("#" * proposed_level)
        + match.group("separator")
        + match.group("rest")
        + newline
    )


def heading_line(level: int, text: str, newline: str) -> str:
    if not 1 <= level <= 6:
        raise ValueError(f"invalid heading level: {level}")
    return f"{'#' * level} {text}{newline}"


def apply_line_decisions(
    content: str,
    decisions: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    lines = content.splitlines(keepends=True)
    seen_lines: set[int] = set()
    applied: list[dict[str, Any]] = []
    for decision in sorted(decisions, key=lambda item: int(item["line"])):
        line_number = int(decision["line"])
        if line_number in seen_lines:
            raise ValueError(f"multiple accepted decisions target line {line_number}")
        seen_lines.add(line_number)
        if not 1 <= line_number <= len(lines):
            raise ValueError(f"decision line is out of range: {line_number}")
        source_line = lines[line_number - 1]
        expected_source_line = decision.get("expected_source_line")
        if expected_source_line is not None and source_line.rstrip("\r\n") != expected_source_line:
            raise ValueError(
                f"decision source mismatch at line {line_number}: "
                f"{source_line.rstrip(chr(10) + chr(13))!r} != {expected_source_line!r}"
            )
        operation = decision["operation"]
        if operation == "change_level":
            target_line = replace_atx_level(source_line, int(decision["proposed_level"]))
        elif operation == "promote":
            newline = "\r\n" if source_line.endswith("\r\n") else "\n" if source_line.endswith("\n") else ""
            target_line = heading_line(
                int(decision.get("proposed_level", 1)),
                str(decision.get("proposed_text") or source_line.strip()),
                newline,
            )
        elif operation == "replace_heading":
            newline = "\r\n" if source_line.endswith("\r\n") else "\n" if source_line.endswith("\n") else ""
            target_line = heading_line(
                int(decision["proposed_level"]),
                str(decision["proposed_text"]),
                newline,
            )
        else:
            raise ValueError(f"unsupported accepted operation: {operation}")
        lines[line_number - 1] = target_line
        record = dict(decision)
        record["source_line"] = source_line.rstrip("\r\n")
        record["target_line"] = target_line.rstrip("\r\n")
        applied.append(record)
    return "".join(lines), applied


def validate_title_only_changes(
    source_content: str,
    target_content: str,
    applied_decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_lines = source_content.splitlines(keepends=True)
    target_lines = target_content.splitlines(keepends=True)
    errors: list[dict[str, Any]] = []
    if len(source_lines) != len(target_lines):
        return [
            {
                "type": "line_count_changed",
                "source_lines": len(source_lines),
                "target_lines": len(target_lines),
            }
        ]
    decisions_by_line = {int(item["line"]): item for item in applied_decisions}
    for line_number, (source_line, target_line) in enumerate(
        zip(source_lines, target_lines), 1
    ):
        decision = decisions_by_line.get(line_number)
        if source_line == target_line:
            if decision is not None and decision.get("source_line") != decision.get("target_line"):
                errors.append({"type": "accepted_decision_not_applied", "line": line_number})
            continue
        if decision is None:
            errors.append(
                {
                    "type": "unreviewed_line_changed",
                    "line": line_number,
                    "source": source_line.rstrip("\r\n"),
                    "target": target_line.rstrip("\r\n"),
                }
            )
            continue
        if (
            source_line.rstrip("\r\n") != decision.get("source_line")
            or target_line.rstrip("\r\n") != decision.get("target_line")
        ):
            errors.append({"type": "decision_diff_mismatch", "line": line_number})
    changed_lines = {
        line_number
        for line_number, (source_line, target_line) in enumerate(
            zip(source_lines, target_lines), 1
        )
        if source_line != target_line
    }
    extra_decisions = sorted(set(decisions_by_line) - changed_lines)
    if extra_decisions:
        errors.append({"type": "decisions_without_changes", "lines": extra_decisions})
    return errors
