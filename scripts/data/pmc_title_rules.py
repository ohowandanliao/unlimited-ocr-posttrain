"""Deterministic, non-destructive title candidate rules for PMC middle.json files."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "pmc-title-candidates-v1"
RULESET_VERSION = "pmc-title-rules-2026-08-23-v1"

_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+){0,5})(?P<trailing>\.)?\s+(?P<body>\S.*)$"
)
_CAPTION_RE = re.compile(
    r"^\s*(?:fig(?:ure)?|table|scheme|chart|plate|box|algorithm|equation|extended\s+data|supplementary\s+"
    r"(?:fig(?:ure)?|table))\s*[A-Za-z]?\d+(?:[\s.:)]|$)",
    re.IGNORECASE,
)
_LONG_DESCRIPTION_RE = re.compile(r"\blong\s+descriptions?\b", re.IGNORECASE)
_HTML_TABLE_RE = re.compile(r"<table(?:\s|>)", re.IGNORECASE)
_REVIEW_MATERIAL_RE = re.compile(
    r"(?:^Review\s*:|^Reviewer$|Reviewer\s*#?\s*\d+|Reviewer\s+declares|"
    r"Open\s+peer\s+review|Public\s+review|Major\s+Comments?|"
    r"Additional\s+Editor\s+Comments?|Review\s+Comments\s+to\s+the\s+Author|"
    r"Authors?[’']?\s+Response|Author\s+response|\bResponse\s*:)",
    re.IGNORECASE,
)
_METADATA_RE = re.compile(
    r"(?:https?://|\bdoi\s*:\s*\S+|\borcid\.org/|\b[\w.+-]+@[\w.-]+\.\w+\b|"
    r"^\s*(?:received|accepted|published)\s*:)",
    re.IGNORECASE,
)
_JOINED_METADATA_PATTERNS = (
    re.compile(r"\d{5,}https?://orcid\.org/", re.IGNORECASE),
    re.compile(r"orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX][A-Za-z]", re.IGNORECASE),
    re.compile(
        r"(?:Research\s+Article|Review\s+Article|Systematic\s+Review)"
        r"(?:AcademicSubjects|[A-Z][a-z]{2,})"
    ),
)
_BAD_LATEX_PATTERNS = (
    re.compile(r"\\(?:fra|su|lef|rig|begi|en)\s+[A-Za-z]"),
    re.compile(r"(?<!\\)\\(?:frac|sum|left|right)\s*$"),
)
_CANONICAL_HEADINGS = {
    "abstract",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "appendix",
    "author contributions",
    "background",
    "conclusion",
    "conclusions",
    "conflict of interest",
    "conflicts of interest",
    "data availability",
    "discussion",
    "funding",
    "introduction",
    "keywords",
    "limitations",
    "materials and methods",
    "methods",
    "references",
    "results",
    "results and discussion",
    "supplementary information",
}
_REFERENCE_HEADINGS = {
    "bibliography",
    "literaturecited",
    "reference",
    "references",
}
_AFFILIATION_RE = re.compile(
    r"^\s*\d+[.)]?\s+.{0,100}\b(?:department|division|faculty|institute|institution|"
    r"laboratory|programme|program|school|unit|university|centre|center|hospital|clinic)\b",
    re.IGNORECASE,
)
_BIBLIOGRAPHY_ENTRY_RE = re.compile(
    r"(?:(?:18|19|20)\d{2}|\bet\s+al\b|\bdoi\s*:)",
    re.IGNORECASE,
)
_MATH_SIGNAL_RE = re.compile(
    r"(?:[=→←↔±×÷]|->|<-|\\[A-Za-z]|[{}]|\b(?:fig|eq)\.\s*\d)",
    re.IGNORECASE,
)
_MONTH_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
    re.IGNORECASE,
)
_BROKEN_YEAR_RE = re.compile(r"(?:1\s*[89]|2\s*0)\s*\d\s*\d")
_NUMBERED_NON_HEADING_BODY_RE = re.compile(
    r"^(?:comparison\s+of\s+(?:table|figure|fig\.|path\s+diagram)|"
    r"(?:table|figure|fig\.|scheme|chart|box|algorithm|equation)\s*[A-Za-z]?\d+)",
    re.IGNORECASE,
)
_LOCATION_SUFFIX_RE = re.compile(
    r",\s*(?:Argentina|Australia|Austria|Belgium|Brazil|Canada|China|Denmark|Finland|"
    r"France|Germany|Greece|India|Ireland|Israel|Italy|Japan|Netherlands|Norway|Poland|"
    r"Portugal|Russia|Singapore|Spain|Sweden|Switzerland|Taiwan|Turkey|UK|USA|"
    r"United\s+Kingdom|United\s+States|[A-Z]{2,3})\s*$",
    re.IGNORECASE,
)
_TITLE_CASE_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "vs",
    "with",
    "without",
}


class MiddleJsonError(ValueError):
    """Raised when a middle.json file does not satisfy the required schema."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_pdf_pages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    pages: Any = payload.get("pdf_info")
    if isinstance(pages, dict):
        pages = pages.get("pdf_info")
    if not isinstance(pages, list):
        raise MiddleJsonError("pdf_info must be a page list")
    for position, page in enumerate(pages):
        if not isinstance(page, dict):
            raise MiddleJsonError(f"pdf_info[{position}] must be an object")
        if not isinstance(page.get("page_idx"), int) or isinstance(page.get("page_idx"), bool):
            raise MiddleJsonError(f"pdf_info[{position}].page_idx must be an integer")
        if page["page_idx"] != position:
            raise MiddleJsonError(
                f"pdf_info[{position}].page_idx must equal {position}, got {page['page_idx']}"
            )
        if not isinstance(page.get("para_blocks"), list):
            raise MiddleJsonError(f"pdf_info[{position}].para_blocks must be a list")
        page_size = page.get("page_size")
        if (
            not isinstance(page_size, list)
            or len(page_size) != 2
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0 for value in page_size)
        ):
            raise MiddleJsonError(f"pdf_info[{position}].page_size must contain two positive numbers")
    return pages


def _span_text(span: dict[str, Any]) -> str:
    value = span.get("content")
    if value is None:
        value = span.get("text")
    return value if isinstance(value, str) else ""


def block_lines(block: dict[str, Any]) -> list[str]:
    result: list[str] = []
    lines = block.get("lines")
    if not isinstance(lines, list):
        return result
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("spans"), list):
            continue
        result.append(
            "".join(_span_text(span) for span in line["spans"] if isinstance(span, dict))
        )
    return result


def block_text(block: dict[str, Any]) -> str:
    """Return source-faithful text; line boundaries are retained separately in output."""
    return "".join(block_lines(block)).strip()


def display_text(block: dict[str, Any]) -> str:
    return " ".join(part.strip() for part in block_lines(block) if part.strip()).strip()


def normalize_heading(text: str, *, strip_number: bool = False) -> str:
    normalized = " ".join(text.split())
    if strip_number:
        match = _NUMBERED_HEADING_RE.match(normalized)
        if match:
            normalized = match.group("body")
    normalized = normalized.casefold()
    normalized = re.sub(r"[^\w]+", "", normalized, flags=re.UNICODE)
    return normalized


def parse_numbered_heading(text: str) -> dict[str, Any] | None:
    match = _NUMBERED_HEADING_RE.match(" ".join(text.split()))
    if not match:
        return None
    parts = [int(part) for part in match.group("number").split(".")]
    suspicious = len(parts) >= 5 or any(part > 99 for part in parts)
    return {
        "number": match.group("number"),
        "level": len(parts),
        "body": match.group("body").strip(),
        "trailing_dot": bool(match.group("trailing")),
        "suspicious": suspicious,
    }


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256(canonical_json_bytes(parts)).hexdigest()[:24]
    return f"{prefix}_{digest}"


def stable_block_id(
    doc_id: str,
    page_idx: int,
    block_path: str,
    block: dict[str, Any],
) -> str:
    return _stable_id("blk", doc_id, page_idx, block_path, block)


def _valid_level(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 6


def _is_running_header_candidate(
    block: dict[str, Any],
    page_idx: int,
    page_size: list[int | float],
    normalized_text: str,
    document_title_keys: set[str],
) -> bool:
    bbox = block.get("bbox")
    if (
        not normalized_text
        or (page_idx != 0 and normalized_text not in document_title_keys)
        or not isinstance(bbox, list)
        or len(bbox) != 4
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in bbox)
    ):
        return False
    x0, y0, _, _ = bbox
    page_width, page_height = page_size
    return y0 < page_height * 0.13 and x0 >= page_width * 0.45


def _looks_caption_like(text: str) -> bool:
    return bool(_CAPTION_RE.match(text))


def _looks_metadata_like(text: str) -> bool:
    return bool(_METADATA_RE.search(text))


def _looks_like_numbered_promotion(text: str, line_count: int) -> bool:
    numbered = parse_numbered_heading(text)
    if not numbered or numbered["suspicious"] or line_count > 3:
        return False
    body = numbered["body"]
    words = re.findall(r"[A-Za-z][\w'-]*", body)
    if not words or len(words) > 24 or len(body) > 180:
        return False
    if (
        _looks_caption_like(text)
        or _looks_caption_like(body)
        or _NUMBERED_NON_HEADING_BODY_RE.match(body)
        or _looks_metadata_like(text)
    ):
        return False
    if (
        _BIBLIOGRAPHY_ENTRY_RE.search(text)
        or _BROKEN_YEAR_RE.search(text)
        or _MATH_SIGNAL_RE.search(text)
        or _MONTH_DATE_RE.search(text)
    ):
        return False
    meaningful_words = [word for word in words if word.casefold() not in _TITLE_CASE_STOPWORDS]
    capitalized = sum(word[0].isupper() for word in meaningful_words)
    if meaningful_words and capitalized / len(meaningful_words) < 0.7:
        return False
    if numbered["level"] == 1:
        first = numbered["number"].split(".")[0]
        if int(first) > 30 or len(words) > 16:
            return False
        if not body[0].isupper():
            return False
    return True


def _looks_like_canonical_promotion(text: str) -> bool:
    candidate = re.sub(r"[\s:.-]+$", "", " ".join(text.split())).casefold()
    return candidate in _CANONICAL_HEADINGS


def _compact_context(
    blocks: list[dict[str, Any]],
    block_position: int,
    radius: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    def summarize(position: int) -> dict[str, Any]:
        block = blocks[position]
        text = display_text(block)
        return {
            "block_path": f"para/{position}",
            "type": block.get("type"),
            "text": text[:500],
            "text_level": block.get("text_level"),
        }

    before_start = max(0, block_position - radius)
    after_end = min(len(blocks), block_position + radius + 1)
    return {
        "before": [summarize(i) for i in range(before_start, block_position)],
        "after": [summarize(i) for i in range(block_position + 1, after_end)],
    }


def _candidate_base(
    *,
    doc_id: str,
    source_middle_json: str,
    source_sha256: str,
    page_idx: int,
    page_size: list[int | float],
    block_position: int,
    block: dict[str, Any],
    operation: str,
) -> dict[str, Any]:
    path = f"para/{block_position}"
    block_id = stable_block_id(doc_id, page_idx, path, block)
    source_lines = block_lines(block)
    raw_text = block_text(block)
    shown_text = display_text(block)
    numbered = parse_numbered_heading(shown_text)
    nonempty_lines = [line for line in source_lines if line]
    text_quality_flags: list[str] = []
    if len(nonempty_lines) > 1:
        text_quality_flags.append("MULTILINE_SOURCE_TEXT")
        if any(line.rstrip().endswith(("-", "‐", "‑", "–")) for line in nonempty_lines[:-1]):
            text_quality_flags.append("HYPHENATED_LINE_BREAK")
    return {
        "schema_version": SCHEMA_VERSION,
        "ruleset_version": RULESET_VERSION,
        "candidate_id": _stable_id("cand", block_id, operation),
        "doc_id": doc_id,
        "source_middle_json": source_middle_json,
        "source_sha256": source_sha256,
        "page_idx": page_idx,
        "pdf_page": page_idx + 1,
        "page_size": page_size,
        "block_id": block_id,
        "block_path": path,
        "source_block_index": block.get("index"),
        "bbox": block.get("bbox"),
        "text": raw_text,
        "line_texts": source_lines,
        "display_text": shown_text,
        "text_quality_flags": text_quality_flags,
        "text_review_status": "VISUAL_REVIEW_REQUIRED" if text_quality_flags else "SOURCE_SINGLE_LINE",
        "input_type": block.get("type"),
        "input_level": block.get("text_level"),
        "numbered_level": numbered["level"] if numbered else None,
        "operation": operation,
        "evidence": [],
    }


def _iter_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _content_flags(
    pages: list[dict[str, Any]],
    document_dir: Path,
) -> tuple[list[str], dict[str, int], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    def add(code: str, page_idx: int, detail: str) -> None:
        counts[code] += 1
        if sum(1 for item in examples if item["code"] == code) < 3:
            examples.append({"code": code, "page_idx": page_idx, "pdf_page": page_idx + 1, "detail": detail[:500]})

    seen_indices: set[int] = set()
    for position, page in enumerate(pages):
        page_idx = page["page_idx"]
        if page_idx in seen_indices or page_idx != position:
            add("PAGE_INDEX_ERROR", page_idx, f"page position={position}, page_idx={page_idx}")
        seen_indices.add(page_idx)

        blocks = page["para_blocks"]
        if not blocks:
            add("EMPTY_PAGE", page_idx, "para_blocks is empty")

        preproc = page.get("preproc_blocks")
        if isinstance(preproc, list) and preproc == blocks:
            add("PREPROC_DUPLICATES_PARA", page_idx, "preproc_blocks equals para_blocks; preproc ignored")

        for block_position, block in enumerate(blocks):
            if not isinstance(block, dict):
                add("INVALID_BLOCK", page_idx, f"para/{block_position} is not an object")
                continue
            all_nodes = list(_iter_dicts(block))
            texts = [
                value
                for node in all_nodes
                for key in ("content", "text", "html")
                if isinstance((value := node.get(key)), str) and value
            ]
            combined = "\n".join(texts)
            if any(pattern.search(combined) for pattern in _BAD_LATEX_PATTERNS):
                add("BAD_LATEX", page_idx, display_text(block) or combined)
            if any(pattern.search(combined) for pattern in _JOINED_METADATA_PATTERNS):
                add("JOINED_METADATA", page_idx, display_text(block) or combined)

            for node in all_nodes:
                image_path = node.get("image_path")
                if not isinstance(image_path, str) or not image_path:
                    continue
                candidate = Path(image_path)
                if not candidate.is_absolute():
                    candidate = document_dir / candidate
                if not candidate.is_file():
                    add("MISSING_IMAGE_ASSET", page_idx, image_path)

            if block.get("type") == "table":
                body_nodes = [node for node in all_nodes if node.get("type") == "table_body"]
                if body_nodes:
                    has_image = any(node.get("image_path") for body in body_nodes for node in _iter_dicts(body))
                    has_structure = any(
                        isinstance(node.get("html"), str) and _HTML_TABLE_RE.search(node["html"])
                        for body in body_nodes
                        for node in _iter_dicts(body)
                    )
                    if has_image and not has_structure:
                        add("IMAGE_ONLY_TABLE", page_idx, f"para/{block_position}")

    if not any(
        isinstance(block, dict) and block.get("type") == "title"
        for page in pages
        for block in page["para_blocks"]
    ):
        add("NO_TITLES", 0, "document contains no title blocks")
    return sorted(counts), dict(sorted(counts.items())), examples


def analyze_document(
    *,
    doc_id: str,
    payload: dict[str, Any],
    source_middle_json: str,
    source_sha256: str,
    document_dir: Path,
    include_promotions: bool = True,
) -> dict[str, Any]:
    pages = get_pdf_pages(payload)
    title_refs: list[tuple[int, int, dict[str, Any]]] = []
    for page in pages:
        for block_position, block in enumerate(page["para_blocks"]):
            if isinstance(block, dict) and block.get("type") == "title":
                title_refs.append((page["page_idx"], block_position, block))

    duplicate_counts = Counter(
        normalized
        for _, _, block in title_refs
        if (normalized := normalize_heading(display_text(block), strip_number=True))
    )
    document_title_keys = {
        normalize_heading(display_text(block))
        for _, _, block in title_refs
        if normalize_heading(display_text(block))
    }

    candidates: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    text_review_records: list[dict[str, Any]] = []
    previous_valid_level: int | None = None
    previous_title_text: str | None = None
    in_reference_section = False
    in_long_description = False
    in_review_material = False

    for page in pages:
        blocks = page["para_blocks"]
        if any(
            _REVIEW_MATERIAL_RE.search(display_text(block))
            for block in blocks
            if isinstance(block, dict)
        ):
            in_review_material = True
        for block_position, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "title":
                candidate = _candidate_base(
                    doc_id=doc_id,
                    source_middle_json=source_middle_json,
                    source_sha256=source_sha256,
                    page_idx=page["page_idx"],
                    page_size=page["page_size"],
                    block_position=block_position,
                    block=block,
                    operation="retain",
                )
                shown_text = candidate["display_text"]
                input_level = candidate["input_level"]
                numbered = parse_numbered_heading(shown_text)
                reasons: list[str] = []

                if not shown_text:
                    reasons.append("EMPTY_TITLE")
                if not _valid_level(input_level):
                    reasons.append("INVALID_LEVEL")
                if numbered:
                    candidate["evidence"].append(
                        {
                            "type": "explicit_numbering",
                            "number": numbered["number"],
                            "inferred_level": numbered["level"],
                        }
                    )
                    if numbered["suspicious"]:
                        reasons.append("SUSPICIOUS_NUMBERING")
                    if _valid_level(input_level) and numbered["level"] != input_level:
                        reasons.append("NUMBER_LEVEL_CONFLICT")
                if _valid_level(input_level) and previous_valid_level is not None:
                    if input_level > previous_valid_level + 1:
                        reasons.append("LEVEL_SKIP")
                        candidate["evidence"].append(
                            {
                                "type": "previous_title_level",
                                "level": previous_valid_level,
                                "text": previous_title_text,
                            }
                        )
                duplicate_key = normalize_heading(shown_text, strip_number=True)
                if duplicate_key and duplicate_counts[duplicate_key] > 1:
                    reasons.append("DUPLICATE_TITLE")
                    candidate["evidence"].append(
                        {"type": "normalized_duplicate_count", "count": duplicate_counts[duplicate_key]}
                    )
                if _looks_caption_like(shown_text):
                    reasons.append("CAPTION_LIKE_TITLE")
                if _LONG_DESCRIPTION_RE.search(shown_text):
                    reasons.append("LONG_DESCRIPTION_TITLE")
                if _looks_metadata_like(shown_text):
                    reasons.append("METADATA_LIKE_TITLE")
                word_count = len(shown_text.split())
                if len(shown_text) > 320 or word_count > 50:
                    reasons.append("TITLE_TOO_LONG")

                candidate["proposed_level"] = input_level if _valid_level(input_level) else None
                candidate["reason_codes"] = sorted(set(reasons))
                if reasons:
                    candidate["label_grade"] = "review_candidate"
                    candidate["review_status"] = "REVIEW_REQUIRED"
                    review = dict(candidate)
                    review["context"] = _compact_context(blocks, block_position)
                    review_records.append(review)
                else:
                    candidate["label_grade"] = "silver_rule"
                    candidate["review_status"] = "AUTO_SILVER"
                candidates.append(candidate)
                if candidate["text_quality_flags"]:
                    text_review = dict(candidate)
                    text_review["context"] = _compact_context(blocks, block_position)
                    text_review_records.append(text_review)

                if _valid_level(input_level):
                    previous_valid_level = input_level
                    previous_title_text = shown_text
                in_reference_section = (
                    normalize_heading(shown_text, strip_number=True) in _REFERENCE_HEADINGS
                )
                in_long_description = bool(_LONG_DESCRIPTION_RE.search(shown_text))
                continue

            if not include_promotions or block_type != "text":
                continue
            shown_text = display_text(block)
            numbered = parse_numbered_heading(shown_text)
            canonical = _looks_like_canonical_promotion(shown_text)
            next_block = blocks[block_position + 1] if block_position + 1 < len(blocks) else None
            previous_block = blocks[block_position - 1] if block_position > 0 else None
            next_is_list = isinstance(next_block, dict) and next_block.get("type") == "list"
            adjacent_numbered_same_level = bool(numbered) and any(
                isinstance(neighbor, dict)
                and neighbor.get("type") == "text"
                and (neighbor_numbered := parse_numbered_heading(display_text(neighbor))) is not None
                and neighbor_numbered["level"] == numbered["level"]
                for neighbor in (previous_block, next_block)
            )
            candidate_key = normalize_heading(shown_text)
            adjacent_duplicate_title = any(
                isinstance(neighbor, dict)
                and neighbor.get("type") == "title"
                and normalize_heading(display_text(neighbor)) == candidate_key
                for neighbor in (
                    previous_block,
                    next_block,
                )
            )
            running_header = _is_running_header_candidate(
                block,
                page["page_idx"],
                page["page_size"],
                candidate_key,
                document_title_keys,
            )
            next_text = display_text(next_block) if isinstance(next_block, dict) else ""
            references_with_entries = (
                candidate_key == "references"
                and bool(
                    _BIBLIOGRAPHY_ENTRY_RE.search(next_text)
                    or _BROKEN_YEAR_RE.search(next_text)
                )
            )
            if canonical and (
                adjacent_duplicate_title
                or (in_review_material and not references_with_entries)
                or running_header
            ):
                canonical = False
            address_like = (
                shown_text.count(",") >= 2
                and bool(_LOCATION_SUFFIX_RE.search(shown_text))
            )
            numbered_candidate = (
                page["page_idx"] > 0
                and not in_reference_section
                and not in_long_description
                and not next_is_list
                and not adjacent_numbered_same_level
                and not address_like
                and not _AFFILIATION_RE.match(shown_text)
                and _looks_like_numbered_promotion(shown_text, len(block_lines(block)))
            )
            if not canonical and not numbered_candidate:
                continue

            candidate = _candidate_base(
                doc_id=doc_id,
                source_middle_json=source_middle_json,
                source_sha256=source_sha256,
                page_idx=page["page_idx"],
                page_size=page["page_size"],
                block_position=block_position,
                block=block,
                operation="promote",
            )
            reasons = ["TEXT_BLOCK_PROMOTION_REQUIRES_REVIEW"]
            if canonical:
                reasons.append("CANONICAL_HEADING_TEXT")
                candidate["evidence"].append({"type": "canonical_heading_lexicon"})
            if numbered_candidate and numbered:
                reasons.append("NUMBERED_TEXT_BLOCK")
                candidate["evidence"].append(
                    {
                        "type": "explicit_numbering",
                        "number": numbered["number"],
                        "inferred_level": numbered["level"],
                    }
                )
            candidate["proposed_level"] = numbered["level"] if numbered_candidate and numbered else 1
            candidate["reason_codes"] = sorted(reasons)
            candidate["label_grade"] = "review_candidate"
            candidate["review_status"] = "REVIEW_REQUIRED"
            candidates.append(candidate)
            review = dict(candidate)
            review["context"] = _compact_context(blocks, block_position)
            review_records.append(review)
            if candidate["text_quality_flags"]:
                text_review = dict(candidate)
                text_review["context"] = _compact_context(blocks, block_position)
                text_review_records.append(text_review)
            if normalize_heading(shown_text, strip_number=True) in _REFERENCE_HEADINGS:
                in_reference_section = True

    content_flags, content_flag_counts, content_examples = _content_flags(pages, document_dir)
    return {
        "pages": len(pages),
        "source_title_count": len(title_refs),
        "candidates": candidates,
        "review_records": review_records,
        "text_review_records": text_review_records,
        "content_flags": content_flags,
        "content_flag_counts": content_flag_counts,
        "content_examples": content_examples,
    }
