"""Deterministic PMC silver-title decisions and MinerU-style Markdown rendering."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from html import unescape
from typing import Any, Iterable, Mapping

from .core import TitleMaskError, canonicalize_target


PMC_SILVER_RULESET_VERSION = "pmc-title-silver-freeze-2026-08-23-v2"
_LINE_END_HYPHENS = "-\u00ad\u2010\u2011\u2043"
_MARKDOWN_SPECIAL = frozenset("*_`~$")
_CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def _escape_markdown_text(text: str) -> str:
    output = []
    backslashes = 0
    for char in text:
        if char == "\\":
            output.append(char)
            backslashes += 1
            continue
        if char in _MARKDOWN_SPECIAL and backslashes % 2 == 0:
            output.append("\\")
        output.append(char)
        backslashes = 0
    return "".join(output)


def _escape_text_prefix(text: str) -> str:
    return re.sub(r"^([ \t]{0,3})(#{1,6}|[+-])(?=[ \t])", r"\1\\\2", text)


def _is_cjk(text: str) -> bool:
    letters = sum(char.isalpha() for char in text)
    return letters > 0 and len(_CJK_RE.findall(text)) / letters >= 0.5


def normalize_title_lines(lines: object, fallback: str) -> str:
    if not isinstance(lines, list):
        lines = []
    parts = [" ".join(str(line).split()) for line in lines if str(line).strip()]
    if not parts:
        return " ".join(fallback.split())
    result = parts[0]
    for part in parts[1:]:
        if result and result[-1] in _LINE_END_HYPHENS and part[:1].islower():
            result = result[:-1] + part
        elif _is_cjk(result + part):
            result += part
        else:
            result += " " + part
    return result.strip()


def _candidate_level(candidate: Mapping[str, Any]) -> int:
    level = candidate.get("input_level")
    if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 6:
        level = 1
    reasons = set(candidate.get("reason_codes") or [])
    numbered_level = candidate.get("numbered_level")
    if (
        "NUMBER_LEVEL_CONFLICT" in reasons
        and "SUSPICIOUS_NUMBERING" not in reasons
        and isinstance(numbered_level, int)
        and not isinstance(numbered_level, bool)
        and level < numbered_level <= 6
    ):
        level = numbered_level
    if "LEVEL_SKIP" in reasons:
        previous_levels = [
            evidence.get("level")
            for evidence in candidate.get("evidence") or []
            if isinstance(evidence, dict) and evidence.get("type") == "previous_title_level"
        ]
        if previous_levels and isinstance(previous_levels[-1], int):
            level = min(level, previous_levels[-1] + 1)
    return level


def freeze_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    input_type = candidate.get("input_type")
    reasons = sorted(set(candidate.get("reason_codes") or []))
    text = normalize_title_lines(candidate.get("line_texts"), str(candidate.get("display_text") or ""))
    if not text:
        raise TitleMaskError(f"{candidate.get('candidate_id')}: empty silver title text")

    if input_type == "text":
        action = "promote"
        level = candidate.get("proposed_level")
        if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 6:
            raise TitleMaskError(f"{candidate.get('candidate_id')}: invalid promotion level")
    elif input_type == "title":
        demote = bool({"CAPTION_LIKE_TITLE", "TITLE_TOO_LONG", "METADATA_LIKE_TITLE"} & set(reasons))
        action = "demote" if demote else "retain"
        level = None if demote else _candidate_level(candidate)
    else:
        raise TitleMaskError(f"{candidate.get('candidate_id')}: unsupported input_type {input_type!r}")

    source_text = str(candidate.get("text") or "")
    operation = action
    if action == "retain" and level != candidate.get("input_level"):
        operation = "change_level"
    elif action == "retain" and text != source_text:
        operation = "normalize_title_text"
    return {
        "candidate_id": candidate.get("candidate_id"),
        "block_id": candidate.get("block_id"),
        "page_idx": candidate.get("page_idx"),
        "block_path": candidate.get("block_path"),
        "input_type": input_type,
        "action": action,
        "operation": operation,
        "level": level,
        "text": text,
        "source_text": source_text,
        "reason_codes": reasons,
    }


def build_decision_index(candidates: Iterable[Mapping[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    decisions = {}
    for candidate in candidates:
        page_idx = candidate.get("page_idx")
        block_path = candidate.get("block_path")
        if not isinstance(page_idx, int) or not isinstance(block_path, str):
            raise TitleMaskError(f"{candidate.get('candidate_id')}: invalid block location")
        key = (page_idx, block_path)
        if key in decisions:
            raise TitleMaskError(f"duplicate PMC candidate location: {key}")
        decisions[key] = freeze_candidate(candidate)
    return decisions


def _span_content(span: Mapping[str, Any]) -> str:
    value = span.get("content", span.get("text", ""))
    return value if isinstance(value, str) else ""


def _following_span(block: Mapping[str, Any], line_index: int, span_index: int):
    lines = block.get("lines") or []
    for next_line_index in range(line_index, len(lines)):
        spans = lines[next_line_index].get("spans") or []
        start = span_index + 1 if next_line_index == line_index else 0
        for span in spans[start:]:
            if isinstance(span, dict) and _span_content(span).strip():
                return span
    return None


def _merge_text(block: Mapping[str, Any], *, escape_prefix: bool = True) -> str:
    lines = block.get("lines") or []
    raw_text = "".join(
        _span_content(span)
        for line in lines
        if isinstance(line, dict)
        for span in line.get("spans") or []
        if isinstance(span, dict) and span.get("type") == "text"
    )
    cjk = _is_cjk(raw_text)
    output = ""
    for line_index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        spans = line.get("spans") or []
        for span_index, span in enumerate(spans):
            if not isinstance(span, dict):
                continue
            span_type = span.get("type")
            value = _span_content(span).strip()
            if not value:
                continue
            if span_type == "text":
                value = _escape_markdown_text(value)
            elif span_type == "inline_equation":
                value = f"${value}$"
            elif span_type == "interline_equation":
                value = f"\n$$\n{value}\n$$\n"
            else:
                continue

            following = _following_span(block, line_index, span_index)
            if span_type == "interline_equation":
                output += value
            elif (
                span_type == "text"
                and value[-1:] in _LINE_END_HYPHENS
                and following is not None
                and _span_content(following)[:1].islower()
            ):
                output += value[:-1]
            elif following is not None and not cjk:
                output += value + " "
            elif following is not None and span_type == "inline_equation":
                output += value + " "
            else:
                output += value
    output = output.strip()
    return _escape_text_prefix(output) if escape_prefix else output


def _iter_nested(block: Mapping[str, Any]):
    children = [child for child in (block.get("blocks") or []) if isinstance(child, dict)]
    indexed = sorted(
        enumerate(children),
        key=lambda item: (item[1].get("index", float("inf")), item[0]),
    )
    for _position, child in indexed:
        yield child


def _render_visual(block: Mapping[str, Any]) -> str:
    segments = []
    for child in _iter_nested(block):
        child_type = child.get("type")
        if child_type in {
            "image_caption",
            "image_footnote",
            "table_caption",
            "table_footnote",
            "chart_caption",
            "chart_footnote",
        }:
            text = _merge_text(child)
            if text:
                segments.append(text)
            continue
        for line in child.get("lines") or []:
            for span in line.get("spans") or []:
                if not isinstance(span, dict):
                    continue
                span_type = span.get("type")
                if span_type in {"image", "chart"}:
                    image_path = span.get("image_path")
                    if isinstance(image_path, str) and image_path:
                        segments.append(f"![]({image_path})")
                    content = _span_content(span).strip()
                    if content:
                        segments.append(f"<details>\n<summary>{span_type} content</summary>\n\n{content}\n</details>")
                elif span_type == "table":
                    html = span.get("html")
                    if isinstance(html, str) and html.strip():
                        segments.append(re.sub(r"<eq>(.*?)</eq>", lambda m: f" ${unescape(m.group(1))}$ ", html, flags=re.S))
                    else:
                        image_path = span.get("image_path")
                        if isinstance(image_path, str) and image_path:
                            segments.append(f"![]({image_path})")
    return "\n\n".join(segment for segment in segments if segment.strip())


def _render_plain_block(block: Mapping[str, Any]) -> str:
    block_type = block.get("type")
    if block_type in {"text", "interline_equation", "ref_text", "phonetic"}:
        return _merge_text(block)
    if block_type == "list":
        items = [_merge_text(child, escape_prefix=False) for child in _iter_nested(block)]
        return "  \n".join(item for item in items if item)
    if block_type in {"image", "table", "chart"}:
        return _render_visual(block)
    return ""


def render_pmc_document(
    payload: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    pages = payload.get("pdf_info")
    if isinstance(pages, dict):
        pages = pages.get("pdf_info")
    if not isinstance(pages, list):
        raise TitleMaskError("PMC middle.json has no pdf_info page list")
    decisions = build_decision_index(candidates)
    used = set()
    rendered_pages = []
    for page_position, page in enumerate(pages):
        if not isinstance(page, dict) or page.get("page_idx") != page_position:
            raise TitleMaskError(f"invalid PMC page at position {page_position}")
        parts = []
        for block_position, block in enumerate(page.get("para_blocks") or []):
            if not isinstance(block, dict):
                raise TitleMaskError(f"page {page_position} para/{block_position} is not an object")
            key = (page_position, f"para/{block_position}")
            decision = decisions.get(key)
            block_type = block.get("type")
            if decision is not None:
                used.add(key)
                if decision["input_type"] != block_type:
                    raise TitleMaskError(f"candidate type mismatch at page {page_position} para/{block_position}")
                if decision["action"] in {"retain", "promote"}:
                    text = _escape_markdown_text(decision["text"])
                    rendered = f"{'#' * decision['level']} {text}"
                else:
                    rendered = _escape_text_prefix(_escape_markdown_text(decision["text"]))
            elif block_type == "title":
                raise TitleMaskError(f"title block missing audit candidate at page {page_position} para/{block_position}")
            else:
                rendered = _render_plain_block(copy.deepcopy(block))
            if rendered.strip():
                parts.append(rendered.strip())
        page_target = "\n\n".join(parts)
        rendered_pages.append(canonicalize_target(page_target) if page_target.strip() else "")
    unused = sorted(set(decisions) - used)
    if unused:
        raise TitleMaskError(f"PMC candidates do not map to source blocks: {unused[:5]}")
    return rendered_pages, [decisions[key] for key in sorted(decisions)]


def stable_split(seed: str, source: str, doc_id: str) -> str:
    value = int(hashlib.sha256(f"{seed}:{source}:{doc_id}".encode("utf-8")).hexdigest()[:16], 16) % 10_000
    if value < 9_000:
        return "train"
    if value < 9_500:
        return "validation"
    return "test"


def stable_page_choice(seed: str, doc_id: str, rendered_pages: list[str], heading_counter) -> int:
    eligible = []
    for page_idx, target in enumerate(rendered_pages):
        headings = heading_counter(target)
        if headings <= 0:
            continue
        length_penalty = 0 if 300 <= len(target) <= 12_000 else 1
        digest = hashlib.sha256(f"{seed}:{doc_id}:{page_idx}".encode("utf-8")).hexdigest()
        eligible.append((length_penalty, abs(len(target) - 4_000), digest, page_idx))
    if not eligible:
        raise TitleMaskError(f"{doc_id}: no page contains a trainable ATX heading")
    return min(eligible)[-1]
