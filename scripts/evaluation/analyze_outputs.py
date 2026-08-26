#!/usr/bin/env python3
"""Analyze saved Unlimited-OCR responses against Markdown ground truth."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import statistics
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "uocr-evaluation-analysis-v1"
_HEADING_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_GROUNDING_RE = re.compile(r"<\|/?det\|>")
_PAGE_RE = re.compile(r"<PAGE>")
_HTML_TABLE_RE = re.compile(r"<table(?:\s|>)", re.IGNORECASE)
_FORMULA_RE = re.compile(r"(?<!\\)\$|\\(?:begin|end)\{|\\\[|\\\]")


class AnalysisError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path, nargs="+")
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--cap-ratio", type=float, default=0.95)
    parser.add_argument("--severe-long-ratio", type=float, default=4.0)
    parser.add_argument("--severe-short-ratio", type=float, default=0.10)
    return parser.parse_args()


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AnalysisError(f"{path}:{line_number}: {exc}") from exc
                key = (str(row.get("file") or ""), str(row.get("model") or ""))
                if not all(key):
                    raise AnalysisError(f"{path}:{line_number}: missing file/model")
                previous = by_key.get(key)
                if previous is not None and previous != row:
                    raise AnalysisError(f"conflicting duplicate response: {key}")
                by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def _all_ground_truth(gt_root: Path) -> list[tuple[Path, str]]:
    records = []
    for path in sorted(gt_root.rglob("*.md")):
        records.append((path, path.read_text(encoding="utf-8")))
    if not records:
        raise AnalysisError(f"no Markdown ground truth found under {gt_root}")
    return records


def match_ground_truth(
    row: Mapping[str, Any],
    ground_truth: list[tuple[Path, str]],
) -> tuple[Path, str]:
    stem = Path(str(row["file"])).stem
    exact = [(path, text) for path, text in ground_truth if path.stem == stem]
    if len(exact) == 1:
        return exact[0]
    expected_chars = row.get("gt_chars")
    by_length = [
        (path, text)
        for path, text in ground_truth
        if isinstance(expected_chars, int) and len(text) == expected_chars
    ]
    if len(by_length) == 1:
        return by_length[0]
    raise AnalysisError(
        f"cannot uniquely match GT for {row['file']!r}: exact={len(exact)} "
        f"length={len(by_length)} expected_chars={expected_chars!r}"
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def extract_headings(text: str) -> list[tuple[int, str]]:
    headings = []
    fence: tuple[str, int] | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[0]
            width = len(stripped) - len(stripped.lstrip(marker))
            if fence is None:
                fence = (marker, width)
            elif marker == fence[0] and width >= fence[1]:
                fence = None
            continue
        if fence is not None:
            continue
        match = _HEADING_RE.match(line)
        if match:
            headings.append((len(match.group(1)), normalize_text(match.group(2)).casefold()))
    return headings


def _f1(predicted: Iterable[Any], expected: Iterable[Any]) -> dict[str, float | int]:
    predicted_counter = Counter(predicted)
    expected_counter = Counter(expected)
    true_positive = sum((predicted_counter & expected_counter).values())
    predicted_count = sum(predicted_counter.values())
    expected_count = sum(expected_counter.values())
    precision = true_positive / predicted_count if predicted_count else float(expected_count == 0)
    recall = true_positive / expected_count if expected_count else float(predicted_count == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted": predicted_count,
        "expected": expected_count,
        "matched": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def line_repetition(text: str) -> dict[str, float | int]:
    lines = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
    counts = Counter(lines)
    repeated_instances = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "nonempty_lines": len(lines),
        "unique_lines": len(counts),
        "duplicate_line_ratio": repeated_instances / len(lines) if lines else 0.0,
        "max_line_occurrences": max(counts.values(), default=0),
    }


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return len(zlib.compress(raw, level=9)) / len(raw) if raw else 0.0


def sequence_similarity(output: str, target: str) -> float:
    left = normalize_text(output)
    right = normalize_text(target)
    return difflib.SequenceMatcher(None, left, right, autojunk=True).ratio()


def analyze_row(
    row: Mapping[str, Any],
    gt_path: Path,
    target: str,
    *,
    max_output_tokens: int,
    cap_ratio: float,
    severe_long_ratio: float,
    severe_short_ratio: float,
) -> dict[str, Any]:
    response = row.get("response")
    response = response if isinstance(response, dict) else {}
    output = response.get("text")
    output = output if isinstance(output, str) else ""
    output_tokens = response.get("output_tokens")
    length_ratio = len(output) / len(target) if target else math.inf
    repetition = line_repetition(output)
    output_headings = extract_headings(output)
    target_headings = extract_headings(target)
    heading_text = _f1((text for _level, text in output_headings), (text for _level, text in target_headings))
    heading_level = _f1(output_headings, target_headings)
    grounding_tags = len(_GROUNDING_RE.findall(output))
    page_tags = len(_PAGE_RE.findall(output))
    cap_hit = isinstance(output_tokens, int) and output_tokens >= max_output_tokens * cap_ratio
    severe_reasons = []
    if row.get("error"):
        severe_reasons.append("request_error")
    if not output.strip():
        severe_reasons.append("empty_output")
    if cap_hit:
        severe_reasons.append("generation_cap")
    if length_ratio >= severe_long_ratio:
        severe_reasons.append("over_generation")
    if length_ratio <= severe_short_ratio:
        severe_reasons.append("under_generation")
    if repetition["duplicate_line_ratio"] >= 0.50:
        severe_reasons.append("line_repetition")

    return {
        "file": row["file"],
        "model": row["model"],
        "pages": row.get("pages"),
        "prompt": row.get("prompt"),
        "ground_truth": str(gt_path),
        "output_chars": len(output),
        "target_chars": len(target),
        "length_ratio": length_ratio,
        "output_tokens": output_tokens,
        "cap_hit": cap_hit,
        "sequence_similarity": sequence_similarity(output, target),
        "compression_ratio": compression_ratio(output),
        "grounding_tags": grounding_tags,
        "page_tags": page_tags,
        "markdown_headings": len(output_headings),
        "html_tables": len(_HTML_TABLE_RE.findall(output)),
        "formula_markers": len(_FORMULA_RE.findall(output)),
        "protocol": (
            "mixed" if grounding_tags and output_headings else
            "grounding" if grounding_tags else
            "markdown" if output_headings else
            "unclassified"
        ),
        "heading_text": heading_text,
        "heading_text_and_level": heading_level,
        "repetition": repetition,
        "severe": bool(severe_reasons),
        "severe_reasons": severe_reasons,
    }


def _mean(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    result = {}
    for model, model_rows in sorted(by_model.items()):
        result[model] = {
            "records": len(model_rows),
            "severe": sum(row["severe"] for row in model_rows),
            "cap_hits": sum(row["cap_hit"] for row in model_rows),
            "protocols": dict(sorted(Counter(row["protocol"] for row in model_rows).items())),
            "mean_length_ratio": _mean(model_rows, "length_ratio"),
            "mean_sequence_similarity": _mean(model_rows, "sequence_similarity"),
            "mean_heading_text_f1": statistics.fmean(
                row["heading_text"]["f1"] for row in model_rows
            ),
            "mean_heading_level_f1": statistics.fmean(
                row["heading_text_and_level"]["f1"] for row in model_rows
            ),
            "mean_duplicate_line_ratio": statistics.fmean(
                row["repetition"]["duplicate_line_ratio"] for row in model_rows
            ),
        }
    return result


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Unlimited-OCR offline evaluation analysis",
        "",
        "This report analyzes saved responses only. It does not rerun inference.",
        "",
        "## Aggregate",
        "",
        "| Model | N | Severe | Cap hit | Mean length/GT | Text similarity | Heading F1 | Heading+level F1 | Duplicate lines | Protocols |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model, stats in report["aggregate"].items():
        lines.append(
            f"| {model} | {stats['records']} | {stats['severe']} | {stats['cap_hits']} | "
            f"{stats['mean_length_ratio']:.2f} | {stats['mean_sequence_similarity']:.3f} | "
            f"{stats['mean_heading_text_f1']:.3f} | {stats['mean_heading_level_f1']:.3f} | "
            f"{_percent(stats['mean_duplicate_line_ratio'])} | "
            f"{json.dumps(stats['protocols'], ensure_ascii=False, sort_keys=True)} |"
        )
    lines.extend(
        [
            "",
            "## Records",
            "",
            "| File | Model | Pages | Output/GT | Similarity | Heading F1 | Duplicate lines | Protocol | Cap | Severe reasons |",
            "|---|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in report["records"]:
        reasons = ", ".join(row["severe_reasons"]) or "-"
        lines.append(
            f"| {row['file']} | {row['model']} | {row.get('pages') or '-'} | "
            f"{row['length_ratio']:.2f} | {row['sequence_similarity']:.3f} | "
            f"{row['heading_text']['f1']:.3f} | "
            f"{_percent(row['repetition']['duplicate_line_ratio'])} | {row['protocol']} | "
            f"{'yes' if row['cap_hit'] else 'no'} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## Metric boundary",
            "",
            "- `sequence_similarity` is Python `SequenceMatcher` on NFKC/whitespace-normalized text; it is a diagnostic signal, not OmniDocBench normalized edit distance.",
            "- Heading scores use fenced-code-aware ATX headings and multiset matching.",
            "- `cap_hit` requires a recorded token count; legacy single-page responses did not record one.",
            "- A severe record is empty/errored, reaches the configured cap, is at least 4x GT, is at most 0.1x GT, or has at least 50% duplicate non-empty lines.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.max_output_tokens <= 0 or not 0 < args.cap_ratio <= 1:
        raise AnalysisError("invalid output-token cap configuration")
    rows = load_rows(args.responses)
    ground_truth = _all_ground_truth(args.gt_root)
    analyzed = []
    for row in rows:
        gt_path, target = match_ground_truth(row, ground_truth)
        analyzed.append(
            analyze_row(
                row,
                gt_path,
                target,
                max_output_tokens=args.max_output_tokens,
                cap_ratio=args.cap_ratio,
                severe_long_ratio=args.severe_long_ratio,
                severe_short_ratio=args.severe_short_ratio,
            )
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "configuration": {
            "max_output_tokens": args.max_output_tokens,
            "cap_ratio": args.cap_ratio,
            "severe_long_ratio": args.severe_long_ratio,
            "severe_short_ratio": args.severe_short_ratio,
        },
        "aggregate": aggregate(analyzed),
        "records": analyzed,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AnalysisError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}")
