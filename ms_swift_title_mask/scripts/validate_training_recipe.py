#!/usr/bin/env python3
"""Validate dataset composition gates for a named Unlimited-OCR experiment."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_BUNDLE_PARENT = Path(__file__).resolve().parents[2]
if str(_BUNDLE_PARENT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_PARENT))

from ms_swift_title_mask.core import TitleMaskError  # noqa: E402
from ms_swift_title_mask.data_contract import read_jsonl, validate_training_row  # noqa: E402


RECIPE_CHOICES = (
    "readoc_r0", "replay_r1", "pmc_s10", "trusted_title", "readoc_view_ablation", "legacy_20260823"
)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_MISSING_ASSET_RE = re.compile(r"!\[[^\]]*\]\((?!https?://|data:)[^\n)]+\)")
_XML_IMAGE_RE = re.compile(r"<(?:img|graphic)\b", re.IGNORECASE)
_BROKEN_LATEX_RE = re.compile(r"\\(?:lef\s+t|righ\s+t|fra\s+c|su\s+m|rangl\s+e)\b")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, nargs="+")
    parser.add_argument("--recipe", choices=RECIPE_CHOICES, required=True)
    parser.add_argument("--skip-image-existence", action="store_true")
    parser.add_argument(
        "--full-row-validation",
        action="store_true",
        help="also rerun the slower Markdown/title/SHA contract for every row",
    )
    parser.add_argument("--min-cjk-char-share", type=float, default=0.15)
    parser.add_argument("--max-pmc-char-share", type=float, default=0.12)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _source_family(source: str) -> str:
    if source.startswith("READoc"):
        return "readoc"
    if source.startswith("PMC"):
        return "pmc"
    return "replay"


def _light_summary(row: dict, path: Path, line_number: int, check_images: bool) -> dict:
    row_id = row.get("id")
    messages = row.get("messages")
    images = row.get("images")
    meta = row.get("meta")
    if not isinstance(row_id, str) or not row_id:
        raise TitleMaskError(f"{path}:{line_number}: missing row id")
    if not isinstance(messages, list) or len(messages) != 2:
        raise TitleMaskError(f"{path}:{line_number}: messages must have two turns")
    if not isinstance(messages[0], dict) or not isinstance(messages[1], dict):
        raise TitleMaskError(f"{path}:{line_number}: messages must contain objects")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise TitleMaskError(f"{path}:{line_number}: invalid message roles")
    if not isinstance(messages[1].get("content"), str) or not messages[1]["content"]:
        raise TitleMaskError(f"{path}:{line_number}: empty assistant target")
    if not isinstance(images, list) or not images or not all(isinstance(item, str) and item for item in images):
        raise TitleMaskError(f"{path}:{line_number}: images must be non-empty paths")
    if check_images:
        missing = next((item for item in images if not Path(item).is_file()), None)
        if missing is not None:
            raise TitleMaskError(f"{path}:{line_number}: image does not exist: {missing}")
    if not isinstance(meta, dict):
        raise TitleMaskError(f"{path}:{line_number}: meta must be an object")
    return {"images": len(images)}


def audit(paths: list[Path], *, check_images: bool, full_row_validation: bool) -> dict:
    totals = Counter()
    by_source = Counter()
    by_family = Counter()
    by_form = Counter()
    by_prompt = Counter()
    content_statuses = Counter()
    title_statuses = Counter()
    pmc_content_statuses = Counter()
    pmc_title_statuses = Counter()
    seen_ids = set()
    document_splits = {}
    pmc_documents = Counter()
    artifacts = Counter()
    pmc_artifacts = Counter()
    for path in paths:
        inferred_split = path.stem if path.stem in {"train", "validation", "test"} else None
        for line_number, row in read_jsonl(path):
            try:
                if full_row_validation:
                    summary = validate_training_row(
                        row,
                        expected_split=inferred_split,
                        check_images=check_images,
                    )
                else:
                    summary = _light_summary(row, path, line_number, check_images)
            except TitleMaskError as exc:
                raise TitleMaskError(f"{path}:{line_number}: {exc}") from exc
            row_id = row["id"]
            if row_id in seen_ids:
                raise TitleMaskError(f"{path}:{line_number}: duplicate row id {row_id!r}")
            seen_ids.add(row_id)
            meta = row["meta"]
            source = meta.get("source")
            doc_id = meta.get("doc_id")
            if not isinstance(source, str) or not source or not isinstance(doc_id, str) or not doc_id:
                raise TitleMaskError(f"{path}:{line_number}: missing meta.source/doc_id")
            split = meta.get("split")
            if split not in {"train", "validation", "test"}:
                raise TitleMaskError(f"{path}:{line_number}: invalid split {split!r}")
            if inferred_split is not None and split != inferred_split:
                raise TitleMaskError(
                    f"{path}:{line_number}: split {split!r} does not match {inferred_split!r}"
                )
            document_key = (source, doc_id)
            previous_split = document_splits.setdefault(document_key, split)
            if previous_split != split:
                raise TitleMaskError(f"document leakage: {document_key} appears in two splits")

            target = row["messages"][1]["content"]
            prompt = row["messages"][0]["content"]
            family = _source_family(source)
            chars = len(target)
            cjk_chars = len(_CJK_RE.findall(target))
            totals.update(rows=1, target_chars=chars, cjk_chars=cjk_chars, images=summary["images"])
            by_source[source] += chars
            by_family[family] += chars
            by_form[str(meta.get("sample_form") or "<missing>")] += 1
            by_prompt[prompt] += 1
            content_statuses[str(meta.get("content_review_status") or "<missing>")] += 1
            title_statuses[str(meta.get("title_review_status") or "<missing>")] += 1
            if family == "pmc":
                pmc_documents[document_key] += 1
                pmc_content_statuses[str(meta.get("content_review_status") or "<missing>")] += 1
                pmc_title_statuses[str(meta.get("title_review_status") or "<missing>")] += 1
            artifact_text = _HTML_COMMENT_RE.sub("", target)
            row_artifacts = {
                "relative_markdown_image": len(_MISSING_ASSET_RE.findall(artifact_text)),
                "xml_image_tag": len(_XML_IMAGE_RE.findall(artifact_text)),
                "known_broken_latex": len(_BROKEN_LATEX_RE.findall(artifact_text)),
            }
            artifacts.update(row_artifacts)
            if family == "pmc":
                pmc_artifacts.update(row_artifacts)

    target_chars = totals["target_chars"]
    return {
        "rows": totals["rows"],
        "target_chars": target_chars,
        "images": totals["images"],
        "cjk_chars": totals["cjk_chars"],
        "cjk_char_share": totals["cjk_chars"] / target_chars if target_chars else 0.0,
        "source_target_chars": dict(sorted(by_source.items())),
        "family_target_chars": dict(sorted(by_family.items())),
        "family_target_char_share": {
            family: chars / target_chars if target_chars else 0.0
            for family, chars in sorted(by_family.items())
        },
        "sample_forms": dict(sorted(by_form.items())),
        "prompts": dict(sorted(by_prompt.items())),
        "content_review_statuses": dict(sorted(content_statuses.items())),
        "title_review_statuses": dict(sorted(title_statuses.items())),
        "pmc_content_review_statuses": dict(sorted(pmc_content_statuses.items())),
        "pmc_title_review_statuses": dict(sorted(pmc_title_statuses.items())),
        "repeated_pmc_documents": sum(count > 1 for count in pmc_documents.values()),
        "artifacts": dict(sorted(artifacts.items())),
        "pmc_artifacts": dict(sorted(pmc_artifacts.items())),
    }


def validate_recipe(report: dict, args: argparse.Namespace) -> None:
    family_share = report["family_target_char_share"]
    pmc_share = family_share.get("pmc", 0.0)
    replay_share = family_share.get("replay", 0.0)
    sample_forms = set(report["sample_forms"])
    artifacts = report["artifacts"]

    if args.recipe == "legacy_20260823":
        return
    if args.recipe == "readoc_r0":
        if set(family_share) != {"readoc"}:
            raise TitleMaskError(f"readoc_r0 must contain only READoc; got {family_share}")
        if sample_forms != {"full_document"}:
            raise TitleMaskError(f"readoc_r0 requires full_document rows; got {sorted(sample_forms)}")
    elif args.recipe == "replay_r1":
        if family_share.get("readoc", 0.0) <= 0 or replay_share <= 0:
            raise TitleMaskError("replay_r1 requires both READoc and replay sources")
        if report["cjk_char_share"] < args.min_cjk_char_share:
            raise TitleMaskError(
                f"replay_r1 CJK char share {report['cjk_char_share']:.4f} "
                f"is below {args.min_cjk_char_share:.4f}"
            )
        if pmc_share > 0:
            raise TitleMaskError("replay_r1 must isolate replay from PMC")
    elif args.recipe == "pmc_s10":
        if family_share.get("readoc", 0.0) <= 0 or pmc_share <= 0:
            raise TitleMaskError("pmc_s10 requires READoc and PMC")
        if pmc_share > args.max_pmc_char_share:
            raise TitleMaskError(
                f"PMC char share {pmc_share:.4f} exceeds {args.max_pmc_char_share:.4f}"
            )
        if report["repeated_pmc_documents"]:
            raise TitleMaskError("pmc_s10 repeats a PMC document; choose one full/window/single form")
        disallowed = {
            status: count
            for status, count in report["pmc_content_review_statuses"].items()
            if status not in {"CONTENT_ACCEPTED", "HUMAN_ACCEPTED"}
        }
        if disallowed:
            raise TitleMaskError(f"pmc_s10 has unaccepted content status: {disallowed}")
        if any(report["pmc_artifacts"].values()):
            raise TitleMaskError(
                f"pmc_s10 has serialized content artifacts: {report['pmc_artifacts']}"
            )
    elif args.recipe == "trusted_title":
        if any(artifacts.values()):
            raise TitleMaskError(f"trusted_title has serialized content artifacts: {artifacts}")
        disallowed_titles = {
            status: count
            for status, count in report["title_review_statuses"].items()
            if status != "HUMAN_ACCEPTED"
        }
        if disallowed_titles:
            raise TitleMaskError(
                f"trusted_title requires HUMAN_ACCEPTED headings: {disallowed_titles}"
            )
        if pmc_share > args.max_pmc_char_share:
            raise TitleMaskError(
                f"trusted_title PMC char share {pmc_share:.4f} exceeds "
                f"{args.max_pmc_char_share:.4f}"
            )
        disallowed_pmc_content = {
            status: count
            for status, count in report["pmc_content_review_statuses"].items()
            if status not in {"CONTENT_ACCEPTED", "HUMAN_ACCEPTED"}
        }
        if disallowed_pmc_content:
            raise TitleMaskError(
                f"trusted_title has unaccepted PMC content status: {disallowed_pmc_content}"
            )
    elif args.recipe == "readoc_view_ablation":
        if set(family_share) != {"readoc"}:
            raise TitleMaskError(f"readoc_view_ablation must contain only READoc; got {family_share}")
        if sample_forms != {"full_document"}:
            raise TitleMaskError(
                f"readoc_view_ablation requires full_document rows; got {sorted(sample_forms)}"
            )
        if set(report["title_review_statuses"]) != {"SILVER_ACCEPTED"}:
            raise TitleMaskError(
                "readoc_view_ablation requires rule-derived SILVER_ACCEPTED headings; "
                f"got {report['title_review_statuses']}"
            )
    else:  # pragma: no cover - argparse owns the enum
        raise AssertionError(args.recipe)


def main() -> int:
    args = parse_args()
    for value, name in (
        (args.min_cjk_char_share, "--min-cjk-char-share"),
        (args.max_pmc_char_share, "--max-pmc-char-share"),
    ):
        if not 0 <= value <= 1:
            raise TitleMaskError(f"{name} must be between 0 and 1")
    report = audit(
        args.jsonl,
        check_images=not args.skip_image_existence,
        full_row_validation=args.full_row_validation,
    )
    report["recipe"] = args.recipe
    report["full_row_validation"] = args.full_row_validation
    validate_recipe(report, args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TitleMaskError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}")
