#!/usr/bin/env python3
"""Build deterministic experiment mixes with per-pool target-character caps."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

_BUNDLE_PARENT = Path(__file__).resolve().parents[2]
if str(_BUNDLE_PARENT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_PARENT))

from ms_swift_title_mask.core import ALLOWED_SPLITS, TitleMaskError  # noqa: E402
from ms_swift_title_mask.data_contract import (  # noqa: E402
    dump_jsonl,
    file_sha256,
    publish_directory,
    read_jsonl,
    validate_training_row,
)


MIX_VERSION = "uocr-recipe-mix-v1"


def _assignment(value: str, *, option: str) -> tuple[str, str]:
    name, separator, raw_value = value.partition("=")
    if not separator or not name or not raw_value:
        raise argparse.ArgumentTypeError(f"{option} must use NAME=VALUE")
    return name, raw_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="split JSONL directory; repeat for each input pool",
    )
    parser.add_argument(
        "--cap-target-char-share",
        action="append",
        default=[],
        metavar="NAME=FRACTION",
        help="deterministically cap an auxiliary pool by assistant-character share",
    )
    parser.add_argument(
        "--require-content-status",
        action="append",
        default=[],
        metavar="NAME=STATUS",
        help="require meta.content_review_status for every row in a pool",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="uocr-recipe-mix-20260826-v1")
    parser.add_argument(
        "--exclude-row-id",
        action="append",
        default=[],
        help="explicitly exclude a reviewed row and record the decision; repeatable",
    )
    parser.add_argument("--skip-image-existence", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_configuration(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, float], dict[str, str]]:
    pools: dict[str, Path] = {}
    for value in args.pool:
        name, raw_path = _assignment(value, option="--pool")
        if name in pools:
            raise TitleMaskError(f"duplicate pool name: {name}")
        pools[name] = Path(raw_path).expanduser().resolve()

    caps = {}
    for value in args.cap_target_char_share:
        name, raw_fraction = _assignment(value, option="--cap-target-char-share")
        if name not in pools:
            raise TitleMaskError(f"cap references unknown pool: {name}")
        try:
            fraction = float(raw_fraction)
        except ValueError as exc:
            raise TitleMaskError(f"invalid cap for {name}: {raw_fraction!r}") from exc
        if not 0 < fraction < 1:
            raise TitleMaskError(f"cap for {name} must be between 0 and 1")
        caps[name] = fraction

    content_status = {}
    for value in args.require_content_status:
        name, status = _assignment(value, option="--require-content-status")
        if name not in pools:
            raise TitleMaskError(f"content status references unknown pool: {name}")
        content_status[name] = status
    return pools, caps, content_status


def stable_key(seed: str, split: str, pool: str, row_id: str) -> str:
    return hashlib.sha256(f"{seed}:{split}:{pool}:{row_id}".encode("utf-8")).hexdigest()


def document_key(row: dict) -> tuple[str, str]:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise TitleMaskError(f"{row.get('id')}: meta must be an object")
    source = meta.get("source")
    doc_id = meta.get("doc_id")
    if not isinstance(source, str) or not source or not isinstance(doc_id, str) or not doc_id:
        raise TitleMaskError(f"{row.get('id')}: meta.source/doc_id must be non-empty strings")
    return source, doc_id


def load_pool(
    root: Path,
    pool: str,
    split: str,
    *,
    check_images: bool,
    content_status: str | None,
    excluded_ids: frozenset[str],
) -> tuple[list[dict], Path, list[dict]]:
    path = root / f"{split}.jsonl"
    rows = []
    seen_ids = set()
    seen_documents = set()
    excluded = []
    for line_number, row in read_jsonl(path):
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id in excluded_ids:
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            excluded.append(
                {
                    "id": row_id,
                    "pool": pool,
                    "split": split,
                    "source": meta.get("source"),
                    "doc_id": meta.get("doc_id"),
                }
            )
            continue
        try:
            validate_training_row(row, expected_split=split, check_images=check_images)
            key = document_key(row)
        except TitleMaskError as exc:
            raise TitleMaskError(f"{path}:{line_number}: {exc}") from exc
        row_id = row["id"]
        if row_id in seen_ids:
            raise TitleMaskError(f"{path}:{line_number}: duplicate row id {row_id!r}")
        if key in seen_documents:
            raise TitleMaskError(f"{path}:{line_number}: pool {pool} repeats document {key}")
        if content_status is not None and row["meta"].get("content_review_status") != content_status:
            raise TitleMaskError(
                f"{path}:{line_number}: expected content_review_status={content_status!r}"
            )
        seen_ids.add(row_id)
        seen_documents.add(key)
        rows.append(row)
    if not rows:
        raise TitleMaskError(f"empty input pool: {path}")
    return rows, path, excluded


def _target_chars(row: dict) -> int:
    return len(row["messages"][1]["content"])


def select_capped_rows(
    rows: list[dict],
    *,
    maximum_chars: int,
    seed: str,
    split: str,
    pool: str,
) -> list[dict]:
    ordered = sorted(rows, key=lambda row: stable_key(seed, split, pool, str(row["id"])))
    selected = []
    used = 0
    for row in ordered:
        size = _target_chars(row)
        if used + size > maximum_chars:
            continue
        selected.append(row)
        used += size
    if not selected:
        raise TitleMaskError(
            f"cap for {pool}/{split} selected no rows; maximum_chars={maximum_chars}"
        )
    return selected


def build_split(
    loaded: dict[str, list[dict]],
    caps: dict[str, float],
    *,
    seed: str,
    split: str,
) -> tuple[list[dict], dict]:
    uncapped_names = sorted(set(loaded) - set(caps))
    selected = {name: list(loaded[name]) for name in uncapped_names}
    current_chars = sum(_target_chars(row) for rows in selected.values() for row in rows)
    if not uncapped_names:
        raise TitleMaskError("at least one uncapped base pool is required")

    for name in sorted(caps):
        share = caps[name]
        maximum = math_floor(share * current_chars / (1.0 - share))
        selected[name] = select_capped_rows(
            loaded[name],
            maximum_chars=maximum,
            seed=seed,
            split=split,
            pool=name,
        )
        current_chars += sum(_target_chars(row) for row in selected[name])

    combined = []
    pool_report = {}
    for name in sorted(selected):
        rows = selected[name]
        chars = sum(_target_chars(row) for row in rows)
        images = sum(len(row["images"]) for row in rows)
        forms = Counter(row["meta"].get("sample_form", "<missing>") for row in rows)
        pool_report[name] = {
            "rows": len(rows),
            "target_chars": chars,
            "images": images,
            "sample_forms": dict(sorted(forms.items())),
        }
        for row in rows:
            annotated = copy.deepcopy(row)
            annotated["meta"]["recipe_mix_pool"] = name
            annotated["meta"]["recipe_mix_version"] = MIX_VERSION
            combined.append(annotated)

    seen_ids = set()
    seen_documents = set()
    for row in combined:
        if row["id"] in seen_ids:
            raise TitleMaskError(f"cross-pool duplicate row id: {row['id']}")
        key = document_key(row)
        if key in seen_documents:
            raise TitleMaskError(
                f"cross-pool duplicate document {key}; do not include full and single variants together"
            )
        seen_ids.add(row["id"])
        seen_documents.add(key)

    combined.sort(key=lambda row: stable_key(seed, split, "combined", str(row["id"])))
    total_chars = sum(stats["target_chars"] for stats in pool_report.values())
    total_rows = sum(stats["rows"] for stats in pool_report.values())
    for stats in pool_report.values():
        stats["row_share"] = stats["rows"] / total_rows
        stats["target_char_share"] = stats["target_chars"] / total_chars
    return combined, {
        "rows": total_rows,
        "target_chars": total_chars,
        "pools": pool_report,
    }


def math_floor(value: float) -> int:
    # Keep selection deterministic at the integer boundary without importing numpy.
    return int(value // 1)


def main() -> int:
    args = parse_args()
    pools, caps, content_status = parse_configuration(args)
    if len(args.exclude_row_id) != len(set(args.exclude_row_id)):
        raise TitleMaskError("--exclude-row-id values must be unique")
    if any(not row_id for row_id in args.exclude_row_id):
        raise TitleMaskError("--exclude-row-id values must be non-empty")
    excluded_ids = frozenset(args.exclude_row_id)
    check_images = not args.skip_image_existence
    output_dir = args.output_dir.expanduser().resolve()
    for root in pools.values():
        if output_dir == root or output_dir.is_relative_to(root) or root.is_relative_to(output_dir):
            raise TitleMaskError("output directory and pool roots must be separate and non-nested")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    try:
        paths = {}
        split_reports = {}
        document_splits = {}
        excluded_records = []
        for split in sorted(ALLOWED_SPLITS):
            loaded = {}
            paths[split] = {}
            for name, root in pools.items():
                rows, path, excluded = load_pool(
                    root,
                    name,
                    split,
                    check_images=check_images,
                    content_status=content_status.get(name),
                    excluded_ids=excluded_ids,
                )
                loaded[name] = rows
                paths[split][name] = path
                excluded_records.extend(excluded)
            rows, split_report = build_split(loaded, caps, seed=args.seed, split=split)
            for row in rows:
                key = document_key(row)
                previous = document_splits.setdefault(key, split)
                if previous != split:
                    raise TitleMaskError(f"document leakage: {key} appears in {previous} and {split}")
            dump_jsonl(temp_dir / f"{split}.jsonl", rows)
            split_reports[split] = split_report

        excluded_counts = Counter(row["id"] for row in excluded_records)
        missing_exclusions = sorted(excluded_ids - set(excluded_counts))
        repeated_exclusions = sorted(row_id for row_id, count in excluded_counts.items() if count != 1)
        if missing_exclusions or repeated_exclusions:
            raise TitleMaskError(
                "explicit exclusions must match exactly one input row: "
                f"missing={missing_exclusions} repeated={repeated_exclusions}"
            )

        report = {
            "schema_version": MIX_VERSION,
            "seed": args.seed,
            "caps": caps,
            "required_content_status": content_status,
            "excluded_rows": sorted(excluded_records, key=lambda row: row["id"]),
            "input_sha256": {
                split: {name: file_sha256(path) for name, path in split_paths.items()}
                for split, split_paths in paths.items()
            },
            "splits": split_reports,
        }
        (temp_dir / "recipe_mix_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_directory(temp_dir, output_dir, args.overwrite)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (TitleMaskError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}")
