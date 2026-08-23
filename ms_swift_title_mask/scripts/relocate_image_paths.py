#!/usr/bin/env python3
"""Relocate frozen image references to an arbitrary server page root."""

from __future__ import annotations

import argparse
import copy
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


SCHEMA_VERSION = "uocr-image-path-relocation-v1"
ALLOWED_SOURCES = {"READoc-arxiv", "READoc-github", "PMC-v26-synthetic"}


def _safe_doc_id(doc_id: object) -> str:
    if (
        not isinstance(doc_id, str)
        or not doc_id
        or doc_id in {".", ".."}
        or "/" in doc_id
        or "\\" in doc_id
    ):
        raise TitleMaskError(f"unsafe doc_id: {doc_id!r}")
    return doc_id


def relative_page_path(source: str, doc_id: str, page: int) -> Path:
    if source == "PMC-v26-synthetic":
        return Path("pmc") / doc_id / f"page_{page:04d}.png"
    if source in {"READoc-arxiv", "READoc-github"}:
        family = source.removeprefix("READoc-")
        return Path("readoc") / family / doc_id / f"page_{page:04d}.png"
    raise TitleMaskError(f"unsupported source: {source!r}")


def relocate_row(row: dict, *, split: str, image_root: Path, target_owners: dict[Path, tuple]) -> dict:
    summary = validate_training_row(row, expected_split=split, check_images=False)
    meta = row["meta"]
    source = meta.get("source")
    if source not in ALLOWED_SOURCES:
        raise TitleMaskError(f"{summary['id']}: unsupported source {source!r}")
    doc_id = _safe_doc_id(meta.get("doc_id"))
    pages, images = meta.get("page_indices"), row.get("images")
    if (
        not isinstance(pages, list)
        or not pages
        or any(not isinstance(page, int) or isinstance(page, bool) or page < 0 for page in pages)
        or len(set(pages)) != len(pages)
        or pages != sorted(pages)
        or not isinstance(images, list)
        or len(images) != len(pages)
        or meta.get("n_pages") != len(pages)
    ):
        raise TitleMaskError(f"{summary['id']}: invalid page_indices/n_pages/images mapping")

    relocated = []
    for page, original_text in zip(pages, images):
        if not isinstance(original_text, str) or not Path(original_text).is_absolute():
            raise TitleMaskError(f"{summary['id']}: source image path must be absolute")
        relative = relative_page_path(source, doc_id, page)
        original = Path(original_text)
        if len(original.parts) < len(relative.parts) or original.parts[-len(relative.parts):] != relative.parts:
            raise TitleMaskError(
                f"{summary['id']}: source image path does not match source/doc/page: {original_text}"
            )
        target = (image_root / relative).resolve()
        if not target.is_relative_to(image_root) or target == image_root:
            raise TitleMaskError(f"{summary['id']}: relocated image escapes image root")
        owner = (source, doc_id, page)
        previous = target_owners.setdefault(target, owner)
        if previous != owner:
            raise TitleMaskError(f"{summary['id']}: relocated image target collision: {target}")
        relocated.append(str(target))

    output = copy.deepcopy(row)
    output["images"] = relocated
    if output["messages"] != row["messages"] or output["meta"] != row["meta"]:
        raise TitleMaskError(f"{summary['id']}: relocation changed supervision or provenance")
    validate_training_row(output, expected_split=split, check_images=False)
    return output


def build_relocated_dataset(
    input_dir: Path,
    output_dir: Path,
    image_root: Path,
    *,
    overwrite: bool = False,
) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not image_root.is_absolute():
        raise TitleMaskError("--image-root must be absolute")
    image_root = image_root.resolve()
    if image_root == Path(image_root.anchor):
        raise TitleMaskError("--image-root cannot be a filesystem root")
    if output_dir == input_dir or output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
        raise TitleMaskError("input and output directories must be separate and non-nested")

    inputs = {split: input_dir / f"{split}.jsonl" for split in sorted(ALLOWED_SPLITS)}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise TitleMaskError(f"missing input split files: {missing}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp.", dir=output_dir.parent))
    target_owners: dict[Path, tuple] = {}
    split_counts = Counter()
    source_counts = Counter()
    try:
        for split, input_path in inputs.items():
            relocated_rows = []
            for line_number, row in read_jsonl(input_path):
                try:
                    relocated = relocate_row(
                        row,
                        split=split,
                        image_root=image_root,
                        target_owners=target_owners,
                    )
                except TitleMaskError as exc:
                    raise TitleMaskError(f"{input_path}:{line_number}: {exc}") from exc
                relocated_rows.append(relocated)
                split_counts[split] += 1
                source_counts[str(relocated["meta"]["source"])] += 1
            if not relocated_rows:
                raise TitleMaskError(f"input split is empty: {input_path}")
            dump_jsonl(temp_dir / f"{split}.jsonl", relocated_rows)

        report = {
            "schema_version": SCHEMA_VERSION,
            "input_dir": str(input_dir),
            "image_root": str(image_root),
            "rows": sum(split_counts.values()),
            "unique_image_targets": len(target_owners),
            "split_counts": dict(sorted(split_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "input_sha256": {split: file_sha256(path) for split, path in inputs.items()},
            "output_sha256": {
                split: file_sha256(temp_dir / f"{split}.jsonl") for split in sorted(ALLOWED_SPLITS)
            },
        }
        (temp_dir / "relocation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_directory(temp_dir, output_dir, overwrite)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_relocated_dataset(
            args.input_dir,
            args.output_dir,
            args.image_root,
            overwrite=args.overwrite,
        )
    except (TitleMaskError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
