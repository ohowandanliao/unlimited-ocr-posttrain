"""Fail-closed data contract for reviewed title training."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from .core import (
    ACCEPTED_TITLE_STATUSES,
    ALLOWED_REVIEW_STATUSES,
    ALLOWED_SPLITS,
    EXPECTED_CHANNEL,
    HUMAN_ACCEPTED,
    TitleMaskError,
    canonicalize_target,
    heading_count,
    sha256_text,
    validate_conversation,
)


HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def read_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, object]]]:
    if not path.is_file():
        raise TitleMaskError(f"JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise TitleMaskError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise TitleMaskError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise TitleMaskError(f"{path}:{line_number}: row must be an object")
            yield line_number, row


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TitleMaskError(f"{field} must be a non-empty string")
    return value


def validate_review_record(row: Mapping[str, object]) -> Dict[str, object]:
    row_id = require_nonempty_string(row.get("id"), "review.id")
    status = require_nonempty_string(row.get("status"), f"review[{row_id}].status")
    if status not in ALLOWED_REVIEW_STATUSES:
        raise TitleMaskError(f"review[{row_id}] has unknown status {status!r}")
    normalized = {"id": row_id, "status": status}
    if status not in ACCEPTED_TITLE_STATUSES:
        return normalized

    split = require_nonempty_string(row.get("split"), f"review[{row_id}].split")
    if split not in ALLOWED_SPLITS:
        raise TitleMaskError(f"review[{row_id}] has invalid split {split!r}")
    review_id = require_nonempty_string(row.get("review_id"), f"review[{row_id}].review_id")
    review_version = require_nonempty_string(
        row.get("review_version"), f"review[{row_id}].review_version"
    )
    reviewer = require_nonempty_string(row.get("reviewer"), f"review[{row_id}].reviewer")
    reviewed_at = require_nonempty_string(row.get("reviewed_at"), f"review[{row_id}].reviewed_at")
    try:
        parsed_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TitleMaskError(f"review[{row_id}].reviewed_at is not ISO-8601") from exc
    if parsed_time.tzinfo is None:
        raise TitleMaskError(f"review[{row_id}].reviewed_at must include a timezone")

    assistant_sha256 = require_nonempty_string(
        row.get("assistant_sha256"), f"review[{row_id}].assistant_sha256"
    )
    if not HEX_SHA256.fullmatch(assistant_sha256):
        raise TitleMaskError(f"review[{row_id}].assistant_sha256 is not lowercase SHA-256")
    title_heading_count = row.get("title_heading_count")
    if not isinstance(title_heading_count, int) or isinstance(title_heading_count, bool) or title_heading_count <= 0:
        raise TitleMaskError(f"review[{row_id}].title_heading_count must be a positive integer")

    normalized.update(
        {
            "split": split,
            "review_id": review_id,
            "review_version": review_version,
            "reviewer": reviewer,
            "reviewed_at": reviewed_at,
            "assistant_sha256": assistant_sha256,
            "title_heading_count": title_heading_count,
        }
    )
    if status != HUMAN_ACCEPTED:
        ruleset_version = require_nonempty_string(
            row.get("ruleset_version"), f"review[{row_id}].ruleset_version"
        )
        source_sha256 = require_nonempty_string(
            row.get("source_sha256"), f"review[{row_id}].source_sha256"
        )
        if not HEX_SHA256.fullmatch(source_sha256):
            raise TitleMaskError(f"review[{row_id}].source_sha256 is not lowercase SHA-256")
        normalized.update(
            {
                "ruleset_version": ruleset_version,
                "source_sha256": source_sha256,
            }
        )
    return normalized


def source_messages(row: Mapping[str, object]) -> Tuple[List[Dict[str, str]], str]:
    if "messages" in row:
        messages = row["messages"]
        if not isinstance(messages, list):
            raise TitleMaskError("source.messages must be a list")
        response = validate_conversation(messages)
        return messages, response

    prompt = row.get("prompt")
    target = row.get("target")
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
    response = validate_conversation(messages)
    return messages, response


def resolve_images(
    images: object,
    *,
    image_root: Optional[Path],
    check_exists: bool,
) -> List[str]:
    if not isinstance(images, list) or not images:
        raise TitleMaskError("source.images must be a non-empty list")
    resolved: List[str] = []
    seen = set()
    root = image_root.resolve() if image_root is not None else None
    for index, value in enumerate(images):
        image_text = require_nonempty_string(value, f"images[{index}]")
        image = Path(image_text).expanduser()
        if not image.is_absolute():
            if root is None:
                raise TitleMaskError("relative image paths require --image-root")
            image = (root / image).resolve()
            if not image.is_relative_to(root):
                raise TitleMaskError(f"image path escapes --image-root: {image_text}")
        else:
            image = image.resolve()
        image_key = str(image)
        if image_key in seen:
            raise TitleMaskError(f"duplicate image path in one document: {image_key}")
        if check_exists and not image.is_file():
            raise TitleMaskError(f"image does not exist: {image}")
        seen.add(image_key)
        resolved.append(image_key)
    return resolved


def build_reviewed_row(
    source: Mapping[str, object],
    review: Mapping[str, object],
    *,
    image_root: Optional[Path],
    check_images: bool,
) -> Dict[str, object]:
    row_id = require_nonempty_string(source.get("id"), "source.id")
    if review.get("id") != row_id or review.get("status") not in ACCEPTED_TITLE_STATUSES:
        raise TitleMaskError(f"source/review mismatch for {row_id}")
    messages, response = source_messages(source)
    response = canonicalize_target(response)
    actual_sha = sha256_text(response)
    if actual_sha != review["assistant_sha256"]:
        raise TitleMaskError(
            f"{row_id}: reviewed assistant SHA mismatch: expected {review['assistant_sha256']}, got {actual_sha}"
        )
    actual_heading_count = heading_count(response)
    if actual_heading_count != review["title_heading_count"]:
        raise TitleMaskError(
            f"{row_id}: reviewed heading count mismatch: expected {review['title_heading_count']}, "
            f"got {actual_heading_count}"
        )
    if actual_heading_count <= 0:
        raise TitleMaskError(f"{row_id}: accepted target has no ATX heading")

    images = resolve_images(source.get("images"), image_root=image_root, check_exists=check_images)
    validate_conversation(messages, image_count=len(images))
    source_meta = source.get("meta") or {}
    if not isinstance(source_meta, dict):
        raise TitleMaskError(f"{row_id}: source.meta must be an object")
    meta = dict(source_meta)
    meta.update(
        {
            "split": review["split"],
            "title_review_status": review["status"],
            "title_review_id": review["review_id"],
            "title_review_version": review["review_version"],
            "title_reviewer": review["reviewer"],
            "title_reviewed_at": review["reviewed_at"],
            "title_target_sha256": actual_sha,
            "title_heading_count": actual_heading_count,
        }
    )
    if review["status"] != HUMAN_ACCEPTED:
        meta["title_ruleset_version"] = review["ruleset_version"]
        meta["title_source_sha256"] = review["source_sha256"]
    return {
        "id": row_id,
        "channel": EXPECTED_CHANNEL,
        "messages": [
            {"role": "user", "content": messages[0]["content"]},
            {"role": "assistant", "content": response},
        ],
        "images": images,
        "meta": meta,
    }


def validate_training_row(
    row: Mapping[str, object],
    *,
    expected_split: Optional[str],
    check_images: bool,
) -> Dict[str, object]:
    row_id = require_nonempty_string(row.get("id"), "row.id")
    if row.get("channel") != EXPECTED_CHANNEL:
        raise TitleMaskError(f"{row_id}: channel must be {EXPECTED_CHANNEL!r}")
    images = resolve_images(row.get("images"), image_root=None, check_exists=check_images)
    response = validate_conversation(row.get("messages"), image_count=len(images))
    canonical = canonicalize_target(response)
    if response != canonical:
        raise TitleMaskError(f"{row_id}: assistant target is not in canonical newline form")
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise TitleMaskError(f"{row_id}: meta must be an object")
    if meta.get("title_review_status") not in ACCEPTED_TITLE_STATUSES:
        raise TitleMaskError(f"{row_id}: title_review_status is not accepted")
    split = meta.get("split")
    if split not in ALLOWED_SPLITS:
        raise TitleMaskError(f"{row_id}: invalid split {split!r}")
    if expected_split is not None and split != expected_split:
        raise TitleMaskError(f"{row_id}: split {split!r} does not match file split {expected_split!r}")
    for key in ("title_review_id", "title_review_version", "title_reviewer", "title_reviewed_at"):
        require_nonempty_string(meta.get(key), f"{row_id}.meta.{key}")
    if meta.get("title_review_status") != HUMAN_ACCEPTED:
        require_nonempty_string(meta.get("title_ruleset_version"), f"{row_id}.meta.title_ruleset_version")
        source_sha256 = require_nonempty_string(
            meta.get("title_source_sha256"), f"{row_id}.meta.title_source_sha256"
        )
        if not HEX_SHA256.fullmatch(source_sha256):
            raise TitleMaskError(f"{row_id}: title_source_sha256 is not lowercase SHA-256")
    actual_sha = sha256_text(response)
    if meta.get("title_target_sha256") != actual_sha:
        raise TitleMaskError(f"{row_id}: title_target_sha256 mismatch")
    actual_count = heading_count(response)
    if actual_count <= 0 or meta.get("title_heading_count") != actual_count:
        raise TitleMaskError(f"{row_id}: title_heading_count mismatch")
    return {
        "id": row_id,
        "split": split,
        "images": len(images),
        "headings": actual_count,
        "target_chars": len(response),
    }


def dump_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def publish_directory(temp_dir: Path, output_dir: Path, overwrite: bool) -> None:
    """Atomically publish a fully built directory, restoring the old one on failure."""
    if output_dir.exists() and not overwrite:
        raise TitleMaskError(f"output already exists; pass --overwrite explicitly: {output_dir}")
    backup = None
    if output_dir.exists():
        backup = output_dir.with_name(f"{output_dir.name}.backup.{uuid.uuid4().hex}")
        os.replace(output_dir, backup)
    try:
        os.replace(temp_dir, output_dir)
    except Exception:
        if backup is not None and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup)
