"""Pure helpers shared by the data gate, validator, plugin, and tests."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

from markdown_it import MarkdownIt


EXPECTED_CHANNEL = "title_reviewed"
SINGLE_PAGE_PROMPT = "<image>document parsing."
MULTI_PAGE_PROMPT = "<image>Multi page merge."
EXPECTED_PROMPTS = frozenset({SINGLE_PAGE_PROMPT, MULTI_PAGE_PROMPT})
MINERU_HEADING_PRIOR_INSTRUCTION = (
    "The following headings are extracted from the MinerU parsing result of the same document.\n"
    "Use them only as structural hints.\n"
    "Verify and correct the heading text, heading levels, ordering, and document structure "
    "according to the document images.\n"
    "Do not blindly copy the MinerU headings if they conflict with the document images."
)
MINERU_HEADING_PRIOR_PREFIX = (
    f"{MULTI_PAGE_PROMPT}\n\n"
    f"{MINERU_HEADING_PRIOR_INSTRUCTION}\n\n"
    "### MinerU headings\n"
)
MINERU_HEADING_PRIOR_SUFFIX = "\n\n### Final corrected Markdown\n"
HUMAN_ACCEPTED = "HUMAN_ACCEPTED"
SILVER_ACCEPTED = "SILVER_ACCEPTED"
ACCEPTED_TITLE_STATUSES = frozenset({HUMAN_ACCEPTED, SILVER_ACCEPTED})
ALLOWED_REVIEW_STATUSES = frozenset(
    ACCEPTED_TITLE_STATUSES
    | {"HUMAN_REJECTED", "REVIEW_REQUIRED", "SILVER_CANDIDATE"}
)
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})


class TitleMaskError(ValueError):
    """Raised when a row cannot be masked without changing its token semantics."""


@dataclass(frozen=True)
class HeadingSpan:
    start: int
    end: int
    level: int


def canonicalize_target(text: str) -> str:
    if not isinstance(text, str):
        raise TitleMaskError("assistant content must be a string")
    if "\x00" in text:
        raise TitleMaskError("assistant content contains NUL")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise TitleMaskError("assistant content is empty")
    return text.rstrip("\n") + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonicalize_prompt_image_prefix(prompt: object) -> object:
    """Collapse a multi-page prompt's repeated image placeholders to one."""
    if not isinstance(prompt, str) or not prompt.startswith("<image>"):
        return prompt
    remainder = prompt
    while remainder.startswith("<image>"):
        remainder = remainder[len("<image>") :]
    return "<image>" + remainder


def is_mineru_heading_prior_prompt(prompt: object) -> bool:
    """Return whether prompt is the reviewed MinerU heading-prior format."""
    prompt = canonicalize_prompt_image_prefix(prompt)
    if not isinstance(prompt, str) or not prompt.startswith(MINERU_HEADING_PRIOR_PREFIX):
        return False
    if not prompt.endswith(MINERU_HEADING_PRIOR_SUFFIX):
        return False
    heading_text = prompt[len(MINERU_HEADING_PRIOR_PREFIX) : -len(MINERU_HEADING_PRIOR_SUFFIX)]
    return bool(heading_text.strip())


def is_supported_prompt(prompt: object) -> bool:
    prompt = canonicalize_prompt_image_prefix(prompt)
    return prompt in EXPECTED_PROMPTS or is_mineru_heading_prior_prompt(prompt)


def _line_starts(text: str) -> List[int]:
    starts = [0]
    starts.extend(i + 1 for i, char in enumerate(text) if char == "\n")
    if starts[-1] != len(text):
        starts.append(len(text))
    return starts


def atx_heading_spans(text: str) -> List[HeadingSpan]:
    """Return source spans for CommonMark ATX headings, including line endings."""
    if not isinstance(text, str):
        raise TitleMaskError("Markdown target must be a string")
    starts = _line_starts(text)
    spans: List[HeadingSpan] = []
    for token in MarkdownIt("commonmark").parse(text):
        if token.type != "heading_open" or not token.markup.startswith("#"):
            continue
        if token.map is None or len(token.map) != 2:
            raise TitleMaskError("Markdown parser returned a heading without source lines")
        start_line, end_line = token.map
        if not (0 <= start_line < end_line < len(starts)):
            raise TitleMaskError(f"invalid heading line map: {token.map!r}")
        level_text = token.tag.removeprefix("h")
        if not level_text.isdigit():
            raise TitleMaskError(f"invalid heading tag: {token.tag!r}")
        spans.append(HeadingSpan(starts[start_line], starts[end_line], int(level_text)))
    for previous, current in zip(spans, spans[1:]):
        if previous.end > current.start:
            raise TitleMaskError("overlapping Markdown heading spans")
    return spans


def heading_count(text: str) -> int:
    return len(atx_heading_spans(text))


def response_token_weights(
    text: str,
    offsets: Sequence[Sequence[int]],
) -> Tuple[List[int], List[HeadingSpan]]:
    """Map tokenizer offsets to a 0/1 mask without guessing boundary tokens."""
    spans = atx_heading_spans(text)
    if not spans:
        raise TitleMaskError("assistant content has no CommonMark ATX heading")

    heading_chars = bytearray(len(text))
    for span in spans:
        heading_chars[span.start:span.end] = b"\x01" * (span.end - span.start)

    weights: List[int] = []
    previous_start = 0
    for token_index, offset in enumerate(offsets):
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise TitleMaskError(f"token {token_index} has invalid offset: {offset!r}")
        start, end = offset
        if not isinstance(start, int) or not isinstance(end, int):
            raise TitleMaskError(f"token {token_index} offset is not integral: {offset!r}")
        if start < previous_start or not (0 <= start < end <= len(text)):
            raise TitleMaskError(f"token {token_index} has unusable offset: {offset!r}")
        previous_start = start

        flags = heading_chars[start:end]
        chars = text[start:end]
        inside_nonspace = any(flag and not char.isspace() for flag, char in zip(flags, chars))
        outside_nonspace = any(not flag and not char.isspace() for flag, char in zip(flags, chars))
        if inside_nonspace and outside_nonspace:
            snippet = text[start:end].replace("\n", "\\n")
            raise TitleMaskError(
                f"token {token_index} crosses heading/body with content on both sides: {snippet!r}"
            )
        overlaps_heading = any(flags)
        weights.append(int(inside_nonspace or (overlaps_heading and not outside_nonspace)))

    if not any(weights):
        raise TitleMaskError("headings exist but no tokenizer token maps to them")
    return weights, spans


def validate_conversation(
    messages: Sequence[Mapping[str, object]],
    *,
    image_count: int | None = None,
) -> str:
    if not isinstance(messages, (list, tuple)) or len(messages) != 2:
        raise TitleMaskError("expected exactly one user turn and one assistant turn")
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise TitleMaskError("messages must be [user, assistant]")
    prompt = user.get("content")
    if not is_supported_prompt(prompt):
        raise TitleMaskError(
            "prompt must be a supported Unlimited-OCR prompt or MinerU heading-prior prompt; "
            f"got {prompt!r}"
        )
    canonical_prompt = canonicalize_prompt_image_prefix(prompt)
    if image_count is not None:
        if not isinstance(image_count, int) or isinstance(image_count, bool) or image_count <= 0:
            raise TitleMaskError("image_count must be a positive integer")
        if canonical_prompt == SINGLE_PAGE_PROMPT and image_count != 1:
            raise TitleMaskError("single-page prompt requires exactly one image")
        if canonical_prompt != SINGLE_PAGE_PROMPT and image_count < 2:
            raise TitleMaskError("multi-page prompt requires at least two images")
    response = assistant.get("content")
    if not isinstance(response, str):
        raise TitleMaskError("assistant content must be a string")
    return response


def locate_response_slice(labels: Sequence[int], response_ids: Sequence[int]) -> Tuple[int, int]:
    supervised = [i for i, label in enumerate(labels) if label != -100]
    if not supervised:
        raise TitleMaskError("native template produced no supervised labels")
    start = supervised[0]
    end = start + len(response_ids)
    if list(labels[start:end]) != list(response_ids):
        raise TitleMaskError("response token IDs do not match the native template labels")
    return start, end


def build_sequence_loss_scale(
    input_ids: Sequence[int],
    labels: Sequence[int],
    response_ids: Sequence[int],
    response_weights: Sequence[int],
    suffix_ids: Sequence[int],
    *,
    body_weight: float = 0.0,
    title_weight: float = 1.0,
    eos_weight: float = 1.0,
) -> List[float]:
    """Create title+EOS weights while preserving native input IDs and labels."""
    for name, value in (
        ("body_weight", body_weight),
        ("title_weight", title_weight),
        ("eos_weight", eos_weight),
    ):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TitleMaskError(f"{name} must be a finite number")
        if value < 0:
            raise TitleMaskError(f"{name} must be non-negative")
    if title_weight <= 0:
        raise TitleMaskError("title_weight must be positive")
    if len(input_ids) != len(labels):
        raise TitleMaskError("input_ids and labels have different lengths")
    if len(response_ids) != len(response_weights):
        raise TitleMaskError("response IDs and response weights have different lengths")
    if not suffix_ids:
        raise TitleMaskError("native template suffix is empty; expected EOS")

    start, response_end = locate_response_slice(labels, response_ids)
    suffix_end = response_end + len(suffix_ids)
    if list(labels[response_end:suffix_end]) != list(suffix_ids):
        raise TitleMaskError("native template suffix does not follow the response exactly")
    if any(label != -100 for label in labels[suffix_end:]):
        raise TitleMaskError("unexpected supervised labels after the native suffix")
    if list(input_ids[start:response_end]) != list(response_ids):
        raise TitleMaskError("response token IDs do not match native input_ids")
    if list(input_ids[response_end:suffix_end]) != list(suffix_ids):
        raise TitleMaskError("suffix token IDs do not match native input_ids")

    loss_scale = [0.0] * len(input_ids)
    loss_scale[start:response_end] = [
        float(title_weight if weight else body_weight)
        for weight in response_weights
    ]
    loss_scale[response_end:suffix_end] = [float(eos_weight)] * len(suffix_ids)
    if not any(loss_scale[start:response_end]):
        raise TitleMaskError("title mask has no active response token")
    return loss_scale


def rswa_prefix_length(labels: Sequence[int]) -> int:
    return next((i for i, label in enumerate(labels) if label != -100), len(labels))


def expected_ddp_gradient(local_gradient_sums: Iterable[float], global_weight_sum: float) -> float:
    """Reference value used by the CPU-only DDP math tests."""
    if global_weight_sum <= 0:
        raise TitleMaskError("global loss weight must be positive")
    return sum(local_gradient_sums) / global_weight_sum
