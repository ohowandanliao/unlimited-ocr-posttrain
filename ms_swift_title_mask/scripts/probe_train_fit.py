#!/usr/bin/env python3
"""Probe whether trained OCR adapters can fit their own training samples.

The input JSONL may use either the current messages/images schema or the
legacy target/prompt/images schema. Every request uses the prompt stored in
that row, so prompt-conditioned training schemes can be compared without
silently replacing their inputs.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import io
import json
import re
import tempfile
import time
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


BASE_NAME = "base"


class InferenceReceiver:
    """Expose Unlimited-OCR custom inference methods on a PeftModel."""

    def __init__(self, receiver: Any, model_class: type[Any], tokenizer: Any):
        self.receiver = receiver
        self.model_class = model_class
        self.tokenizer = tokenizer

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_class.infer(self.receiver, *args, **kwargs)

    def infer_multi(self, *args: Any, **kwargs: Any) -> Any:
        return self.model_class.infer_multi(self.receiver, *args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="adapter label and directory; may be repeated",
    )
    parser.add_argument(
        "--sample-line",
        action="append",
        type=int,
        default=[],
        help="0-based JSONL line to probe; may be repeated",
    )
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35)
    parser.add_argument("--ngram-window", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--single-image-size", type=int, default=640)
    parser.add_argument("--single-crop-mode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_adapters(raw_values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    names = {BASE_NAME}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"adapter must be NAME=PATH, got {raw!r}")
        name, raw_path = raw.split("=", 1)
        if not name or name in names:
            raise ValueError(f"duplicate or invalid adapter name: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"adapter directory does not exist: {path}")
        parsed.append((name, path))
        names.add(name)
    return parsed


def extract_record(item: dict[str, Any], line_no: int) -> dict[str, Any]:
    images = item.get("images") or item.get("image_paths") or item.get("image")
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list) or not images:
        raise ValueError(f"line {line_no}: missing images")
    images = [str(path) for path in images]

    target = item.get("target")
    prompt = item.get("prompt")
    if target is None:
        messages = item.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"line {line_no}: missing target/messages")
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        if not user_messages or not assistant_messages:
            raise ValueError(f"line {line_no}: messages lack user/assistant")
        prompt = user_messages[0].get("content")
        target = assistant_messages[-1].get("content")
    if not isinstance(prompt, str) or not isinstance(target, str):
        raise ValueError(f"line {line_no}: prompt and target must be strings")

    return {
        "line": line_no,
        "id": str(item.get("id") or f"line-{line_no}"),
        "prompt": prompt,
        "target": target,
        "images": images,
        "source": item.get("source"),
        "mode": item.get("mode"),
        "meta": item.get("meta"),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line.strip():
                records.append(extract_record(json.loads(line), line_no))
    if not records:
        raise ValueError(f"dataset is empty: {path}")
    return records


def choose_records(
    records: list[dict[str, Any]], requested_lines: list[int], count: int
) -> list[dict[str, Any]]:
    by_line = {record["line"]: record for record in records}
    if requested_lines:
        missing = sorted(set(requested_lines) - set(by_line))
        if missing:
            raise ValueError(f"requested sample lines are missing: {missing}")
        return [by_line[line] for line in requested_lines]

    if count < 1:
        raise ValueError("--sample-count must be positive")
    ordered = sorted(
        records,
        key=lambda record: (len(record["target"]), len(record["images"]), record["line"]),
    )
    if count >= len(ordered):
        return ordered
    positions = [
        round(index * (len(ordered) - 1) / (count - 1))
        for index in range(count)
    ] if count > 1 else [len(ordered) // 2]
    selected = []
    seen = set()
    for position in positions:
        record = ordered[position]
        if record["line"] not in seen:
            selected.append(record)
            seen.add(record["line"])
    return selected


def normalize(text: str) -> str:
    return "\n".join(
        " ".join(line.split())
        for line in text.replace("\r\n", "\n").splitlines()
        if line.strip()
    )


def heading_texts(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", match.group(1).strip()).lower()
        for match in re.finditer(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*$", text)
    ]


def diagnostics(target: str, output: str) -> dict[str, Any]:
    normalized_target = normalize(target)
    normalized_output = normalize(output)
    target_lines = normalized_target.splitlines()
    output_lines = normalized_output.splitlines()
    common_lines = sum(
        (collections.Counter(target_lines) & collections.Counter(output_lines)).values()
    )
    target_heads = heading_texts(target)
    output_heads = heading_texts(output)
    common_heads = sum(
        (collections.Counter(target_heads) & collections.Counter(output_heads)).values()
    )
    output_counts = collections.Counter(output_lines)
    duplicate_lines = sum(n - 1 for n in output_counts.values() if n > 1)
    length_ratio = len(output) / len(target) if target else 0.0
    grounding = len(re.findall(r"<\|/?det\|>", output))
    output_tokens = None
    severe = []
    if length_ratio >= 4.0:
        severe.append("over_generation")
    if output_lines and duplicate_lines / len(output_lines) >= 0.5:
        severe.append("line_repetition")
    return {
        "output_chars": len(output),
        "length_ratio": length_ratio,
        "sequence_similarity": difflib.SequenceMatcher(
            None, normalized_target, normalized_output, autojunk=True
        ).ratio(),
        "target_line_recall": common_lines / len(target_lines) if target_lines else 0.0,
        "output_line_precision": common_lines / len(output_lines) if output_lines else 0.0,
        "target_heading_recall": common_heads / len(target_heads) if target_heads else 0.0,
        "target_heading_count": len(target_heads),
        "output_heading_count": len(output_heads),
        "duplicate_line_ratio": duplicate_lines / len(output_lines) if output_lines else 0.0,
        "max_line_occurrences": max(output_counts.values(), default=0),
        "grounding_tags": grounding,
        "page_tags": output.count("<PAGE>"),
        "protocol": (
            "mixed" if grounding and output_heads
            else "grounding" if grounding
            else "markdown" if output_heads
            else "unclassified"
        ),
        "severe": severe,
        "first_target_heading": target_heads[0] if target_heads else "",
        "first_output_heading": output_heads[0] if output_heads else "",
    }


def write_response_markdown(
    path: Path, result: dict[str, Any], target: str, output: str
) -> None:
    metrics = result.get("metrics", {})
    lines = [
        f"# Train Fit: line {result['sample_line']} / {result['model']}",
        "",
        f"- id: `{result['id']}`",
        f"- pages: {result['pages']}",
        f"- target chars: {result['target_chars']}",
        f"- target tokens: {result['target_tokens']}",
        f"- prompt: `{result['prompt']}`",
        f"- ok: `{result['ok']}`",
    ]
    if result["ok"]:
        lines.extend(
            [
                f"- output chars: {metrics.get('output_chars', '')}",
                f"- sequence similarity: {metrics.get('sequence_similarity', '')}",
                f"- target line recall: {metrics.get('target_line_recall', '')}",
                f"- target heading recall: {metrics.get('target_heading_recall', '')}",
                f"- protocol: `{metrics.get('protocol', '')}`",
                f"- severe: `{', '.join(metrics.get('severe', [])) or 'none'}`",
            ]
        )
    else:
        lines.append(f"- error: `{result.get('error', '')}`")
    lines.extend(
        [
            "",
            "## Target",
            "",
            "~~~text",
            target,
            "~~~",
            "",
            "## Output",
            "",
            "~~~text",
            output,
            "~~~",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_model(
    base_path: Path, adapter_paths: list[tuple[str, Path]], device: str, attn: str
) -> tuple[Any, Any, PeftModel | None, dict[str, str]]:
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, local_files_only=True
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
    }
    if attn:
        model_kwargs["attn_implementation"] = attn
    try:
        base_model = AutoModel.from_pretrained(
            base_path, dtype=torch.bfloat16, **model_kwargs
        )
    except TypeError:
        base_model = AutoModel.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, **model_kwargs
        )
    base_model = base_model.eval().to(device)

    peft_model = None
    adapter_names: dict[str, str] = {}
    if adapter_paths:
        first_name, first_path = adapter_paths[0]
        peft_model = PeftModel.from_pretrained(
            base_model,
            first_path,
            adapter_name=first_name,
            is_trainable=False,
        ).eval()
        adapter_names[first_name] = first_name
        for name, path in adapter_paths[1:]:
            peft_model.load_adapter(
                path, adapter_name=name, is_trainable=False
            )
            adapter_names[name] = name
        peft_model.eval()
    print(
        f"[fit] model={base_path} device={device} "
        f"adapters={','.join(adapter_names) or 'none'}",
        flush=True,
    )
    return tokenizer, base_model, peft_model, adapter_names


def generate(
    receiver: InferenceReceiver,
    record: dict[str, Any],
    *,
    max_length: int,
    no_repeat_ngram_size: int,
    ngram_window: int,
    temperature: float,
    single_image_size: int,
    single_crop_mode: bool,
) -> tuple[str, int | None]:
    with tempfile.TemporaryDirectory(prefix="uocr-fit-") as output_dir:
        common = {
            "max_length": max_length,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "ngram_window": ngram_window,
            "temperature": temperature,
        }
        if len(record["images"]) == 1:
            text = receiver.infer(
                receiver.tokenizer,
                prompt=record["prompt"],
                image_file=record["images"][0],
                output_path=output_dir,
                base_size=1024,
                image_size=single_image_size,
                crop_mode=single_crop_mode,
                eval_mode=True,
                **common,
            )
            return text or "", None
        text, output_tokens = receiver.infer_multi(
            receiver.tokenizer,
            prompt=record["prompt"],
            image_files=record["images"],
            output_path=output_dir,
            image_size=1024,
            save_results=False,
            **common,
        )
        return text or "", output_tokens


def main() -> int:
    args = parse_args()
    adapter_paths = parse_adapters(args.adapter)
    records = load_records(args.dataset)
    selected = choose_records(records, args.sample_line, args.sample_count)
    for record in selected:
        missing = [path for path in record["images"] if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"line {record['line']} has missing images: {missing[:3]}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = args.output_dir / "responses.jsonl"
    if responses_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{responses_path} exists; use --overwrite only for an intentional rerun"
        )
    if args.overwrite:
        responses_path.unlink(missing_ok=True)

    tokenizer, base_model, peft_model, adapter_names = load_model(
        args.base_model,
        adapter_paths,
        args.device,
        args.attn_implementation,
    )
    for record in selected:
        record["target_tokens"] = len(
            tokenizer(record["target"], add_special_tokens=False)["input_ids"]
        )

    samples_path = args.output_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    route_names = [BASE_NAME] + [name for name, _path in adapter_paths]
    markdown_dir = args.output_dir / "markdown"
    markdown_dir.mkdir(exist_ok=True)
    total = len(selected) * len(route_names)
    completed = 0
    started_all = time.monotonic()
    summary_rows = []
    with responses_path.open("a", encoding="utf-8") as response_handle:
        for record in selected:
            for route_name in route_names:
                started = time.monotonic()
                result: dict[str, Any] = {
                    "sample_line": record["line"],
                    "id": record["id"],
                    "pages": len(record["images"]),
                    "target_chars": len(record["target"]),
                    "target_tokens": record["target_tokens"],
                    "model": route_name,
                    "prompt": record["prompt"],
                    "max_length": args.max_length,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                    "ngram_window": args.ngram_window,
                    "temperature": args.temperature,
                }
                try:
                    if peft_model is None:
                        receiver_model = base_model
                        adapter_context = nullcontext()
                    elif route_name == BASE_NAME:
                        receiver_model = peft_model
                        adapter_context = peft_model.disable_adapter()
                    else:
                        peft_model.set_adapter(adapter_names[route_name])
                        receiver_model = peft_model
                        adapter_context = nullcontext()
                    receiver = InferenceReceiver(
                        receiver_model, base_model.__class__, tokenizer
                    )
                    with adapter_context:
                        with torch.inference_mode():
                            with redirect_stdout(io.StringIO()):
                                output, output_tokens = generate(
                                    receiver,
                                    record,
                                    max_length=args.max_length,
                                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                                    ngram_window=args.ngram_window,
                                    temperature=args.temperature,
                                    single_image_size=args.single_image_size,
                                    single_crop_mode=args.single_crop_mode,
                                )
                    result.update(
                        {
                            "ok": True,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "output_tokens": output_tokens,
                            "metrics": diagnostics(record["target"], output),
                            "target": record["target"],
                            "output": output,
                        }
                    )
                except Exception as exc:
                    result.update(
                        {
                            "ok": False,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "error": repr(exc),
                            "target": record["target"],
                            "output": "",
                        }
                    )
                safe_route_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", route_name)
                markdown_path = markdown_dir / (
                    f"{record['line']:04d}__{safe_route_name}.md"
                )
                write_response_markdown(
                    markdown_path, result, record["target"], result["output"]
                )
                response_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                response_handle.flush()
                completed += 1
                summary_rows.append(result)
                if result["ok"]:
                    output_chars = result["metrics"]["output_chars"]
                    status = f"output_chars={output_chars}"
                else:
                    status = f"error={result['error']}"
                print(
                    f"[fit] [{completed}/{total}] line={record['line']} "
                    f"model={route_name} ok={result['ok']} {status} "
                    f"elapsed={result['elapsed_seconds']}s",
                    flush=True,
                )

    aggregate: dict[str, dict[str, Any]] = {}
    for route_name in route_names:
        route_rows = [
            row for row in summary_rows
            if row["model"] == route_name and row["ok"]
        ]
        aggregate[route_name] = {
            "requests": sum(row["model"] == route_name for row in summary_rows),
            "successes": len(route_rows),
            "errors": sum(
                row["model"] == route_name and not row["ok"]
                for row in summary_rows
            ),
            "mean_sequence_similarity": (
                sum(row["metrics"]["sequence_similarity"] for row in route_rows)
                / len(route_rows) if route_rows else 0.0
            ),
            "mean_target_line_recall": (
                sum(row["metrics"]["target_line_recall"] for row in route_rows)
                / len(route_rows) if route_rows else 0.0
            ),
            "mean_target_heading_recall": (
                sum(row["metrics"]["target_heading_recall"] for row in route_rows)
                / len(route_rows) if route_rows else 0.0
            ),
            "mean_duplicate_line_ratio": (
                sum(row["metrics"]["duplicate_line_ratio"] for row in route_rows)
                / len(route_rows) if route_rows else 0.0
            ),
            "severe": sum(
                bool(row["metrics"]["severe"]) for row in route_rows
            ),
        }
    summary = {
        "dataset": str(args.dataset.resolve()),
        "base_model": str(args.base_model.resolve()),
        "samples": [
            {
                "line": record["line"],
                "id": record["id"],
                "pages": len(record["images"]),
                "target_chars": len(record["target"]),
                "target_tokens": record["target_tokens"],
            }
            for record in selected
        ],
        "models": route_names,
        "requests": len(summary_rows),
        "successes": sum(row["ok"] for row in summary_rows),
        "errors": sum(not row["ok"] for row in summary_rows),
        "elapsed_seconds": round(time.monotonic() - started_all, 3),
        "config": {
            "max_length": args.max_length,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "ngram_window": args.ngram_window,
            "temperature": args.temperature,
            "attn_implementation": args.attn_implementation,
            "single_image_size": args.single_image_size,
            "single_crop_mode": args.single_crop_mode,
        },
        "by_model": aggregate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
