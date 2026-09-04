#!/usr/bin/env python3
"""Complete samples.jsonl metadata and render readable Train Fit responses."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", nargs="+", type=Path)
    return parser.parse_args()


def render(result_dir: Path) -> tuple[int, int]:
    response_path = result_dir / "responses.jsonl"
    sample_path = result_dir / "samples.jsonl"
    responses = [
        json.loads(line)
        for line in response_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target_tokens_by_line = {}
    for response in responses:
        target_tokens_by_line.setdefault(
            response["sample_line"], response["target_tokens"]
        )

    samples = [
        json.loads(line)
        for line in sample_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for sample in samples:
        sample["target_tokens"] = target_tokens_by_line[sample["line"]]
    sample_path.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )

    markdown_dir = result_dir / "markdown"
    markdown_dir.mkdir(exist_ok=True)
    for response in responses:
        route = re.sub(r"[^A-Za-z0-9_.-]+", "_", response["model"])
        metrics = response.get("metrics", {})
        lines = [
            f"# Train Fit: line {response['sample_line']} / {response['model']}",
            "",
            f"- id: `{response['id']}`",
            f"- pages: {response['pages']}",
            f"- target chars: {response['target_chars']}",
            f"- target tokens: {response['target_tokens']}",
            f"- prompt: `{response['prompt']}`",
            f"- ok: `{response['ok']}`",
        ]
        if response["ok"]:
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
            lines.append(f"- error: `{response.get('error', '')}`")
        lines.extend(
            [
                "",
                "## Target",
                "",
                "~~~text",
                response.get("target", ""),
                "~~~",
                "",
                "## Output",
                "",
                "~~~text",
                response.get("output", ""),
                "~~~",
                "",
            ]
        )
        (markdown_dir / f"{response['sample_line']:04d}__{route}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
    return len(responses), len(list(markdown_dir.glob("*.md")))


def main() -> int:
    args = parse_args()
    for result_dir in args.result_dir:
        responses, markdown = render(result_dir)
        print(f"{result_dir}: responses={responses} markdown={markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
