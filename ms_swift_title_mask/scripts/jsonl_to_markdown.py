#!/usr/bin/env python3
"""Historical 2026-08-25 reproduction tool for rendering inference JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def safe_name(value: str) -> str:
    value = re.sub(r"\.(pdf|jsonl)$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[\\/:*?\"<>|]", "_", value)
    value = re.sub(r"\s+", "_", value).strip("._")
    return value or "output"


def fence_for(text: str) -> str:
    runs = [len(match.group(0)) for match in re.finditer(r"`+", text)]
    return "`" * max(3, (max(runs) + 1) if runs else 3)


def render_row(row: dict) -> str:
    response = row.get("response") or {}
    text = response.get("text") or ""
    output_tokens = response.get("output_tokens")
    output_tokens = "-" if output_tokens is None else str(output_tokens)
    status = row.get("error") or "完成"
    fence = fence_for(text)
    return "\n".join(
        [
            f"# 推理输出：{row.get('file', 'unknown')}",
            "",
            f"- evaluation 文件：`{row.get('file', '-')}`",
            f"- 模型：`{row.get('model', '-')}`",
            f"- 页数：{row.get('pages', '-')}",
            f"- prompt：`{row.get('prompt', '-')}`",
            f"- GT 字符数：{row.get('gt_chars', '-')}",
            f"- 输出字符数：{len(text)}",
            f"- 输出 token：{output_tokens}",
            f"- 重复/格式信号：{row.get('signal', '-')}",
            f"- 状态：{status}",
            "",
            "## 完整输出",
            "",
            fence + "text",
            text,
            fence,
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = (args.output_dir or args.jsonl.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in args.jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    written = []
    for index, row in enumerate(rows, start=1):
        file_name = safe_name(str(row.get("file", f"row-{index}")))
        model = safe_name(str(row.get("model", "model")))
        target = output_dir / f"{file_name}__{model}.md"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {target}; pass --overwrite")
        target.write_text(render_row(row), encoding="utf-8")
        written.append(target)
    print(f"wrote {len(written)} markdown files to {output_dir}")
    for target in written:
        print(target.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
