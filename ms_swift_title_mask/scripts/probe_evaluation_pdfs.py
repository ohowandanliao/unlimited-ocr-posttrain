#!/usr/bin/env python3
"""Historical 2026-08-25 reproduction tool for PDF inference evaluation.

Run a reproducible inference evaluation on selected or all evaluation PDFs.

It renders the named PDFs (or all PDFs with ``--all-files``), calls the local
base/full-CE/title-weighted routes, stores complete responses in JSONL, and can
write one readable Markdown file per response.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from jsonl_to_markdown import render_row, safe_name


MODEL_BASE = "unlimited-ocr-base"
MODEL_FULL_CE = "unlimited-ocr-full-ce"
MODEL_TITLE_WEIGHTED = "unlimited-ocr-title-weighted"
MODEL_CHOICES = [MODEL_BASE, MODEL_FULL_CE, MODEL_TITLE_WEIGHTED]
TRAINED_MODELS = {MODEL_FULL_CE, MODEL_TITLE_WEIGHTED}


def render_pdf(pdf_path: Path, page_root: Path, dpi: int) -> list[Path]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required: pip install PyMuPDF") from exc

    page_root.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    with fitz.open(pdf_path) as document:
        scale = dpi / 72.0
        for index, page in enumerate(document, start=1):
            target = page_root / f"page-{index:04d}.png"
            if not target.exists():
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                pixmap.save(target, output="png")
            pages.append(target)
    return pages


def call_service(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def repetition_signal(text: str) -> str:
    if not text:
        return "空输出"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4:
        duplicate_ratio = 1.0 - len(set(lines)) / len(lines)
        if duplicate_ratio >= 0.20:
            return f"重复行比例 {duplicate_ratio:.0%}"
    for size in (256, 128, 64):
        if len(text) >= size * 2 and text[-size:] == text[-size * 2 : -size]:
            return f"末尾重复块 {size} 字符"
    if "<|det|>" in text and "<|/det|>" in text:
        return "grounding 标签格式"
    return "未见明显重复"


def prompt_for(model: str, pages: int) -> str:
    if pages == 1:
        return "<image>document parsing."
    return "<image>Multi page merge." if model in TRAINED_MODELS else "<image>Multi page parsing."


def markdown_summary(
    records: list[dict[str, Any]],
    names: list[str],
    models: list[str],
    max_length: int,
    notes: list[str],
) -> str:
    expected_records = len(names) * len(models)
    lines = [
        "# evaluation 推理抽测记录",
        "",
        f"日期：{date.today().isoformat()}",
        "",
        "范围：evaluation PDF 抽测；不是全量 OmniDocBench 评分。输入为 PDF 渲染页图，输出为 Unlimited-OCR Markdown/grounding 文本。",
        "",
        "参数：单页和多页使用各自训练约定的 prompt；`max_length=32768`；`no_repeat_ngram_size=35`；单页 `ngram_window=128`，多页 `ngram_window=1024`；temperature=0。",
        "",
        f"完成记录：{len(records)}/{expected_records}；未完成请求不会被当作通过。",
        "",
        "## 结果",
        "",
        "| evaluation 文件 | 页数 | 模型 | prompt | 输出字符 | 输出 token | 重复/格式信号 | GT | 状态 |",
        "|---|---:|---|---|---:|---:|---|---:|---|",
    ]
    for record in records:
        response = record.get("response") or {}
        output_tokens = response.get("output_tokens")
        output_tokens_text = "-" if output_tokens is None else str(output_tokens)
        status = record.get("error") or "完成"
        lines.append(
            "| {name} | {pages} | {model} | `{prompt}` | {chars} | {tokens} | {signal} | {gt} | {status} |".format(
                name=record["file"],
                pages=record["pages"],
                model=record["model"],
                prompt=record["prompt"],
                chars=len(response.get("text") or ""),
                tokens=output_tokens_text,
                signal=record.get("signal", "-"),
                gt=record.get("gt_chars", 0),
                status=status.replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 完整输入和输出保存在同目录的 `responses.jsonl`，每条记录只保留 evaluation 文件名，不依赖服务端临时目录。",
            "- 这次记录用于确认输入、prompt、输出格式和截断/重复情况；没有把字符数当作准确率，也没有替代正式 OmniDocBench 评分。",
            "- 单页 prompt 没有加入 page；多页训练后模型使用 `Multi page merge.`。",
            "- 若出现 `grounding 标签格式`，说明模型输出仍是原 Unlimited-OCR 的 grounding 表达，后续正式评测需要先确认是否沿用该输出协议。",
            "",
            f"本次指定文件数：{len(names)}；模型路由数：{len(models)}；单条最大生成长度：{max_length}。",
        ]
    )
    if notes:
        lines.extend(["", "## 备注", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", action="append", default=[], help="evaluation PDF filename, repeatable")
    parser.add_argument("--all-files", action="store_true", help="evaluate every PDF directly under --pdf-root")
    parser.add_argument("--model", action="append", choices=MODEL_CHOICES)
    parser.add_argument("--url", default="http://127.0.0.1:18080/infer")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--write-markdown", action="store_true", help="write one Markdown file per response")
    args = parser.parse_args()

    if args.all_files:
        names = sorted(path.name for path in args.pdf_root.glob("*.pdf"))
    elif args.name:
        names = args.name
    else:
        parser.error("provide --name at least once or use --all-files")
    if not names:
        parser.error(f"no PDF files found under {args.pdf_root}")

    models = args.model or MODEL_CHOICES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    page_root = args.output_dir / "pages"
    records: list[dict[str, Any]] = []
    responses_path = args.output_dir / "responses.jsonl"

    if args.summary_only:
        if responses_path.is_file():
            records = [
                json.loads(line)
                for line in responses_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        (args.output_dir / "summary.md").write_text(
            markdown_summary(records, names, models, args.max_length, args.note),
            encoding="utf-8",
        )
        return 0

    (args.output_dir / "evaluation_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    with responses_path.open("w", encoding="utf-8") as handle:
        for name in names:
            pdf_path = args.pdf_root / name
            if not pdf_path.is_file():
                raise FileNotFoundError(f"evaluation PDF not found: {name}")
            pages = render_pdf(pdf_path, page_root / pdf_path.stem, args.dpi)
            gt_path = args.gt_root / f"{pdf_path.stem}.md"
            gt_chars = len(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else 0
            for model in models:
                prompt = prompt_for(model, len(pages))
                payload = {
                    "model": model,
                    "image_paths": [str(path.resolve()) for path in pages],
                    "prompt": prompt,
                    "max_length": args.max_length,
                    "no_repeat_ngram_size": 35,
                    "ngram_window": 128 if len(pages) == 1 else 1024,
                    "temperature": 0.0,
                }
                record: dict[str, Any] = {
                    "file": pdf_path.name,
                    "pages": len(pages),
                    "model": model,
                    "prompt": prompt,
                    "gt_chars": gt_chars,
                }
                try:
                    response = call_service(args.url, payload, args.timeout)
                    text = response.get("text") or ""
                    record["response"] = {
                        "model": response.get("model"),
                        "prompt": response.get("prompt"),
                        "num_images": response.get("num_images"),
                        "output_tokens": response.get("output_tokens"),
                        "text": text,
                    }
                    record["signal"] = repetition_signal(text)
                except Exception as exc:
                    record["error"] = str(exc)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if args.write_markdown:
                    target = args.output_dir / f"{safe_name(name)}__{safe_name(model)}.md"
                    target.write_text(render_row(record), encoding="utf-8")
                (args.output_dir / "summary.md").write_text(
                    markdown_summary(records, names, models, args.max_length, args.note),
                    encoding="utf-8",
                )
                print(
                    f"{pdf_path.name}\t{len(pages)} pages\t{model}\t"
                    f"{('error: ' + record['error']) if 'error' in record else record['signal']}",
                    flush=True,
                )
    (args.output_dir / "summary.md").write_text(
        markdown_summary(records, names, models, args.max_length, args.note), encoding="utf-8"
    )
    return 0 if all("error" not in record for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
