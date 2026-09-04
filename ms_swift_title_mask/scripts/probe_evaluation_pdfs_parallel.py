#!/usr/bin/env python3
"""Official parallel PDF evaluation runner for the post-training evaluation protocol.

Evaluate PDF/model jobs with one serialized worker per inference endpoint.

The service keeps adapter selection and generation state in one process, so
this runner uses a dynamic queue across independent service endpoints instead
of sending concurrent requests to the same endpoint. Rendered page images can
be shared across evaluation runs through ``--page-cache-dir``.

Resuming is safe by default: existing ``responses.jsonl`` rows are only kept
when their prompt matches the current prompt contract (manifest or default)
and their recorded generation parameters match this invocation. Rows without a
recorded ``gen`` block are re-run unless ``--trust-legacy-rows`` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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

    with fitz.open(pdf_path) as document:
        scale = dpi / 72.0
        pages = [page_root / f"page-{index:04d}.png" for index in range(1, document.page_count + 1)]
        existing_pages = {path for path in page_root.glob("page-*.png") if path.is_file()} if page_root.exists() else set()
        if existing_pages == set(pages):
            return pages

        page_root.mkdir(parents=True, exist_ok=True)
        for target in page_root.glob("page-*.png"):
            target.unlink()
        for index, page in enumerate(document, start=1):
            target = pages[index - 1]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pixmap.save(target, output="png")
    return pages


def page_cache_dir(cache_root: Path, pdf_path: Path, dpi: int) -> Path:
    """Content-addressed cache directory for one PDF at one render DPI."""
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return cache_root / f"dpi-{dpi}" / digest


def pdf_page_count(pdf_path: Path) -> int:
    """Page count without rendering, used to validate resumed rows cheaply."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required: pip install PyMuPDF") from exc
    with fitz.open(pdf_path) as document:
        return document.page_count


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
        "# evaluation 推理记录",
        "",
        f"日期：{date.today().isoformat()}",
        "",
        "范围：evaluation PDF 推理记录（--name 指定或 --all-files 全量）；本文件不是 OmniDocBench 评分本身。"
        "输入为 PDF 渲染页图，输出为 Unlimited-OCR Markdown/grounding 文本。",
        "",
        f"参数：单页和多页使用各自训练约定的 prompt；`max_length={max_length}`；`no_repeat_ngram_size=35`；"
        "单页 `ngram_window=128`，多页 `ngram_window=1024`；temperature=0。",
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
            "",
            f"本次指定文件数：{len(names)}；模型路由数：{len(models)}；单条最大生成长度：{max_length}。",
        ]
    )
    if notes:
        lines.extend(["", "## 备注", ""])
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class Job:
    file_name: str
    model: str
    pages: tuple[Path, ...]
    gt_chars: int
    prompt: str | None = None


def normalize_url(raw_url: str) -> str:
    parts = urlsplit(raw_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"invalid service URL: {raw_url!r}")
    path = parts.path.rstrip("/")
    if not path:
        path = "/infer"
    elif path != "/infer":
        raise ValueError(f"service URL must end at /infer: {raw_url!r}")
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def load_existing(path: Path) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"ignoring malformed existing row {line_number}: {exc}", file=sys.stderr)
            continue
        key = (row.get("file"), row.get("model"))
        if key[0] and key[1]:
            records[key] = row
    return records


def load_prompt_manifest(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid prompt manifest JSON at line {line_number}: {exc}") from exc
        file_name = row.get("file") if isinstance(row, dict) else None
        prompt = row.get("prompt") if isinstance(row, dict) else None
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"prompt manifest line {line_number} has no file name")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt manifest line {line_number} has an empty prompt")
        previous = prompts.get(file_name)
        if previous is not None and previous != prompt:
            raise ValueError(f"prompt manifest has conflicting prompts for {file_name!r}")
        prompts[file_name] = prompt
    if not prompts:
        raise ValueError(f"prompt manifest is empty: {path}")
    return prompts


def expected_prompt(name: str, model: str, page_count: int, prompt_by_file: dict[str, str]) -> str:
    """Prompt contract for a job; the manifest wins for trained models."""
    if prompt_by_file and model in TRAINED_MODELS:
        return prompt_by_file[name]
    return prompt_for(model, page_count)


def row_is_compatible(
    row: dict, expected: str, args: argparse.Namespace, page_count: int
) -> bool:
    """A resumed row may only be reused when prompt and generation config match."""
    try:
        if "error" in row or row.get("prompt") != expected:
            return False
        if int(row.get("pages")) != page_count:
            return False
        response = row.get("response")
        if not isinstance(response, dict) or int(response.get("num_images")) != page_count:
            return False
        if response.get("prompt") != expected:
            return False
        gen = row.get("gen")
        if gen is None:
            return bool(args.trust_legacy_rows)
        if not isinstance(gen, dict):
            return False
        return (
            int(gen.get("max_length")) == int(args.max_length)
            and int(gen.get("dpi")) == int(args.dpi)
            and int(gen.get("no_repeat_ngram_size")) == 35
            and int(gen.get("ngram_window")) == (128 if page_count == 1 else 1024)
            and float(gen.get("temperature")) == 0.0
        )
    except (AttributeError, TypeError, ValueError):
        return False


def ordered_records(
    records: dict[tuple[str, str], dict], names: list[str], models: list[str]
) -> list[dict]:
    return [
        records[(name, model)]
        for name in names
        for model in models
        if (name, model) in records
    ]


def write_outputs(
    output_dir: Path,
    records: dict[tuple[str, str], dict],
    names: list[str],
    models: list[str],
    max_length: int,
    notes: list[str],
) -> None:
    rows = ordered_records(records, names, models)
    responses_path = output_dir / "responses.jsonl"
    temporary_path = output_dir / "responses.jsonl.tmp"
    temporary_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary_path.replace(responses_path)

    for row in rows:
        target = output_dir / f"{safe_name(str(row.get('file', 'unknown')))}__{safe_name(str(row.get('model', 'model')))}.md"
        target.write_text(render_row(row), encoding="utf-8")

    expected = len(names) * len(models)
    failed = sum("error" in row for row in rows)
    successful = sum("error" not in row for row in rows)
    summary = markdown_summary(rows, names, models, max_length, notes)
    summary = summary.replace(
        f"完成记录：{len(rows)}/{expected}；未完成请求不会被当作通过。",
        f"完成记录：{successful}/{expected}；记录 {len(rows)} 条；错误 {failed} 条。未完成请求不会被当作通过。",
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def make_error_row(job: Job, error: Exception) -> dict:
    prompt = job.prompt or prompt_for(job.model, len(job.pages))
    return {
        "file": job.file_name,
        "pages": len(job.pages),
        "model": job.model,
        "prompt": prompt,
        "gt_chars": job.gt_chars,
        "error": f"{type(error).__name__}: {error}",
    }


def run_job(
    job: Job,
    url: str,
    max_length: int,
    dpi: int,
    timeout: int,
    attempts: int,
    retry_delay: float,
) -> dict:
    prompt = job.prompt or prompt_for(job.model, len(job.pages))
    payload = {
        "model": job.model,
        "image_paths": [str(path.resolve()) for path in job.pages],
        "prompt": prompt,
        "max_length": max_length,
        "no_repeat_ngram_size": 35,
        "ngram_window": 128 if len(job.pages) == 1 else 1024,
        "temperature": 0.0,
    }
    base_row = {
        "file": job.file_name,
        "pages": len(job.pages),
        "model": job.model,
        "prompt": prompt,
        "gt_chars": job.gt_chars,
        "gen": {
            "max_length": max_length,
            "dpi": dpi,
            "no_repeat_ngram_size": 35,
            "ngram_window": 128 if len(job.pages) == 1 else 1024,
            "temperature": 0.0,
        },
    }
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = call_service(url, payload, timeout)
            text = response.get("text") or ""
            base_row["response"] = {
                "model": response.get("model"),
                "prompt": response.get("prompt"),
                "num_images": response.get("num_images"),
                "output_tokens": response.get("output_tokens"),
                "text": text,
            }
            base_row["signal"] = repetition_signal(text)
            return base_row
        except Exception as exc:  # keep one failed request from stopping all jobs
            last_error = exc
            if attempt < max(1, attempts):
                time.sleep(retry_delay * attempt)
    assert last_error is not None
    base_row["error"] = f"{type(last_error).__name__}: {last_error}"
    return base_row


def worker(
    endpoint: str,
    jobs: queue.Queue[Job],
    results: queue.Queue[tuple[str, dict]],
    max_length: int,
    dpi: int,
    timeout: int,
    attempts: int,
    retry_delay: float,
) -> None:
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            try:
                row = run_job(job, endpoint, max_length, dpi, timeout, attempts, retry_delay)
            except Exception as exc:
                row = make_error_row(job, exc)
            results.put((endpoint, row))
        finally:
            jobs.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--all-files", action="store_true")
    parser.add_argument("--model", action="append", choices=MODEL_CHOICES)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        help="JSONL mapping evaluation PDF filenames to per-file prompts",
    )
    parser.add_argument("--url", action="append", required=True, help="one independent /infer endpoint per worker")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument(
        "--page-cache-dir",
        type=Path,
        help="reusable root for rendered PDF pages; cache entries are grouped by DPI",
    )
    parser.add_argument("--max-length", type=int, default=20480)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument(
        "--trust-legacy-rows",
        action="store_true",
        help="accept resumed rows that lack the recorded gen block; default re-runs them",
    )
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
    prompt_by_file = load_prompt_manifest(args.prompt_manifest) if args.prompt_manifest else {}
    if prompt_by_file:
        missing_prompts = sorted(set(names) - set(prompt_by_file))
        extra_prompts = sorted(set(prompt_by_file) - set(names))
        if missing_prompts or extra_prompts:
            raise ValueError(
                "prompt manifest does not match evaluation files; "
                f"missing={missing_prompts[:5]} extra={extra_prompts[:5]}"
            )
    endpoints = [normalize_url(url) for url in args.url]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    responses_path = args.output_dir / "responses.jsonl"
    records = load_existing(responses_path)
    expected_keys = {(name, model) for name in names for model in models}
    records = {key: row for key, row in records.items() if key in expected_keys}

    # Resume validation: reuse an existing row only when its prompt matches the
    # current contract and its recorded generation config matches this run.
    kept_records: dict[tuple[str, str], dict] = {}
    requeued = 0
    if records:
        page_counts = {name: pdf_page_count(args.pdf_root / name) for name in names}
        for key, row in records.items():
            expected = expected_prompt(key[0], key[1], page_counts[key[0]], prompt_by_file)
            if row_is_compatible(row, expected, args, page_counts[key[0]]):
                kept_records[key] = row
            else:
                requeued += 1
        if requeued:
            print(f"re-queuing {requeued} existing rows: prompt or generation config mismatch", flush=True)
    records = kept_records
    completed = set(records)
    pending_keys = [key for key in expected_keys if key not in completed]

    pages_by_name: dict[str, tuple[Path, ...]] = {}
    gt_chars_by_name: dict[str, int] = {}
    pending_files = {name for name, _ in pending_keys}
    for name in names:
        pdf_path = args.pdf_root / name
        if not pdf_path.is_file():
            raise FileNotFoundError(f"evaluation PDF not found: {name}")
        gt_path = args.gt_root / f"{pdf_path.stem}.md"
        gt_chars_by_name[name] = len(gt_path.read_text(encoding="utf-8")) if gt_path.is_file() else 0
        if name in pending_files:
            cache_root = args.page_cache_dir or (args.output_dir / "pages")
            page_root = page_cache_dir(cache_root, pdf_path, args.dpi)
            pages = render_pdf(pdf_path, page_root, args.dpi)
            pages_by_name[name] = tuple(pages)

    notes = list(args.note)
    if args.prompt_manifest:
        notes.append(
            f"逐文件 prompt manifest：{args.prompt_manifest}；trained model 的单页和多页请求均使用对应 prompt；无标题文件使用 manifest 中的标准 page prompt。"
        )
    notes.append(
        f"动态任务队列：{len(endpoints)} 个独立 /infer endpoint，每个 endpoint 同时只处理一个请求；已恢复 {len(completed)} 条，待处理 {len(pending_keys)} 条。"
    )
    if requeued:
        notes.append(f"续跑校验：{requeued} 条旧记录因 prompt 或生成参数不一致被重跑。")
    write_outputs(args.output_dir, records, names, models, args.max_length, notes)
    status_path = args.output_dir / "parallel_status.txt"
    status_path.write_text(
        f"status=running\nendpoints={len(endpoints)}\ncompleted={len(completed)}\npending={len(pending_keys)}\n",
        encoding="utf-8",
    )

    jobs: queue.Queue[Job] = queue.Queue()
    model_order = {model: index for index, model in enumerate(models)}
    job_objects = [
        Job(
            name,
            model,
            pages_by_name[name],
            gt_chars_by_name[name],
            prompt_by_file.get(name)
            if prompt_by_file and model in TRAINED_MODELS
            else None,
        )
        for name, model in pending_keys
    ]
    job_objects.sort(key=lambda job: (-len(job.pages), job.file_name, model_order[job.model]))
    for job in job_objects:
        jobs.put(job)
    results: queue.Queue[tuple[str, dict]] = queue.Queue()

    print(
        f"endpoints={len(endpoints)} names={len(names)} models={len(models)} "
        f"existing_success={len(completed)} pending={len(job_objects)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = [
            executor.submit(
                worker,
                endpoint,
                jobs,
                results,
                args.max_length,
                args.dpi,
                args.timeout,
                args.attempts,
                args.retry_delay,
            )
            for endpoint in endpoints
        ]
        remaining = len(job_objects)
        while remaining:
            endpoint, row = results.get()
            key = (row["file"], row["model"])
            records[key] = row
            remaining -= 1
            write_outputs(args.output_dir, records, names, models, args.max_length, notes)
            status_path.write_text(
                f"status=running\nendpoints={len(endpoints)}\ncompleted={sum('error' not in value for value in records.values())}\n"
                f"pending={remaining}\nlast_endpoint={endpoint}\nlast_file={row['file']}\nlast_model={row['model']}\n",
                encoding="utf-8",
            )
            print(
                f"[{len(job_objects) - remaining}/{len(job_objects)}] {row['file']}\t{row['model']}\t"
                f"{('error: ' + row['error']) if 'error' in row else row.get('signal', 'done')}",
                flush=True,
            )
        for future in futures:
            future.result()

    failed = sum("error" in row for row in records.values())
    successful = sum("error" not in row for row in records.values())
    final_status = "complete" if successful == len(expected_keys) and failed == 0 else "failed"
    status_path.write_text(
        f"status={final_status}\nendpoints={len(endpoints)}\nexpected={len(expected_keys)}\n"
        f"successful={successful}\nfailed={failed}\n",
        encoding="utf-8",
    )
    print(f"successful={successful} failed={failed} expected={len(expected_keys)}", flush=True)
    return 0 if final_status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
