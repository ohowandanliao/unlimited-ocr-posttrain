#!/usr/bin/env python3
"""Historical 2026-08-25 reproduction tool for parallel PDF evaluation.

Evaluate PDF/model jobs with one serialized worker per inference endpoint.

The service keeps adapter selection and generation state in one process, so
this runner uses a dynamic queue across independent service endpoints instead
of sending concurrent requests to the same endpoint.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from jsonl_to_markdown import render_row, safe_name
from probe_evaluation_pdfs import (
    MODEL_CHOICES,
    MODEL_BASE,
    MODEL_FULL_CE,
    MODEL_TITLE_WEIGHTED,
    call_service,
    markdown_summary,
    prompt_for,
    render_pdf,
    repetition_signal,
)


@dataclass(frozen=True)
class Job:
    file_name: str
    model: str
    pages: tuple[Path, ...]
    gt_chars: int


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
    prompt = prompt_for(job.model, len(job.pages))
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
    timeout: int,
    attempts: int,
    retry_delay: float,
) -> dict:
    prompt = prompt_for(job.model, len(job.pages))
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
                row = run_job(job, endpoint, max_length, timeout, attempts, retry_delay)
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
    parser.add_argument("--url", action="append", required=True, help="one independent /infer endpoint per worker")
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--note", action="append", default=[])
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
    endpoints = [normalize_url(url) for url in args.url]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    responses_path = args.output_dir / "responses.jsonl"
    records = load_existing(responses_path)
    expected_keys = {(name, model) for name in names for model in models}
    records = {key: row for key, row in records.items() if key in expected_keys}
    completed = {key for key, row in records.items() if "error" not in row}
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
            pages = render_pdf(pdf_path, args.output_dir / "pages" / pdf_path.stem, args.dpi)
            pages_by_name[name] = tuple(pages)

    notes = list(args.note)
    notes.append(
        f"动态任务队列：{len(endpoints)} 个独立 /infer endpoint，每个 endpoint 同时只处理一个请求；已恢复 {len(completed)} 条，待处理 {len(pending_keys)} 条。"
    )
    write_outputs(args.output_dir, records, names, models, args.max_length, notes)
    status_path = args.output_dir / "parallel_status.txt"
    status_path.write_text(
        f"status=running\nendpoints={len(endpoints)}\ncompleted={len(completed)}\npending={len(pending_keys)}\n",
        encoding="utf-8",
    )

    jobs: queue.Queue[Job] = queue.Queue()
    model_order = {model: index for index, model in enumerate(models)}
    job_objects = [
        Job(name, model, pages_by_name[name], gt_chars_by_name[name])
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
