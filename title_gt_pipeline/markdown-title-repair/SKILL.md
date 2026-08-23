---
name: markdown-title-repair
description: Repair heading recognition and ATX levels in existing Markdown through a configurable OpenAI-compatible LLM API. Use when Markdown already exists and the requested output is improved Markdown; do not use for PDF, OCR, middle.json conversion, or recovery of text absent from the Markdown.
---

# Markdown Title Repair

Use the bundled deterministic script to turn existing Markdown into a copy with better
heading markers and levels. The LLM returns line-level decisions; the script applies only
ATX `#` marker edits, so it never lets the model rewrite body content.

## Run

1. Read [references/usage.md](references/usage.md) when configuring or executing the
   pipeline.
2. Use `scripts/repair_titles.py` with a `.md` file or a directory of `.md` files.
3. Keep credentials in environment variables. Start from
   `references/config.example.json` and customize `references/review_rules.md` only when
   the document domain needs different heading rules.
4. Inspect `summary.json` and `results.jsonl`. Treat any failed or unresolved document as
   not produced.

## Invariants

- Input and output paths must differ, and the tool must not overwrite existing output.
- Only promote a line to heading, demote an existing heading, or change levels 1-6.
- Do not merge, split, rewrite, reorder, or invent text.
- Do not edit fenced code. Do not classify page furniture, captions, UI text, or TOC
  dotted entries as body headings without domain-specific evidence.
- Text missing from the Markdown cannot be recovered by this pipeline; that requires an
  upstream PDF/OCR workflow outside this skill.
- LLM output is an `LLM_CANDIDATE` for title GT or training data. It becomes accepted
  data only after the caller's source/target SHA, PDF evidence, split, provenance, and
  review checks; this skill must not label its own output `SILVER_ACCEPTED` or gold.
