#!/usr/bin/env python3
"""Aggregate per-doc OmniDocBench sweep results under a glob prefix.

Usage:
    python3 aggregate_sweep.py --result-dir <dir> <prefix>

  <prefix>  glob prefix of per-doc artifacts in OMNI result dir,
            e.g. "swp3_*" (wildcard REQUIRED, same convention as the
            proven pmc-fullce sweep aggregator).

Prints pooled text_edit / table_TEDS / formula_edit / total(0-100) JSON.
Parametrized copy of evaluation/pmc-fullce-16k_all129/sweep/aggregate_sweep.py.
"""
import argparse
import json
import math
import os
from pathlib import Path


def aggregate(prefix: str, result_dir: Path) -> dict:
    texts, teds, formulas = [], [], []
    for path in result_dir.glob(f"{prefix}_text_block_per_page_edit.json"):
        with path.open(encoding="utf-8") as handle:
            texts += list(json.load(handle).values())
    for path in result_dir.glob(f"{prefix}_table_per_table_TEDS.json"):
        with path.open(encoding="utf-8") as handle:
            teds += [value["TEDS"] for value in json.load(handle).values()]
    for path in result_dir.glob(f"{prefix}_display_formula_per_page_edit.json"):
        with path.open(encoding="utf-8") as handle:
            formulas += list(json.load(handle).values())
    text = sum(texts) / len(texts) if texts else math.nan
    t = sum(teds) / len(teds) if teds else math.nan
    f = sum(formulas) / len(formulas) if formulas else math.nan
    total = None
    if all(not math.isnan(x) for x in (text, t, f)):
        total = ((1 - text) * 100 + t * 100 + (1 - f) * 100) / 3
    return {"pages": len(texts), "tables": len(teds), "formula_pages": len(formulas),
            "text_edit": text, "table_teds": t, "formula_edit": f, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="glob prefix, including any required wildcard")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=os.environ.get(
            "OMNIDOCBENCH_RESULT_DIR",
            "/home/jovyan/hyx/pdf2text-auto-label-eval/OmniDocBench_v1.5/result",
        ),
        help="OmniDocBench result directory (or set OMNIDOCBENCH_RESULT_DIR)",
    )
    args = parser.parse_args()
    print(json.dumps(aggregate(args.prefix, args.result_dir), indent=1))


if __name__ == "__main__":
    main()
