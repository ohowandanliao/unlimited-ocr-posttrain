#!/usr/bin/env python3
"""Synthesize an official-style combined OmniDocBench metric_result.json from
per-document sweep outputs, plus concatenated *_result.json sample files for
AgentBuilder per-doc breakdowns.

Usage:
    python3 synthesize_combined_metric.py --result-dir <dir> <prefix> <out_dir|-> [save_name]

  <prefix>     glob prefix of per-doc artifacts in OMNI result dir,
               e.g. "swp_" (sweep) or "readoc16k_20260827_full_ce_gtpdf_" (old full run)
  <out_dir>    "-" = validate only: recompute the 6 AgentBuilder-consumed fields
               and diff them against <prefix>metric_result.json (requires the
               prefix to be a full-run with an existing metric_result.json)
  [save_name]  combined file stem, default "combined"
"""
import argparse
import json
import os
from pathlib import Path

MODULES = ("text_block", "table", "display_formula", "reading_order")


def load_all(result_dir: Path, pattern: str):
    out = []
    for path in sorted(result_dir.glob(pattern)):
        with path.open(encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def page_values(result_dir: Path, prefix: str, module: str):
    vals = []
    for d in load_all(result_dir, f"{prefix}*{module}_per_page_edit.json"):
        vals += list(d.values())
    return vals


def sample_list(result_dir: Path, prefix: str, module: str):
    samples = []
    for d in load_all(result_dir, f"{prefix}*{module}_result.json"):
        if isinstance(d, list):
            samples += d
    return samples


def mean(vs):
    vs = [v for v in vs if isinstance(v, (int, float))]
    return sum(vs) / len(vs) if vs else None


def edit_block(result_dir: Path, prefix: str, module: str):
    page_mean = mean(page_values(result_dir, prefix, module))
    ss = [s for s in sample_list(result_dir, prefix, module) if isinstance(s, dict)]
    num = [s["Edit_num"] for s in ss if "Edit_num" in s and "upper_len" in s]
    den = [s["upper_len"] for s in ss if "Edit_num" in s and "upper_len" in s]
    block = {"ALL_page_avg": page_mean if page_mean is not None else float("nan")}
    if num:
        block["edit_whole"] = sum(num) / sum(den)
        block["edit_sample_avg"] = sum(n / d for n, d in zip(num, den)) / len(num)
    return block, ss


def teds_block(result_dir: Path, prefix: str):
    tables = []
    for d in load_all(result_dir, f"{prefix}*table_per_table_TEDS.json"):
        tables += list(d.values())
    teds = mean([t["TEDS"] for t in tables if isinstance(t, dict) and "TEDS" in t])
    teds_s = mean(
        [t["TEDS_structure_only"] for t in tables if isinstance(t, dict) and "TEDS_structure_only" in t]
    )
    return teds, teds_s


def combine(result_dir: Path, prefix: str):
    combined, concat = {}, {}
    for module in MODULES:
        block, ss = edit_block(result_dir, prefix, module)
        concat[module] = ss
        combined[module] = {"all": {"Edit_dist": block}, "group": {}, "page": {}}
    teds, teds_s = teds_block(result_dir, prefix)
    combined["table"]["all"]["TEDS"] = {"all": teds if teds is not None else float("nan")}
    combined["table"]["all"]["TEDS_structure_only"] = {
        "all": teds_s if teds_s is not None else float("nan")
    }
    return combined, concat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix")
    parser.add_argument("out_dir")
    parser.add_argument("save_name", nargs="?", default="combined")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=os.environ.get(
            "OMNIDOCBENCH_RESULT_DIR",
            "/home/jovyan/hyx/pdf2text-auto-label-eval/OmniDocBench_v1.5/result",
        ),
        help="OmniDocBench result directory (or set OMNIDOCBENCH_RESULT_DIR)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    combined, concat = combine(args.result_dir, args.prefix)

    if args.out_dir == "-":
        ref_path = args.result_dir / f"{args.prefix}metric_result.json"
        with ref_path.open(encoding="utf-8") as f:
            ref = json.load(f)
        checks = [
            ("text_block.Edit.ALL_page_avg",
             combined["text_block"]["all"]["Edit_dist"]["ALL_page_avg"],
             ref["text_block"]["all"]["Edit_dist"]["ALL_page_avg"]),
            ("table.Edit.ALL_page_avg",
             combined["table"]["all"]["Edit_dist"]["ALL_page_avg"],
             ref["table"]["all"]["Edit_dist"]["ALL_page_avg"]),
            ("table.TEDS.all", combined["table"]["all"]["TEDS"]["all"],
             ref["table"]["all"]["TEDS"]["all"]),
            ("table.TEDS_S.all", combined["table"]["all"]["TEDS_structure_only"]["all"],
             ref["table"]["all"]["TEDS_structure_only"]["all"]),
            ("display_formula.Edit.ALL_page_avg",
             combined["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"],
             ref["display_formula"]["all"]["Edit_dist"]["ALL_page_avg"]),
            ("reading_order.Edit.ALL_page_avg",
             combined["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"],
             ref["reading_order"]["all"]["Edit_dist"]["ALL_page_avg"]),
        ]
        ok = True
        for name, got, want in checks:
            same = got == want
            ok &= same
            print(f"{'OK ' if same else 'DIFF'} {name}: got={got!r} want={want!r}")
        print("VALIDATION", "PASS" if ok else "FAIL")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.save_name}_metric_result.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for module in MODULES:
        (out / f"{args.save_name}_{module}_result.json").write_text(
            json.dumps(concat[module], ensure_ascii=False) + "\n", encoding="utf-8"
        )
        n = len(concat[module])
        print(f"{args.save_name}_{module}_result.json: {n} samples")
    print(f"wrote combined metric_result to {out}/{args.save_name}_metric_result.json")


if __name__ == "__main__":
    main()
