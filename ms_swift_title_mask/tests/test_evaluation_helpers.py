import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ms_swift_title_mask.scripts.aggregate_sweep import aggregate
from ms_swift_title_mask.scripts.build_pmc_readoc_mix import main as build_mix
from ms_swift_title_mask.scripts.synthesize_combined_metric import combine


class EvaluationHelpersTest(unittest.TestCase):
    def test_aggregate_uses_requested_result_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "swp_doc_text_block_per_page_edit.json").write_text(
                json.dumps({"p1": 0.2, "p2": 0.4}), encoding="utf-8"
            )
            (root / "swp_doc_table_per_table_TEDS.json").write_text(
                json.dumps({"t1": {"TEDS": 0.6}}), encoding="utf-8"
            )
            (root / "swp_doc_display_formula_per_page_edit.json").write_text(
                json.dumps({"p1": 0.5}), encoding="utf-8"
            )

            result = aggregate("swp_*", root)

        self.assertAlmostEqual(result["text_edit"], 0.3)
        self.assertAlmostEqual(result["table_teds"], 0.6)
        self.assertAlmostEqual(result["formula_edit"], 0.5)
        self.assertEqual(result["pages"], 2)

    def test_combine_reads_page_and_table_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "swp_doc_text_block_per_page_edit.json").write_text(
                json.dumps({"p1": 0.25}), encoding="utf-8"
            )
            (root / "swp_doc_table_per_table_TEDS.json").write_text(
                json.dumps({"t1": {"TEDS": 0.7, "TEDS_structure_only": 0.8}}),
                encoding="utf-8",
            )

            combined, _ = combine(root, "swp_")

        self.assertEqual(
            combined["text_block"]["all"]["Edit_dist"]["ALL_page_avg"], 0.25
        )
        self.assertEqual(combined["table"]["all"]["TEDS"]["all"], 0.7)
        self.assertEqual(
            combined["table"]["all"]["TEDS_structure_only"]["all"], 0.8
        )

    def test_build_mix_preserves_order_and_split_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pmc = root / "pmc"
            spage = root / "spage"
            pmc.mkdir()
            spage.mkdir()
            readoc = root / "readoc.jsonl"
            out = root / "out"

            def row(row_id, source, doc_id, prompt, *, extra_meta=None):
                meta = {"source": source, "doc_id": doc_id, "split": "train"}
                if extra_meta:
                    meta.update(extra_meta)
                return {
                    "id": row_id,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": f"# {row_id}\n"},
                    ],
                    "images": [str(root / f"{row_id}.png")],
                    "meta": meta,
                }

            (pmc / "train.jsonl").write_text(
                json.dumps(row("p-full", "PMC-v2", "p-full", "<image>Multi page merge."))
                + "\n",
                encoding="utf-8",
            )
            (spage / "train.jsonl").write_text(
                json.dumps(row("p-page", "PMC-v2", "p-page", "<image>document parsing."))
                + "\n",
                encoding="utf-8",
            )
            readoc.write_text(
                json.dumps(
                    row(
                        "r-full",
                        "READoc-arxiv",
                        "r-full",
                        "<image>Multi page merge.",
                        extra_meta={"readoc_only": True},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (pmc / "validation.jsonl").write_text("{}\n", encoding="utf-8")
            (pmc / "test.jsonl").write_text("{}\n", encoding="utf-8")

            argv = [
                "build_pmc_readoc_mix.py",
                "--pmc-dir",
                str(pmc),
                "--spage-dir",
                str(spage),
                "--readoc-train",
                str(readoc),
                "--out-dir",
                str(out),
                "--skip-image-check",
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(build_mix(), 0)

            rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
            self.assertEqual([item["id"] for item in rows], ["p-full", "p-page", "r-full"])
            self.assertNotIn("readoc_only", rows[-1]["meta"])
            self.assertEqual(
                (out / "validation.jsonl").resolve(), (pmc / "validation.jsonl").resolve()
            )
            self.assertEqual((out / "test.jsonl").resolve(), (pmc / "test.jsonl").resolve())


if __name__ == "__main__":
    unittest.main()
