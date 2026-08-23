import unittest
from collections import Counter

from ms_swift_title_mask.core import SILVER_ACCEPTED, canonicalize_target, sha256_text
from ms_swift_title_mask.pmc_silver import stable_split
from ms_swift_title_mask.scripts.build_pmc_silver_jsonl import _row
from ms_swift_title_mask.scripts.build_reviewed_mix import audit_global_identity, build_split


class SilverBoundaryTest(unittest.TestCase):
    def test_full_and_single_use_the_same_document_split(self):
        split = stable_split("seed", "PMC-v26-synthetic", "PMC123")
        pools = {}
        for pool, prefix in (("pmc_full", "full"), ("pmc_single", "single"), ("readoc_full", "readoc")):
            pools[pool] = {}
            for current in ("train", "validation", "test"):
                doc = "PMC123" if pool != "readoc_full" and current == split else (
                    "PMC-" + current if pool != "readoc_full" else prefix + "-" + current
                )
                source = "PMC-v26-synthetic" if pool != "readoc_full" else "READoc"
                family = "pmc" if pool != "readoc_full" else "readoc"
                pools[pool][current] = [(self._entry(prefix + "-" + current, current), {}, (family, source, doc))]
        audit_global_identity(pools)

    def test_cross_split_same_pmc_document_is_rejected(self):
        pools = {}
        for pool, prefix in (("pmc_full", "full"), ("pmc_single", "single"), ("readoc_full", "readoc")):
            pools[pool] = {
                s: [(self._entry(prefix + "-" + s, s), {}, ("readoc" if pool == "readoc_full" else "pmc", "SRC", prefix + "-" + s))]
                for s in ("train", "validation", "test")
            }
        pools["pmc_single"]["train"][0] = (
            self._entry("single-leak", "train"), {}, ("pmc", "SRC", "full-validation")
        )
        with self.assertRaisesRegex(Exception, "document leakage"):
            audit_global_identity(pools)

    def test_pmc_row_keeps_complete_target_and_silver_contract(self):
        target = canonicalize_target("# Title\n\n" + ("complete body " * 3000))
        row = _row(
            row_id="pmc_full_PMC123", doc_id="PMC123", split="train", target=target,
            images=["/tmp/page.png"], sample_form="strict_single_page", page_indices=[0],
            source_sha256="a" * 64, pdf_sha256="b" * 64,
            operation_counts=Counter(), reason_counts=Counter(),
        )
        self.assertEqual(row["messages"][1]["content"], target)
        self.assertEqual(row["meta"]["title_review_status"], SILVER_ACCEPTED)
        self.assertEqual(row["meta"]["title_target_sha256"], sha256_text(target))

    def test_mix_target_char_stats_use_complete_targets(self):
        target = "# Title\n\n" + ("x" * 20000) + "\n"
        pools = {
            pool: {
                s: [(self._entry(pool + "-" + s, s, target),
                     {"images": 1, "target_chars": len(target)},
                     ("readoc" if pool == "readoc_full" else "pmc", "SRC", pool + "-" + s))]
                for s in ("train", "validation", "test")
            }
            for pool in ("readoc_full", "pmc_full", "pmc_single")
        }
        _rows, report = build_split(pools, "train", "seed")
        self.assertEqual(report["target_chars"], len(target) * 3)

    @staticmethod
    def _entry(row_id, split, target="# T\n"):
        return {"id": row_id, "meta": {"split": split}, "messages": [{}, {"content": target}]}


if __name__ == "__main__":
    unittest.main()
