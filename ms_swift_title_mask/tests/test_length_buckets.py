import json
import tempfile
import unittest
from pathlib import Path

from ms_swift_title_mask.scripts.build_length_buckets import (
    VISUAL_TOKENS_PER_IMAGE,
    bucket_for,
    count_row_tokens,
    scan_rows,
    validate_tokenizer,
    build_outputs,
)
from ms_swift_title_mask.core import sha256_text


class FakeFastTokenizer:
    is_fast = True
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        if text == "<image>":
            return {"input_ids": [999]}
        return {"input_ids": list(range(len(text.split()))) if text else []}


def row(target="a b", images=1):
    return {"id": "x", "messages": [{"role": "user", "content": "<image> prompt"},
            {"role": "assistant", "content": target}], "images": ["p"] * images,
            "meta": {"source": "s", "doc_id": "d", "split": "train"}}


class LengthBucketTest(unittest.TestCase):
    def test_build_outputs_dry_run_and_published_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            root.mkdir()
            rows = []
            for split in ("train", "validation", "test"):
                rows.append(self._training_row(split, "short", "a"))
                rows.append(self._training_row(split, "long", " ".join(["x"] * 5000)))
                with (root / f"{split}.jsonl").open("w") as handle:
                    for row in rows[-2:]:
                        handle.write(json.dumps(row) + "\n")
            dry = build_outputs(root, Path(tmp) / "dry", FakeFastTokenizer(), max_length=100,
                                dry_run=True)
            self.assertTrue(dry["dry_run"])
            self.assertIn("overall", dry)
            self.assertIn("pools", dry["splits"]["train"])
            self.assertIn("sources", dry["splits"]["train"])
            self.assertEqual(set(dry["splits"]), {"train", "validation", "test"})
            self.assertEqual(dry["length_contract"]["template"], "unlimited_ocr")
            self.assertEqual(dry["length_contract"]["deepseek_version"], "v1")
            self.assertFalse(dry["length_contract"]["crop_mode"])
            self.assertEqual(dry["length_contract"]["image_size"], 1024)
            self.assertEqual(dry["length_contract"]["base_size"], 1024)
            self.assertIn("not a universal model constant", dry["length_contract"]["scope"])
            self.assertEqual(dry["visual_tokens_per_image"], VISUAL_TOKENS_PER_IMAGE)
            self.assertEqual(set(dry["input_sha256"]), {"train", "validation", "test"})
            self.assertIn("pools", dry["overall"])
            self.assertIn("sources", dry["overall"])
            self.assertFalse((Path(tmp) / "dry").exists())
            output = Path(tmp) / "out"
            build_outputs(root, output, FakeFastTokenizer(), max_length=300)
            report = json.loads((output / "report.json").read_text())
            self.assertFalse(report["dry_run"])
            self.assertEqual(set(report["splits"]), {"train", "validation", "test"})
            for split in ("train", "validation", "test"):
                self.assertTrue((output / f"fit/{split}.jsonl").is_file())
                self.assertTrue((output / f"overflow/{split}.jsonl").is_file())
                self.assertTrue((output / f"buckets/le_4k_{split}.jsonl").is_file())
                self.assertTrue((output / f"smoke/{split}.jsonl").is_file())
            fit = [json.loads(x) for x in (output / "fit/train.jsonl").read_text().splitlines()]
            overflow = [json.loads(x) for x in (output / "overflow/train.jsonl").read_text().splitlines()]
            self.assertEqual(fit[0]["messages"], self._training_row("train", "short", "a")["messages"])
            self.assertEqual(overflow[0]["messages"], self._training_row("train", "long", " ".join(["x"] * 5000))["messages"])
            smoke = json.loads((output / "smoke/train.jsonl").read_text().strip())
            self.assertEqual(smoke["id"], "short-train")
            self.assertTrue((output / "buckets/le_4k_train.jsonl").exists())
            with self.assertRaises(ValueError):
                build_outputs(root, root / "nested", FakeFastTokenizer(), dry_run=False)

    @staticmethod
    def _training_row(split, row_id, target):
        target = "# T\n\n" + target.rstrip() + "\n"
        return {"id": row_id + "-" + split, "channel": "title_reviewed",
                "messages": [{"role": "user", "content": "<image>document parsing."},
                             {"role": "assistant", "content": target}],
                "images": ["/tmp/page.png"],
                "meta": {"source": "READoc-test", "doc_id": row_id, "sample_form": "full_document",
                         "n_pages": 1, "page_indices": [0], "split": split,
                         "title_review_status": "SILVER_ACCEPTED",
                         "title_review_id": "review", "title_review_version": "v1",
                         "title_reviewer": "test", "title_reviewed_at": "2026-01-01T00:00:00+00:00",
                         "title_target_sha256": sha256_text(target), "title_heading_count": 1,
                         "title_ruleset_version": "v1", "title_source_sha256": "a" * 64,
                         "mix_pool": "readoc_full"}}
    def test_exact_visual_count_and_target_sha_without_mutation(self):
        r = row("one two three", 2)
        original = r["messages"][1]["content"]
        audit = count_row_tokens(r, FakeFastTokenizer())
        self.assertEqual(audit["visual_tokens"], 2 * VISUAL_TOKENS_PER_IMAGE)
        self.assertEqual(audit["target_sha256"], sha256_text(original))
        self.assertEqual(r["messages"][1]["content"], original)

    def test_boundaries_and_stable_input_order(self):
        self.assertEqual([bucket_for(n) for n in (4096, 8192, 16384, 24576, 32768, 32769)],
                         ["le_4k", "le_8k", "le_16k", "le_24k", "le_32k", "overflow"])
        rows = [row("a"), row("a b")]
        self.assertEqual([a["id"] for _, a in scan_rows(rows, FakeFastTokenizer(), 32768)], ["x", "x"])

    def test_overflow_is_classified_without_truncation(self):
        r = row(" ".join(["x"] * 40000))
        original = r["messages"][1]["content"]
        result = scan_rows([r], FakeFastTokenizer(), 4096)[0][1]
        self.assertFalse(result["fits"])
        self.assertEqual(result["target_tokens"], 40000)
        self.assertEqual(r["messages"][1]["content"], original)

    def test_placeholder_contract_fails_closed(self):
        class Bad(FakeFastTokenizer):
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [1, 2]} if text == "<image>" else super().__call__(text, add_special_tokens)
        with self.assertRaises(ValueError):
            validate_tokenizer(Bad())


if __name__ == "__main__":
    unittest.main()
