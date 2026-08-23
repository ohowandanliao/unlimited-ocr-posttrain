import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ms_swift_title_mask.core import canonicalize_target, sha256_text
from ms_swift_title_mask.data_contract import read_jsonl


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
PREPARE = BUNDLE_ROOT / "scripts/prepare_reviewed_jsonl.py"
VALIDATE = BUNDLE_ROOT / "scripts/validate_reviewed_jsonl.py"


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class ReviewedDataGateTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for name in ["train.png", "validation.png", "test.png", "silver.png"]:
            (self.images / name).write_bytes(b"not-decoded-by-the-builder")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def target(name):
        return canonicalize_target(f"# {name}\n\nBody for {name}.\n")

    def source_rows(self):
        rows = []
        for name in ["train", "validation", "test", "silver"]:
            rows.append(
                {
                    "id": name,
                    "prompt": "<image>document parsing.",
                    "target": self.target(name),
                    "images": [f"{name}.png"],
                    "meta": {"source": "unit_test"},
                }
            )
        return rows

    def review_rows(self, sha_override=None):
        rows = []
        for name in ["train", "validation", "test"]:
            rows.append(
                {
                    "id": name,
                    "status": "HUMAN_ACCEPTED",
                    "split": name,
                    "review_id": f"human-{name}-v1",
                    "review_version": "title-review-v1",
                    "reviewer": "reviewer@example",
                    "reviewed_at": "2026-08-23T10:00:00+08:00",
                    "assistant_sha256": sha_override or sha256_text(self.target(name)),
                    "title_heading_count": 1,
                }
            )
        rows.append({"id": "silver", "status": "SILVER_CANDIDATE"})
        return rows

    def run_prepare(self, reviews=None):
        source = self.root / "source.jsonl"
        manifest = self.root / "review.jsonl"
        output = self.root / "out"
        write_jsonl(source, self.source_rows())
        write_jsonl(manifest, reviews if reviews is not None else self.review_rows())
        result = subprocess.run(
            [
                sys.executable,
                str(PREPARE),
                "--source-jsonl",
                str(source),
                "--review-manifest",
                str(manifest),
                "--image-root",
                str(self.images),
                "--output-dir",
                str(output),
            ],
            text=True,
            capture_output=True,
        )
        return result, output

    def test_only_human_accepted_rows_are_published(self):
        result, output = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        ids = []
        for split in ["train", "validation", "test"]:
            rows = [row for _, row in read_jsonl(output / f"{split}.jsonl")]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["channel"], "title_reviewed")
            self.assertEqual(rows[0]["meta"]["title_review_status"], "HUMAN_ACCEPTED")
            self.assertTrue(Path(rows[0]["images"][0]).is_absolute())
            ids.append(rows[0]["id"])
        self.assertNotIn("silver", ids)

        validate = subprocess.run(
            [sys.executable, str(VALIDATE)]
            + [str(output / f"{split}.jsonl") for split in ["train", "validation", "test"]],
            text=True,
            capture_output=True,
        )
        self.assertEqual(validate.returncode, 0, validate.stderr)

    def test_rules_accepted_silver_is_published_with_provenance(self):
        reviews = self.review_rows()
        for review in reviews:
            if review["id"] == "train":
                review.update(
                    {
                        "status": "SILVER_ACCEPTED",
                        "ruleset_version": "unit-test-rules-v1",
                        "source_sha256": "1" * 64,
                    }
                )
        result, output = self.run_prepare(reviews)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [row for _, row in read_jsonl(output / "train.jsonl")]
        self.assertEqual(rows[0]["meta"]["title_review_status"], "SILVER_ACCEPTED")
        self.assertEqual(rows[0]["meta"]["title_ruleset_version"], "unit-test-rules-v1")
        self.assertEqual(rows[0]["meta"]["title_source_sha256"], "1" * 64)

    def test_zero_human_accepted_fails_closed(self):
        result, output = self.run_prepare([{"id": "silver", "status": "SILVER_CANDIDATE"}])
        self.assertEqual(result.returncode, 2)
        self.assertIn("0 accepted", result.stderr)
        self.assertFalse(output.exists())

    def test_sha_mismatch_fails_without_partial_output(self):
        result, output = self.run_prepare(self.review_rows(sha_override="0" * 64))
        self.assertEqual(result.returncode, 2)
        self.assertIn("SHA mismatch", result.stderr)
        self.assertFalse(output.exists())

    def test_empty_required_split_fails(self):
        reviews = self.review_rows()[:-2] + [self.review_rows()[-1]]
        result, output = self.run_prepare(reviews)
        self.assertEqual(result.returncode, 2)
        self.assertIn("empty required splits", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
