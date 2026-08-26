import json
import tempfile
import types
import unittest
from pathlib import Path

from ms_swift_title_mask.core import TitleMaskError, sha256_text
from ms_swift_title_mask.scripts.build_recipe_mix import build_split, load_pool
from ms_swift_title_mask.scripts.validate_training_recipe import audit, validate_recipe


def _row(
    row_id,
    source,
    doc_id,
    target,
    *,
    content_status=None,
    title_status="SILVER_ACCEPTED",
):
    meta = {
        "source": source,
        "doc_id": doc_id,
        "sample_form": "full_document",
        "split": "train",
        "title_review_status": title_status,
    }
    if content_status is not None:
        meta["content_review_status"] = content_status
    return {
        "id": row_id,
        "messages": [
            {"role": "user", "content": "<image>Multi page merge."},
            {"role": "assistant", "content": target},
        ],
        "images": ["/tmp/page.png"],
        "meta": meta,
    }


def _reviewed_row(row_id, doc_id):
    target = "# T\n\nBody.\n"
    row = _row(row_id, "READoc-github", doc_id, target)
    row["channel"] = "title_reviewed"
    row["messages"][0]["content"] = "<image>document parsing."
    row["meta"].update(
        {
            "title_review_id": f"review-{row_id}",
            "title_review_version": "v1",
            "title_reviewer": "test",
            "title_reviewed_at": "2026-08-26T00:00:00+08:00",
            "title_target_sha256": sha256_text(target),
            "title_heading_count": 1,
            "title_ruleset_version": "v1",
            "title_source_sha256": "a" * 64,
        }
    )
    return row


class RecipeMixTest(unittest.TestCase):
    def test_explicit_row_exclusion_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "train.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (_reviewed_row("keep", "d1"), _reviewed_row("drop", "d2"))
                ),
                encoding="utf-8",
            )
            rows, loaded_path, excluded = load_pool(
                root,
                "readoc",
                "train",
                check_images=False,
                content_status=None,
                excluded_ids=frozenset({"drop"}),
            )
        self.assertEqual(loaded_path, path)
        self.assertEqual([row["id"] for row in rows], ["keep"])
        self.assertEqual(excluded, [{
            "id": "drop",
            "pool": "readoc",
            "split": "train",
            "source": "READoc-github",
            "doc_id": "d2",
        }])

    def test_auxiliary_pool_is_capped_by_target_chars(self):
        loaded = {
            "readoc": [_row("r1", "READoc-arxiv", "r1", "x" * 900)],
            "pmc": [
                _row(f"p{i}", "PMC-v26-synthetic", f"p{i}", "y" * 50)
                for i in range(10)
            ],
        }
        rows, report = build_split(
            loaded,
            {"pmc": 0.10},
            seed="seed",
            split="train",
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(report["pools"]["pmc"]["target_chars"], 100)
        self.assertEqual(report["pools"]["pmc"]["target_char_share"], 0.10)

    def test_cross_pool_document_duplicate_is_rejected(self):
        loaded = {
            "full": [_row("full", "PMC-v26-synthetic", "same", "x" * 100)],
            "single": [_row("single", "PMC-v26-synthetic", "same", "x" * 10)],
        }
        with self.assertRaisesRegex(TitleMaskError, "duplicate document"):
            build_split(loaded, {}, seed="seed", split="train")

    def test_readoc_recipe_rejects_legacy_natural_union(self):
        rows = [
            _row("r1", "READoc-arxiv", "r1", "# R\n"),
            _row("p1", "PMC-v26-synthetic", "p1", "# P\n"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = audit([path], check_images=False, full_row_validation=False)
        args = types.SimpleNamespace(
            recipe="readoc_r0",
            min_cjk_char_share=0.15,
            max_pmc_char_share=0.12,
        )
        with self.assertRaisesRegex(TitleMaskError, "only READoc"):
            validate_recipe(report, args)

    def test_artifact_scan_ignores_html_comments(self):
        rows = [
            _row(
                "r1",
                "READoc-github",
                "r1",
                "# R\n\n<!-- <img src=\"hidden.png\"> -->\n",
            )
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            report = audit([path], check_images=False, full_row_validation=False)
        self.assertEqual(report["artifacts"]["xml_image_tag"], 0)

    def test_trusted_title_requires_human_accepted_headings(self):
        rows = [_row("r1", "READoc-arxiv", "r1", "# R\n")]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = audit([path], check_images=False, full_row_validation=False)
        args = types.SimpleNamespace(
            recipe="trusted_title",
            min_cjk_char_share=0.15,
            max_pmc_char_share=0.12,
        )
        with self.assertRaisesRegex(TitleMaskError, "HUMAN_ACCEPTED"):
            validate_recipe(report, args)

    def test_trusted_title_accepts_human_reviewed_readoc(self):
        rows = [
            _row(
                "r1",
                "READoc-arxiv",
                "r1",
                "# R\n",
                title_status="HUMAN_ACCEPTED",
            )
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = audit([path], check_images=False, full_row_validation=False)
        args = types.SimpleNamespace(
            recipe="trusted_title",
            min_cjk_char_share=0.15,
            max_pmc_char_share=0.12,
        )
        validate_recipe(report, args)


if __name__ == "__main__":
    unittest.main()
