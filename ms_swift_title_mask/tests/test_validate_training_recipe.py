import json
import tempfile
import types
import unittest
from pathlib import Path

from ms_swift_title_mask.core import TitleMaskError
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


class ValidateTrainingRecipeTest(unittest.TestCase):
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
