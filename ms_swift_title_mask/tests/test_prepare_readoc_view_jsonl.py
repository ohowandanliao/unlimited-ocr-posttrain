import unittest
from pathlib import Path

from ms_swift_title_mask.core import TitleMaskError
from ms_swift_title_mask.scripts.prepare_readoc_view_jsonl import normalize_row


class PrepareReadocViewTest(unittest.TestCase):
    def _row(self, target="# Heading\n\nBody"):
        return {
            "id": "readoc_READoc-github_demo",
            "source": "READoc-github",
            "images": ["/tmp/page-0.png", "/tmp/page-1.png"],
            "mode": "multi_base",
            "prompt": "<image>Multi page merge.",
            "target": target,
            "target_format": "markdown",
            "task": "multi_page_merge",
            "meta": {"doc_id": "demo", "n_pages": 2},
        }

    def test_normalizes_without_changing_content(self):
        normalized = normalize_row(
            self._row(),
            split="train",
            path=Path("/tmp/train.jsonl"),
            line_number=1,
            image_root=None,
            check_images=False,
        )
        self.assertEqual(normalized["channel"], "title_reviewed")
        self.assertEqual(normalized["messages"][1]["content"], "# Heading\n\nBody\n")
        self.assertEqual(normalized["meta"]["title_review_status"], "SILVER_ACCEPTED")
        self.assertEqual(normalized["meta"]["title_heading_count"], 1)

    def test_requires_a_heading_for_weighted_training(self):
        with self.assertRaisesRegex(TitleMaskError, "no CommonMark ATX heading"):
            normalize_row(
                self._row("Body only"),
                split="train",
                path=Path("/tmp/train.jsonl"),
                line_number=1,
                image_root=None,
                check_images=False,
            )

    def test_accepts_mineru_heading_prior_only_when_enabled(self):
        row = self._row()
        row["prompt"] = (
            "<image>Multi page merge.\n\n"
            "The following headings are extracted from the MinerU parsing result of the same document.\n"
            "Use them only as structural hints.\n"
            "Verify and correct the heading text, heading levels, ordering, and document structure "
            "according to the document images.\n"
            "Do not blindly copy the MinerU headings if they conflict with the document images.\n\n"
            "### MinerU headings\n\n"
            "## Heading\n\n"
            "### Final corrected Markdown\n"
        )
        row["meta"]["prior"] = "mineru_heading"

        with self.assertRaisesRegex(TitleMaskError, "requires --allow-title-prior"):
            normalize_row(
                row,
                split="train",
                path=Path("/tmp/train.jsonl"),
                line_number=1,
                image_root=None,
                check_images=False,
            )

        normalized = normalize_row(
            row,
            split="train",
            path=Path("/tmp/train.jsonl"),
            line_number=1,
            image_root=None,
            check_images=False,
            allow_title_prior=True,
        )
        self.assertEqual(normalized["messages"][0]["content"], row["prompt"])
        self.assertEqual(normalized["meta"]["input_prompt_variant"], "mineru_heading_prior")


if __name__ == "__main__":
    unittest.main()
