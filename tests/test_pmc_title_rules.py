import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPT_DIR))

from build_pmc_title_candidates import build_dataset  # noqa: E402
from pmc_title_rules import (  # noqa: E402
    MiddleJsonError,
    analyze_document,
    parse_numbered_heading,
)


def span_line(text):
    return {"spans": [{"type": "text", "content": text}]}


def block(block_type, text, *, level=None, index=0, **extra):
    value = {
        "type": block_type,
        "index": index,
        "bbox": [10, 20, 100, 40],
        "lines": [span_line(text)] if text is not None else [],
    }
    if level is not None:
        value["text_level"] = level
    value.update(extra)
    return value


def payload(pages):
    for page in pages:
        page.setdefault("page_size", [595, 842])
    return {"_backend": "vlm", "_version_name": "3.1.14", "pdf_info": pages}


def write_pdf(path, page_count=1):
    try:
        import pymupdf
    except ImportError:
        from pypdf import PdfWriter

        writer = PdfWriter()
        for _ in range(page_count):
            writer.add_blank_page(width=595, height=842)
        with path.open("wb") as handle:
            writer.write(handle)
    else:
        document = pymupdf.open()
        for _ in range(page_count):
            document.new_page(width=595, height=842)
        document.save(path)
        document.close()


class PmcTitleRulesTest(unittest.TestCase):
    def analyze(self, pages, include_promotions=True):
        return analyze_document(
            doc_id="pmc-test",
            payload=payload(pages),
            source_middle_json="documents/pmc-test/middle.json",
            source_sha256="a" * 64,
            document_dir=Path("/nonexistent/pmc-test"),
            include_promotions=include_promotions,
        )

    def test_numbered_heading_levels(self):
        self.assertEqual(parse_numbered_heading("1 Introduction")["level"], 1)
        self.assertEqual(parse_numbered_heading("2.1 Methods")["level"], 2)
        self.assertEqual(parse_numbered_heading("3.4.1 Analysis")["level"], 3)
        self.assertTrue(parse_numbered_heading("3.4.33.4.3 Principal Component Analysis")["suspicious"])
        self.assertIsNone(parse_numbered_heading("Introduction"))

    def test_page_indices_must_match_pdf_order(self):
        invalid_indices = ([0, 0], [0, 2], [1, 0])
        for indices in invalid_indices:
            with self.subTest(indices=indices):
                pages = [
                    {"page_idx": page_idx, "para_blocks": []}
                    for page_idx in indices
                ]
                with self.assertRaisesRegex(MiddleJsonError, "page_idx must equal"):
                    self.analyze(pages)

    def test_clean_title_is_auto_silver(self):
        result = self.analyze(
            [{"page_idx": 0, "para_blocks": [block("title", "2.1 Methods", level=2)]}]
        )
        candidate = result["candidates"][0]
        self.assertEqual(candidate["operation"], "retain")
        self.assertEqual(candidate["review_status"], "AUTO_SILVER")
        self.assertEqual(candidate["proposed_level"], 2)
        self.assertEqual(candidate["page_idx"], 0)
        self.assertEqual(candidate["pdf_page"], 1)
        self.assertEqual(candidate["page_size"], [595, 842])

    def test_number_conflict_is_review_only_and_not_releveled(self):
        result = self.analyze(
            [{"page_idx": 0, "para_blocks": [block("title", "1. Familiarization with the Data.", level=3)]}]
        )
        candidate = result["candidates"][0]
        self.assertEqual(candidate["review_status"], "REVIEW_REQUIRED")
        self.assertIn("NUMBER_LEVEL_CONFLICT", candidate["reason_codes"])
        self.assertEqual(candidate["input_level"], 3)
        self.assertEqual(candidate["proposed_level"], 3)

    def test_title_risk_rules(self):
        pages = [
            {
                "page_idx": 0,
                "para_blocks": [
                    block("title", "Introduction", level=1, index=0),
                    block("title", "Deep child", level=3, index=1),
                    block("title", "Introduction", level=1, index=2),
                    block("title", "Figure 2. Study flow", level=2, index=3),
                    block("title", "https://example.org", level=2, index=4),
                    block("title", None, index=5),
                ],
            }
        ]
        result = self.analyze(pages)
        by_text = {candidate["display_text"]: candidate for candidate in result["candidates"]}
        self.assertIn("DUPLICATE_TITLE", by_text["Introduction"]["reason_codes"])
        self.assertIn("LEVEL_SKIP", by_text["Deep child"]["reason_codes"])
        self.assertIn("CAPTION_LIKE_TITLE", by_text["Figure 2. Study flow"]["reason_codes"])
        self.assertIn("METADATA_LIKE_TITLE", by_text["https://example.org"]["reason_codes"])
        self.assertIn("EMPTY_TITLE", by_text[""]["reason_codes"])
        self.assertIn("INVALID_LEVEL", by_text[""]["reason_codes"])

    def test_text_promotions_always_require_review(self):
        pages = [
            {
                "page_idx": 0,
                "para_blocks": [
                    block("text", "Introduction", index=0),
                ],
            },
            {
                "page_idx": 1,
                "para_blocks": [
                    block("text", "2.3 Statistical Analysis", index=0),
                    block("text", "This is an ordinary paragraph.", index=1),
                ],
            },
        ]
        result = self.analyze(pages)
        self.assertEqual([row["operation"] for row in result["candidates"]], ["promote", "promote"])
        self.assertTrue(all(row["review_status"] == "REVIEW_REQUIRED" for row in result["candidates"]))
        self.assertEqual(result["candidates"][1]["proposed_level"], 2)

    def test_numbered_promotion_rejects_page_one_math_year_and_sentence_fragments(self):
        pages = [
            {
                "page_idx": 0,
                "para_blocks": [block("text", "2.3 Statistical Analysis", index=0)],
            },
            {
                "page_idx": 1,
                "para_blocks": [
                    block("text", "1 Methylene blue + OH -> Products", index=0),
                    block("text", "2 Smith A. Article title. 2025.", index=1),
                    block("text", "3.1 given in Section 4, we write", index=2),
                    block("text", "4 SmithAArticleJournal202 11251299", index=3),
                    block("text", "5. Table 1.", index=4),
                    block("text", "6. Comparison of Path Diagram", index=5),
                ],
            },
        ]
        result = self.analyze(pages)
        self.assertEqual(result["candidates"], [])

    def test_numbered_promotion_rejects_addresses_and_long_description_lists(self):
        pages = [
            {
                "page_idx": 0,
                "para_blocks": [block("title", "Article Title", level=1)],
            },
            {
                "page_idx": 1,
                "para_blocks": [
                    block("text", "10 Example Clinic, Berlin, Germany", index=0),
                    block("title", "Figure 1. Long description", level=1, index=1),
                    block("text", "1. Identification Phase", index=2),
                    block("title", "Appendix", level=1, index=3),
                    block("text", "2. Visual Composition", index=4),
                    block("list", "* first list item", index=5),
                ],
            },
        ]
        result = self.analyze(pages)
        self.assertEqual(
            [candidate["display_text"] for candidate in result["candidates"]],
            ["Article Title", "Figure 1. Long description", "Appendix"],
        )

    def test_numbered_heading_with_commas_is_not_treated_as_an_address(self):
        result = self.analyze(
            [
                {"page_idx": 0, "para_blocks": [block("title", "Article Title", level=1)]},
                {
                    "page_idx": 1,
                    "para_blocks": [
                        block("text", "6. Randomization, Blinding, and Sample Size", index=0)
                    ],
                },
            ]
        )
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["candidates"][1]["operation"], "promote")

    def test_adjacent_numbered_text_is_treated_as_a_list_not_headings(self):
        result = self.analyze(
            [
                {"page_idx": 0, "para_blocks": [block("title", "Article Title", level=1)]},
                {
                    "page_idx": 1,
                    "para_blocks": [
                        block("text", "1. Experts in Group A", index=0),
                        block("text", "2. Residents in Group B", index=1),
                    ],
                },
            ]
        )
        self.assertEqual(len(result["candidates"]), 1)

    def test_adjacent_parent_and_child_numbered_headings_remain_candidates(self):
        result = self.analyze(
            [
                {"page_idx": 0, "para_blocks": [block("title", "Article Title", level=1)]},
                {
                    "page_idx": 1,
                    "para_blocks": [
                        block("text", "2. Strengths of the Manuscript", index=0),
                        block("text", "2.1 Practical Relevance", index=1),
                    ],
                },
            ]
        )
        self.assertEqual(
            [candidate["display_text"] for candidate in result["candidates"]],
            ["Article Title", "2. Strengths of the Manuscript", "2.1 Practical Relevance"],
        )

    def test_references_with_entries_remains_review_candidate_in_review_material(self):
        result = self.analyze(
            [
                {
                    "page_idx": 0,
                    "para_blocks": [block("text", "Reviewer", index=0)],
                },
                {
                    "page_idx": 1,
                    "para_blocks": [
                        block("text", "References", index=0),
                        block("text", "Smith A. Example article. (2024) 1, 2-3.", index=1),
                    ],
                },
            ]
        )
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["display_text"], "References")

    def test_canonical_promotion_rejects_duplicate_review_material_and_running_header(self):
        pages = [
            {
                "page_idx": 0,
                "para_blocks": [
                    block("title", "Methods", level=1, index=0),
                    block("text", "Methods", index=1),
                    block("text", "Results", index=2, bbox=[300, 50, 350, 70]),
                    block("text", "Body paragraph.", index=3),
                    block("title", "Results", level=1, index=4),
                ],
            },
            {
                "page_idx": 1,
                "para_blocks": [
                    block("text", "Reviewer #1", index=0),
                    block("text", "Introduction", index=1),
                ],
            },
            {
                "page_idx": 2,
                "para_blocks": [block("text", "Discussion", index=0)],
            },
        ]
        result = self.analyze(pages)
        self.assertEqual(
            [candidate["display_text"] for candidate in result["candidates"]],
            ["Methods", "Results"],
        )

    def test_affiliations_and_numbered_references_are_not_promoted(self):
        pages = [
            {
                "page_idx": 0,
                "para_blocks": [
                    block("text", "1 Department of Biology, Example University", index=0),
                    block("text", "5 Neuroscience Graduate Program, Example University, USA", index=1),
                    block("text", "2 Smith A., Article title, Journal. (2025) 1, 2-3.", index=2),
                    block("title", "References", level=1, index=3),
                    block("text", "1 Smith A. A long article title. Journal. 2025.", index=4),
                ],
            }
        ]
        result = self.analyze(pages)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["candidates"][0]["display_text"], "References")

    def test_preproc_is_ignored_and_content_flags_are_separate(self):
        title = block("title", "Methods", level=1)
        pages = [
            {
                "page_idx": 0,
                "para_blocks": [title, block("text", r"Broken \\fra c formula", index=1)],
                "preproc_blocks": [title, block("text", r"Broken \\fra c formula", index=1)],
            },
            {"page_idx": 1, "para_blocks": [], "preproc_blocks": []},
        ]
        result = self.analyze(pages)
        self.assertEqual(result["source_title_count"], 1)
        self.assertEqual(len(result["candidates"]), 1)
        self.assertIn("PREPROC_DUPLICATES_PARA", result["content_flags"])
        self.assertIn("BAD_LATEX", result["content_flags"])
        self.assertIn("EMPTY_PAGE", result["content_flags"])
        self.assertEqual(result["candidates"][0]["review_status"], "AUTO_SILVER")

    def test_table_wrap_with_only_an_image_is_not_treated_as_structured_table(self):
        image_only_table = block(
            "table",
            None,
            blocks=[
                {
                    "type": "table_body",
                    "lines": [
                        {
                            "spans": [
                                {
                                    "type": "table",
                                    "html": '<table-wrap><img src="table.jpg"></table-wrap>',
                                    "image_path": "assets/table.png",
                                }
                            ]
                        }
                    ],
                }
            ],
        )
        structured_table = block(
            "table",
            None,
            index=1,
            blocks=[
                {
                    "type": "table_body",
                    "lines": [
                        {
                            "spans": [
                                {
                                    "type": "table",
                                    "html": "<table-wrap><table><tr><td>x</td></tr></table></table-wrap>",
                                    "image_path": "assets/table2.png",
                                }
                            ]
                        }
                    ],
                }
            ],
        )
        result = self.analyze(
            [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        block("title", "Methods", level=1),
                        image_only_table,
                        structured_table,
                    ],
                }
            ]
        )
        self.assertIn("IMAGE_ONLY_TABLE", result["content_flags"])
        self.assertEqual(result["content_flag_counts"]["IMAGE_ONLY_TABLE"], 1)

    def test_multiline_title_has_independent_text_review_status(self):
        multiline = block("title", None, level=1)
        multiline["lines"] = [span_line("Clinical Isolates and"), span_line("Phenotypic Results")]
        result = self.analyze([{"page_idx": 0, "para_blocks": [multiline]}])
        candidate = result["candidates"][0]
        self.assertEqual(candidate["review_status"], "AUTO_SILVER")
        self.assertEqual(candidate["text_review_status"], "VISUAL_REVIEW_REQUIRED")
        self.assertEqual(candidate["text_quality_flags"], ["MULTILINE_SOURCE_TEXT"])
        self.assertEqual(len(result["text_review_records"]), 1)


class PmcTitleBuildTest(unittest.TestCase):
    def _make_input(self, root, *, pdf_pages=1):
        doc_dir = root / "documents" / "pmc1"
        doc_dir.mkdir(parents=True)
        source = payload(
            [
                {
                    "page_idx": 0,
                    "para_blocks": [
                        block("title", "1 Introduction", level=1, index=0),
                        block("text", "Body text.", index=1),
                    ],
                    "preproc_blocks": [
                        block("title", "1 Introduction", level=1, index=0),
                        block("text", "Body text.", index=1),
                    ],
                }
            ]
        )
        (doc_dir / "middle.json").write_text(json.dumps(source), encoding="utf-8")
        write_pdf(doc_dir / "document.pdf", pdf_pages)

    def test_build_is_deterministic_and_does_not_duplicate_preproc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            self._make_input(root)
            output_a = Path(tmp) / "out-a"
            output_b = Path(tmp) / "out-b"
            summary_a = build_dataset(input_root=root, output_root=output_a)
            summary_b = build_dataset(input_root=root, output_root=output_b)

            self.assertEqual(summary_a, summary_b)
            self.assertEqual(summary_a["totals"]["source_titles"], 1)
            self.assertEqual(summary_a["totals"]["candidates"], 1)
            manifest = json.loads(
                (output_a / "document_manifest.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["pages"], 1)
            self.assertEqual(manifest["pdf_pages"], 1)
            for filename in (
                "document_manifest.jsonl",
                "title_candidates.jsonl",
                "review_queue.jsonl",
                "text_review_queue.jsonl",
                "content_quarantine.jsonl",
                "summary.json",
            ):
                self.assertEqual(
                    (output_a / filename).read_bytes(),
                    (output_b / filename).read_bytes(),
                    filename,
                )

    def test_output_inside_input_is_rejected_without_touching_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            self._make_input(root)
            middle_path = root / "documents" / "pmc1" / "middle.json"
            source_before = middle_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "disjoint sibling trees"):
                build_dataset(
                    input_root=root,
                    output_root=root / "processed",
                    overwrite=True,
                )

            self.assertEqual(middle_path.read_bytes(), source_before)
            self.assertFalse((root / "processed").exists())

    def test_unreadable_pdf_is_recorded_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            self._make_input(root)
            (root / "documents" / "pmc1" / "document.pdf").write_bytes(
                b"not-a-real-pdf"
            )
            output = Path(tmp) / "output"

            with self.assertRaisesRegex(RuntimeError, "1 failed document"):
                build_dataset(input_root=root, output_root=output)

            manifest = json.loads(
                (output / "document_manifest.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "ERROR")
            self.assertIn("unreadable PDF", manifest["error"])

    def test_pdf_and_middle_json_page_counts_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            self._make_input(root, pdf_pages=2)
            output = Path(tmp) / "output"

            with self.assertRaisesRegex(RuntimeError, "1 failed document"):
                build_dataset(input_root=root, output_root=output)

            manifest = json.loads(
                (output / "document_manifest.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "ERROR")
            self.assertIn(
                "PDF page count 2 != middle.json page count 1",
                manifest["error"],
            )


if __name__ == "__main__":
    unittest.main()
