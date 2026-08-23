import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"
sys.path.insert(0, str(SCRIPT_DIR))

from build_readoc_title_gt import build_dataset  # noqa: E402
from readoc_title_rules import (  # noqa: E402
    RULESET_VERSION,
    analyze_markdown,
    apply_line_decisions,
    parse_headings,
    sha256_bytes,
    validate_title_only_changes,
)


def pdf_bytes(page_count=1):
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=595, height=842)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def write_policy(path, document_decisions=None):
    path.write_text(
        json.dumps(
            {
                "schema_version": "readoc-title-review-policy-v1",
                "ruleset_version": RULESET_VERSION,
                "bulk_decisions": [
                    {
                        "decision_id": "ARXIV_H6_ABSTRACT_TO_H2_V1",
                        "decision": "ACCEPT_SILVER",
                        "source": "arxiv",
                        "operation": "change_level",
                        "current_level": 6,
                        "normalized_text": "abstract",
                        "required_source_kind": "atx_top_level",
                        "proposed_level": 2,
                        "evidence_documents": ["sample"],
                        "reason": "test policy",
                    }
                ],
                "document_decisions": document_decisions or [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def make_input(root, source, documents):
    gt_root = root / "ground_truth" / source
    archive_root = root / "archives"
    gt_root.mkdir(parents=True)
    archive_root.mkdir(parents=True)
    with zipfile.ZipFile(archive_root / f"{source}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for doc_id, content in documents.items():
            (gt_root / f"{doc_id}.md").write_text(content, encoding="utf-8")
            archive.writestr(f"{source}/pdf/{doc_id}.pdf", pdf_bytes())


class ReadocTitleRulesTest(unittest.TestCase):
    def analyze(self, content, source="arxiv", doc_id="doc"):
        return analyze_markdown(
            source=source,
            doc_id=doc_id,
            content=content,
            source_sha256=sha256_bytes(content.encode("utf-8")),
        )

    def test_commonmark_parser_ignores_fenced_code_headings(self):
        content = "# Real title\n\n```bash\n# shell comment\n## another comment\n```\n\n## Section\n"
        headings = parse_headings(content)
        self.assertEqual([(item["level"], item["text"]) for item in headings], [(1, "Real title"), (2, "Section")])

    def test_only_arxiv_h6_abstract_is_policy_candidate(self):
        content = (
            "# Paper\n\n###### Abstract.\n\nText\n\n###### Contents\n\n"
            "###### Acknowledgements.\n"
        )
        result = self.analyze(content)
        changes = [item for item in result["candidates"] if item["operation"] == "change_level"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["current_text"], "Abstract.")
        self.assertEqual(changes[0]["proposed_level"], 2)
        self.assertTrue(all(item["operation"] == "retain" for item in result["candidates"][2:]))

    def test_github_h6_abstract_is_preserved(self):
        result = self.analyze("# Repo\n\n###### Abstract\n", source="github")
        self.assertEqual(result["candidates"][1]["operation"], "retain")

    def test_missing_h1_and_hierarchy_anomalies_require_review(self):
        content = "## Project\n\n#### Deep section\n\n## Project\n"
        result = self.analyze(content, source="github")
        self.assertIn("NO_H1", result["document_risks"])
        self.assertIn("LEVEL_SKIP", result["document_risks"])
        self.assertIn("DUPLICATE_HEADING_TEXT", result["document_risks"])
        first = result["candidates"][0]
        self.assertIn("DOCUMENT_HAS_NO_H1", first["reason_codes"])
        self.assertEqual(first["review_status"], "REVIEW_REQUIRED")
        self.assertTrue(all(item["operation"] == "retain" for item in result["candidates"]))

    def test_multiple_h1_is_not_automatically_flattened(self):
        result = self.analyze("# One\n\n# Two\n\n## Child\n", source="github")
        self.assertIn("MULTIPLE_H1", result["document_risks"])
        self.assertEqual([item["operation"] for item in result["candidates"]], ["retain", "retain", "retain"])

    def test_empty_h1_is_reviewed_not_guessed(self):
        result = self.analyze("#\n\n## Usage\n", source="github")
        self.assertIn("EMPTY_HEADING", result["document_risks"])
        self.assertIn("EMPTY_HEADING", result["candidates"][0]["reason_codes"])
        self.assertIsNone(
            next((item for item in result["candidates"] if item["operation"] == "promote"), None)
        )

    def test_title_only_validator_rejects_unreviewed_body_change(self):
        source = "# Title\n\nBody\n"
        target = "# Title\n\nChanged body\n"
        errors = validate_title_only_changes(source, target, [])
        self.assertEqual(errors[0]["type"], "unreviewed_line_changed")

    def test_apply_decisions_changes_only_declared_lines(self):
        source = "# Title\n\n###### Abstract.\n\nBody\n"
        target, applied = apply_line_decisions(
            source,
            [
                {
                    "line": 3,
                    "operation": "change_level",
                    "expected_source_line": "###### Abstract.",
                    "proposed_level": 2,
                }
            ],
        )
        self.assertEqual(target, "# Title\n\n## Abstract.\n\nBody\n")
        self.assertEqual(validate_title_only_changes(source, target, applied), [])

    def test_builder_is_complete_deterministic_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "READoc"
            documents = {
                "paper": "# Paper\n\n###### Abstract\n\nBody\n",
                "risk": "# Repo\n\n#### Skipped level\n",
            }
            make_input(input_root, "arxiv", documents)
            policy = tmp_path / "policy.json"
            write_policy(policy)
            source_before = {
                doc_id: (input_root / "ground_truth" / "arxiv" / f"{doc_id}.md").read_bytes()
                for doc_id in documents
            }
            archive_before = (input_root / "archives" / "arxiv.zip").read_bytes()
            output_a = tmp_path / "output-a"
            output_b = tmp_path / "output-b"

            summary_a = build_dataset(
                input_root=input_root,
                output_root=output_a,
                policy_path=policy,
                sources=["arxiv"],
                include_pdf_text_profile=False,
            )
            summary_b = build_dataset(
                input_root=input_root,
                output_root=output_b,
                policy_path=policy,
                sources=["arxiv"],
                include_pdf_text_profile=False,
            )

            self.assertEqual(summary_a, summary_b)
            self.assertTrue(json.loads((output_a / "validation.json").read_text())["valid"])
            self.assertEqual(len(list((output_a / "ground_truth" / "arxiv").glob("*.md"))), 2)
            self.assertIn(
                "## Abstract",
                (output_a / "ground_truth" / "arxiv" / "paper.md").read_text(),
            )
            self.assertIn(
                "#### Skipped level",
                (output_a / "ground_truth" / "arxiv" / "risk.md").read_text(),
            )
            self.assertEqual(
                json.loads((output_a / "summary.json").read_text())["totals"]["accepted_changes"],
                1,
            )
            manifest = [
                json.loads(line)
                for line in (output_a / "document_manifest.jsonl").read_text().splitlines()
            ]
            self.assertTrue(all(item["full_ce_eligible"] is None for item in manifest))
            self.assertTrue(all(item["title_weighted_eligible"] is None for item in manifest))
            for doc_id, before in source_before.items():
                self.assertEqual(
                    (input_root / "ground_truth" / "arxiv" / f"{doc_id}.md").read_bytes(),
                    before,
                )
            self.assertEqual((input_root / "archives" / "arxiv.zip").read_bytes(), archive_before)
            for filename in (
                "document_manifest.jsonl",
                "pdf_profiles.jsonl",
                "heading_inventory.jsonl",
                "title_candidates.jsonl",
                "review_queue.jsonl",
                "change_manifest.jsonl",
                "summary.json",
                "validation.json",
            ):
                self.assertEqual((output_a / filename).read_bytes(), (output_b / filename).read_bytes())

    def test_document_decision_is_bound_to_pdf_and_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "READoc"
            content = "Paper title\n\n## Section\n"
            make_input(input_root, "arxiv", {"paper": content})
            with zipfile.ZipFile(input_root / "archives" / "arxiv.zip") as archive:
                source_pdf = archive.read("arxiv/pdf/paper.pdf")
            decision = {
                "decision_id": "PROMOTE_TITLE_V1",
                "decision": "ACCEPT_SILVER",
                "source": "arxiv",
                "doc_id": "paper",
                "source_sha256": sha256_bytes(content.encode("utf-8")),
                "pdf_sha256": sha256_bytes(source_pdf),
                "line": 1,
                "operation": "promote",
                "expected_source_line": "Paper title",
                "proposed_level": 1,
                "proposed_text": "Paper title",
                "pdf_page": 1,
                "evidence_method": "pdf_page_visual_review",
                "reason": "test decision",
            }
            policy = tmp_path / "policy.json"
            write_policy(policy, [decision])
            output = tmp_path / "output"

            build_dataset(
                input_root=input_root,
                output_root=output,
                policy_path=policy,
                sources=["arxiv"],
                include_pdf_text_profile=False,
            )
            change = json.loads((output / "change_manifest.jsonl").read_text())
            self.assertEqual(change["pdf_sha256"], decision["pdf_sha256"])
            self.assertEqual(change["pdf_page"], 1)
            self.assertEqual(change["pdf_page_evidence_type"], "pdf_page_visual_review")
            self.assertIsNotNone(change["candidate_id"])

            decision["pdf_sha256"] = "0" * 64
            write_policy(policy, [decision])
            with self.assertRaisesRegex(RuntimeError, "failed validation"):
                build_dataset(
                    input_root=input_root,
                    output_root=tmp_path / "bad-output",
                    policy_path=policy,
                    sources=["arxiv"],
                    include_pdf_text_profile=False,
                )
            error_record = json.loads(
                (tmp_path / "bad-output" / "document_manifest.jsonl").read_text()
            )
            self.assertEqual(error_record["status"], "ERROR")
            self.assertIn("PDF hash mismatch", error_record["error"])

            decision["pdf_sha256"] = sha256_bytes(source_pdf)
            decision["evidence_method"] = None
            write_policy(policy, [decision])
            with self.assertRaisesRegex(RuntimeError, "failed validation"):
                build_dataset(
                    input_root=input_root,
                    output_root=tmp_path / "null-evidence-output",
                    policy_path=policy,
                    sources=["arxiv"],
                    include_pdf_text_profile=False,
                )
            error_record = json.loads(
                (tmp_path / "null-evidence-output" / "document_manifest.jsonl").read_text()
            )
            self.assertIn("lacks evidence method", error_record["error"])

    def test_output_inside_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "READoc"
            make_input(root, "arxiv", {"paper": "# Paper\n"})
            policy = Path(tmp) / "policy.json"
            write_policy(policy)
            with self.assertRaisesRegex(ValueError, "disjoint sibling trees"):
                build_dataset(
                    input_root=root,
                    output_root=root / "derived",
                    policy_path=policy,
                    sources=["arxiv"],
                )
            self.assertFalse((root / "derived").exists())


if __name__ == "__main__":
    unittest.main()
