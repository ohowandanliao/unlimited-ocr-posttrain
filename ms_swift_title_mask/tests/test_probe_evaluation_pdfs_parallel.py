import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import probe_evaluation_pdfs_parallel as runner


class ProbeEvaluationPdfsParallelTest(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(max_length=20480, dpi=144, trust_legacy_rows=False)
        self.row = {
            "prompt": "<image>Multi page parsing.",
            "pages": 2,
            "response": {"num_images": 2, "prompt": "<image>Multi page parsing."},
            "gen": {
                "max_length": 20480,
                "dpi": 144,
                "no_repeat_ngram_size": 35,
                "ngram_window": 1024,
                "temperature": 0.0,
            },
        }

    def test_resume_rejects_changed_generation_or_page_contract(self):
        expected = self.row["prompt"]
        self.assertTrue(runner.row_is_compatible(self.row, expected, self.args, 2))

        cases = (
            ("temperature", {"gen": {**self.row["gen"], "temperature": 0.1}}, 2),
            ("no_repeat_ngram_size", {"gen": {**self.row["gen"], "no_repeat_ngram_size": 34}}, 2),
            ("ngram_window", {"gen": {**self.row["gen"], "ngram_window": 128}}, 2),
            ("row_pages", {"pages": 1}, 2),
            ("response_pages", {"response": {**self.row["response"], "num_images": 1}}, 2),
            ("response_prompt", {"response": {**self.row["response"], "prompt": "wrong"}}, 2),
        )
        for name, change, page_count in cases:
            with self.subTest(name=name):
                row = {**self.row, **change}
                self.assertFalse(runner.row_is_compatible(row, expected, self.args, page_count))

    def test_resume_type_errors_fail_closed_and_legacy_requires_flag(self):
        self.assertFalse(runner.row_is_compatible({"prompt": self.row["prompt"], "pages": "bad"}, self.row["prompt"], self.args, 2))
        legacy = {key: value for key, value in self.row.items() if key != "gen"}
        self.assertFalse(runner.row_is_compatible(legacy, self.row["prompt"], self.args, 2))
        self.args.trust_legacy_rows = True
        self.assertTrue(runner.row_is_compatible(legacy, self.row["prompt"], self.args, 2))

    def test_same_stem_different_pdf_content_uses_different_cache_key(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "one" / "sample.pdf"
            second = root / "two" / "sample.pdf"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first PDF")
            second.write_bytes(b"second PDF")
            self.assertNotEqual(
                runner.page_cache_dir(root / "cache", first, 144),
                runner.page_cache_dir(root / "cache", second, 144),
            )

    def test_incomplete_cached_pages_are_re_rendered(self):
        class Pixmap:
            def save(self, target, **kwargs):
                Path(target).write_bytes(b"png")

        class Page:
            def __init__(self, document):
                self.document = document

            def get_pixmap(self, **kwargs):
                self.document.renders += 1
                return Pixmap()

        class Document:
            page_count = 2

            def __init__(self):
                self.renders = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter((Page(self), Page(self)))

        class Fitz:
            Matrix = staticmethod(lambda *args: args)

            def __init__(self, document):
                self.document = document

            def open(self, path):
                return self.document

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = root / "sample.pdf"
            pdf.write_bytes(b"pdf")
            pages = root / "pages"
            document = Document()
            with mock.patch.dict(sys.modules, {"fitz": Fitz(document)}):
                runner.render_pdf(pdf, pages, 72)
                (pages / "page-0002.png").unlink()
                runner.render_pdf(pdf, pages, 72)
            self.assertEqual(document.renders, 4)


if __name__ == "__main__":
    unittest.main()
