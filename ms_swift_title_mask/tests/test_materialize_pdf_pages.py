import hashlib
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from ms_swift_title_mask.scripts.materialize_pdf_pages import _error_category, load_tasks, sha256_bytes


class MaterializePdfPagesTest(unittest.TestCase):
    def test_path_escape_error_is_not_mislabeled_as_page_mismatch(self):
        message = "image target escapes output root: /tmp/pages/page_0000.png"
        self.assertEqual(_error_category(message), "source_path")

    def test_load_tasks_rejects_path_escape_and_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = {"meta": {"source": "PMC-v26-synthetic", "doc_id": "pmc1", "split": "train", "sample_form": "strict_single_page", "n_pages": 1, "page_indices": [0], "source_pdf_sha256": "a" * 64}, "images": [str(root / ".." / "escape.png")]}
            path = root / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tasks([path], root / "images")

    def test_load_tasks_deduplicates_same_document_page(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / "images"
            base = {"meta": {"source": "PMC-v26-synthetic", "doc_id": "pmc1", "split": "train", "sample_form": "strict_single_page", "n_pages": 1, "page_indices": [0], "source_pdf_sha256": "a" * 64}, "images": [str(out / "pmc" / "pmc1" / "page_0000.png")]}
            path = root / "rows.jsonl"; path.write_text(json.dumps(base) + "\n" + json.dumps(base) + "\n", encoding="utf-8")
            manifest = {("PMC-v26-synthetic", "pmc1"): {"pdf_sha256": "a" * 64, "source_pdf": "documents/pmc1/document.pdf", "pdf_pages": 1}}
            tasks, _ = load_tasks([path], out, manifest)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["pages"], {0: (out / "pmc" / "pmc1" / "page_0000.png").resolve()})

    def test_sources_are_allowlisted_and_readoc_manifest_keys_are_normalized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / "images"
            row = {"meta": {"source": "READoc-arxiv", "doc_id": "1234", "split": "train", "sample_form": "strict_single_page", "n_pages": 1, "page_indices": [2], "source_pdf_sha256": "b" * 64}, "images": [str(out / "readoc" / "arxiv" / "1234" / "page_0002.png")]}
            path = root / "rows.jsonl"; path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest = {("READoc-arxiv", "1234"): {"pdf_sha256": "b" * 64, "pdf_pages": 3, "source_pdf_archive": "archives/arxiv.zip", "source_pdf_member": "arxiv/pdf/1234.pdf"}}
            tasks, _ = load_tasks([path], out, manifest)
            self.assertEqual(tasks[0]["source"], "READoc-arxiv")
            row["meta"]["source"] = "unknown"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError): load_tasks([path], out, manifest)

    def test_pixmap_save_is_explicitly_png(self):
        from ms_swift_title_mask.scripts import materialize_pdf_pages as module
        class Pix:
            def __init__(self): self.output = None
            def save(self, name, **kwargs): self.output = kwargs.get("output"); Path(name).write_bytes(b"png")
        pix = Pix()
        class Page:
            def get_pixmap(self, **kwargs): return pix
        class Pdf:
            def __len__(self): return 1
            def __getitem__(self, index): return Page()
            def close(self): pass
        class FitZ:
            Matrix = staticmethod(lambda *args: args)
            def open(self, **kwargs): return Pdf()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "images" / "pmc" / "d" / "page_0000.png"
            task = {"source": "PMC-v26-synthetic", "doc_id": "d", "pages": {0: target}, "pdf_sha256": "a" * 64, "pdf_pages": 1}
            with mock.patch.object(module, "_fitz", return_value=FitZ()), mock.patch.object(module, "_pdf_bytes", return_value=b"pdf"):
                module.materialize_task(task, readoc_root=None, pmc_root=None, manifest={}, dpi=72, overwrite=False)
            self.assertEqual(pix.output, "png")

    def test_dry_run_does_not_create_output_root(self):
        from ms_swift_title_mask.scripts import materialize_pdf_pages as module
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); output = root / "does-not-exist"; rows = root / "rows.jsonl"
            row = {"meta": {"source": "PMC-v26-synthetic", "doc_id": "d", "split": "test", "sample_form": "strict_single_page", "n_pages": 1, "page_indices": [0], "source_pdf_sha256": "a" * 64}, "images": [str(output / "pmc" / "d" / "page_0000.png")]}
            rows.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest = root / "pmc_manifest.jsonl"
            manifest.write_text(json.dumps({"doc_id": "d", "source_pdf": "documents/d/document.pdf", "pdf_sha256": "a" * 64, "pdf_pages": 1}) + "\n", encoding="utf-8")
            self.assertEqual(module.main(["--dry-run", "--pmc-manifest", str(manifest), "--output-root", str(output), str(rows)]), 0)
            self.assertFalse(output.exists())

    def test_sha256_bytes(self):
        self.assertEqual(sha256_bytes(b"abc"), hashlib.sha256(b"abc").hexdigest())

    def test_pmc_manifest_path_and_page_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / "images"
            manifest = {("PMC-v26-synthetic", "pmc1"): {"pdf_sha256": "a" * 64, "source_pdf": "documents/pmc1/document.pdf", "pdf_pages": 2}}
            row = {"meta": {"source": "PMC-v26-synthetic", "doc_id": "pmc1", "split": "train", "sample_form": "strict_single_page", "n_pages": 1, "page_indices": [2], "source_pdf_sha256": "a" * 64}, "images": [str(out / "pmc" / "pmc1" / "page_0002.png")]}
            path = root / "rows.jsonl"; path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            errors = []
            tasks, _ = load_tasks([path], out, manifest, errors)
            self.assertEqual(tasks, [])
            self.assertEqual(errors[0]["category"], "page_mismatch")

    def test_sample_form_n_pages_and_ordered_pages_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / "images"; path = root / "rows.jsonl"
            manifest = {("PMC-v26-synthetic", "pmc1"): {"pdf_sha256": "a" * 64, "source_pdf": "documents/pmc1/document.pdf", "pdf_pages": 3}}

            cases = [
                {"sample_form": "unknown", "n_pages": 1, "page_indices": [0], "message": "sample_form"},
                {"sample_form": "strict_single_page", "n_pages": 2, "page_indices": [0], "message": "n_pages"},
                {"sample_form": "strict_single_page", "n_pages": 2, "page_indices": [0, 1], "message": "strict_single_page"},
                {"sample_form": "full_document", "n_pages": 2, "page_indices": [1, 0], "message": "ordered"},
            ]
            for case in cases:
                row = {"meta": {"source": "PMC-v26-synthetic", "doc_id": "pmc1", "split": "train", "source_pdf_sha256": "a" * 64, **{key: case[key] for key in ("sample_form", "n_pages", "page_indices")}}, "images": [str(out / "pmc" / "pmc1" / f"page_{page:04d}.png") for page in case["page_indices"]]}
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                with self.subTest(message=case["message"]):
                    with self.assertRaisesRegex(ValueError, case["message"]):
                        load_tasks([path], out, manifest)

    def test_readoc_full_requires_manifest_page_range(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / "images"; path = root / "rows.jsonl"
            row = {"meta": {"source": "READoc-arxiv", "doc_id": "1234", "split": "train", "sample_form": "full_document", "n_pages": 2, "page_indices": [0, 1], "source_pdf_sha256": "b" * 64}, "images": [str(out / "readoc" / "arxiv" / "1234" / f"page_{page:04d}.png") for page in (0, 1)]}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest = {("READoc-arxiv", "1234"): {"pdf_sha256": "b" * 64, "pdf_pages": 3, "source_pdf_archive": "archives/arxiv.zip", "source_pdf_member": "arxiv/pdf/1234.pdf"}}
            with self.assertRaisesRegex(ValueError, "full page_indices"):
                load_tasks([path], out, manifest)

    def test_actual_pdf_page_count_must_match_manifest(self):
        from ms_swift_title_mask.scripts import materialize_pdf_pages as module

        class Pdf:
            def __len__(self): return 1
            def close(self): pass

        class FitZ:
            def open(self, **kwargs): return Pdf()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "images" / "pmc" / "d" / "page_0000.png"
            task = {"source": "PMC-v26-synthetic", "doc_id": "d", "pages": {0: target}, "pdf_sha256": "a" * 64, "pdf_pages": 2}
            with mock.patch.object(module, "_fitz", return_value=FitZ()), mock.patch.object(module, "_pdf_bytes", return_value=b"pdf"):
                with self.assertRaisesRegex(ValueError, "manifest declares 2 pages"):
                    module.materialize_task(task, readoc_root=None, pmc_root=None, manifest={}, dpi=72, overwrite=False)

    def test_validation_accumulates_sha_and_missing_manifest_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / "images"
            rows = root / "rows.jsonl"
            records = [
                {"meta": {"source": "PMC-v26-synthetic", "doc_id": "missing", "split": "train", "sample_form": "strict_single_page", "n_pages": 1, "page_indices": [0], "source_pdf_sha256": "a" * 64}, "images": [str(out / "pmc" / "missing" / "page_0000.png")]},
                {"meta": {"source": "PMC-v26-synthetic", "doc_id": "sha", "split": "train", "sample_form": "strict_single_page", "n_pages": 1, "page_indices": [0], "source_pdf_sha256": "b" * 64}, "images": [str(out / "pmc" / "sha" / "page_0000.png")]},
            ]
            rows.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            manifest = {("PMC-v26-synthetic", "sha"): {"pdf_sha256": "c" * 64, "source_pdf": "documents/sha/document.pdf", "pdf_pages": 1}}
            errors = []
            load_tasks([rows], out, manifest, errors)
            self.assertEqual(len(errors), 2)
            self.assertEqual({item["category"] for item in errors}, {"missing", "sha_mismatch"})

    @unittest.skipUnless(importlib.util.find_spec("fitz"), "PyMuPDF is not installed")
    def test_small_pdf_render(self):
        import fitz
        from ms_swift_title_mask.scripts.materialize_pdf_pages import materialize_task

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); pdf_path = root / "documents" / "doc1" / "document.pdf"
            pdf_path.parent.mkdir(parents=True)
            doc = fitz.open(); doc.new_page(); doc.save(pdf_path)
            sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            target = (root / "images" / "p0000.png").resolve()
            task = {"source": "PMC-v26-synthetic", "doc_id": "doc1", "pages": {0: target}, "pdf_sha256": sha, "pdf_pages": 1}
            manifest = {("PMC-v26-synthetic", "doc1"): {"source_pdf": "documents/doc1/document.pdf"}}
            result = materialize_task(task, readoc_root=None, pmc_root=root, manifest=manifest, dpi=72, overwrite=False)
            self.assertEqual(result["pages"], 1)
            self.assertTrue(target.is_file())


if __name__ == "__main__":
    unittest.main()
