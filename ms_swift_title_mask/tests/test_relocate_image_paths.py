import copy
import json
import tempfile
import unittest
from pathlib import Path

from ms_swift_title_mask.core import sha256_text
from ms_swift_title_mask.scripts.relocate_image_paths import build_relocated_dataset


class RelocateImagePathsTest(unittest.TestCase):
    def row(self, split, source="READoc-arxiv", doc_id="doc-1", page=0):
        target = "# Title\n\nBody.\n"
        family = "pmc" if source.startswith("PMC") else "readoc/" + source.removeprefix("READoc-")
        old = f"/old/root/{family}/{doc_id}/page_{page:04d}.png"
        return {
            "id": f"{source}-{split}-{doc_id}",
            "channel": "title_reviewed",
            "messages": [{"role": "user", "content": "<image>document parsing."},
                         {"role": "assistant", "content": target}],
            "images": [old],
            "meta": {
                "source": source, "doc_id": doc_id, "sample_form": "strict_single_page",
                "page_indices": [page], "n_pages": 1, "split": split,
                "title_review_status": "SILVER_ACCEPTED", "title_review_id": "r",
                "title_review_version": "v1", "title_reviewer": "test",
                "title_reviewed_at": "2026-01-01T00:00:00+00:00",
                "title_target_sha256": sha256_text(target), "title_heading_count": 1,
                "title_ruleset_version": "v1", "title_source_sha256": "a" * 64,
            },
        }

    def make_input(self, root):
        rows = {
            "train": self.row("train"),
            "validation": self.row("validation", "PMC-v26-synthetic", "pmc-1"),
            "test": self.row("test", "READoc-github", "gh-1"),
        }
        root.mkdir()
        for split, row in rows.items():
            (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
        return rows

    def test_three_splits_any_root_preserve_input_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            rows = self.make_input(root)
            before = {p: p.read_bytes() for p in root.glob("*.jsonl")}
            output = Path(tmp) / "relocated"
            report = build_relocated_dataset(root, output, Path(tmp) / "server" / "pages")
            self.assertEqual(report["rows"], 3)
            for split, original in rows.items():
                relocated = json.loads((output / f"{split}.jsonl").read_text())
                self.assertEqual(relocated["messages"], original["messages"])
                self.assertEqual(relocated["meta"], original["meta"])
                self.assertEqual(relocated["images"][0].split("/server/pages/")[-1],
                                 relocated["images"][0].split("/server/pages/")[-1])
                self.assertEqual(before[root / f"{split}.jsonl"], (root / f"{split}.jsonl").read_bytes())

    def test_rejects_bad_source_path_and_nested_or_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            rows = self.make_input(root)
            bad = copy.deepcopy(rows["train"])
            bad["images"] = ["/wrong/doc/page_0000.png"]
            (root / "train.jsonl").write_text(json.dumps(bad) + "\n")
            with self.assertRaises(ValueError):
                build_relocated_dataset(root, Path(tmp) / "out", Path(tmp) / "pages")
            with self.assertRaises(ValueError):
                build_relocated_dataset(root, root / "nested", Path(tmp) / "pages")
            output = Path(tmp) / "out"
            output.mkdir()
            with self.assertRaises(ValueError):
                build_relocated_dataset(root, output, Path(tmp) / "pages")

    def test_rejects_path_escape_doc_id_and_relative_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            rows = self.make_input(root)
            bad = copy.deepcopy(rows["train"])
            bad["meta"]["doc_id"] = "../escape"
            (root / "train.jsonl").write_text(json.dumps(bad) + "\n")
            with self.assertRaises(ValueError):
                build_relocated_dataset(root, Path(tmp) / "out", Path(tmp) / "pages")


if __name__ == "__main__":
    unittest.main()
