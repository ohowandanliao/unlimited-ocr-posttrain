import unittest
from pathlib import Path

from scripts.evaluation.analyze_outputs import analyze_row, extract_headings, match_ground_truth


class EvaluationAnalysisTest(unittest.TestCase):
    def test_headings_ignore_fenced_code(self):
        text = "# Real\n\n```python\n# not a heading\n```\n\n## Child\n"
        self.assertEqual(extract_headings(text), [(1, "real"), (2, "child")])

    def test_over_generation_is_severe_without_duplicate_lines(self):
        row = {
            "file": "sample.pdf",
            "model": "model",
            "pages": 1,
            "prompt": "<image>document parsing.",
            "response": {"text": "unique words " * 100, "output_tokens": 100},
        }
        result = analyze_row(
            row,
            Path("sample.md"),
            "# GT\nshort\n",
            max_output_tokens=32768,
            cap_ratio=0.95,
            severe_long_ratio=4.0,
            severe_short_ratio=0.10,
        )
        self.assertTrue(result["severe"])
        self.assertIn("over_generation", result["severe_reasons"])

    def test_ground_truth_can_match_unique_legacy_char_count(self):
        row = {"file": "localized-name.pdf", "gt_chars": 3}
        path, text = match_ground_truth(
            row,
            [(Path("gt__translated.md"), "abc"), (Path("other.md"), "longer")],
        )
        self.assertEqual(path, Path("gt__translated.md"))
        self.assertEqual(text, "abc")


if __name__ == "__main__":
    unittest.main()
