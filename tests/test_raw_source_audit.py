import unittest
from pathlib import Path

from scripts.data.audit_raw_sources import markdown_profile, pmc_middle_profile


class RawSourceAuditTest(unittest.TestCase):
    def test_markdown_profile_uses_commonmark_headings(self):
        profile = markdown_profile("# Title\n\n```\n# code\n```\n")
        self.assertEqual(profile["headings"], 1)
        self.assertEqual(profile["fence_markers"], 2)
        self.assertTrue(profile["ends_with_newline"])

    def test_pmc_profile_counts_each_nested_asset_once(self):
        payload = {
            "_backend": "vlm",
            "_version_name": "3.1.14",
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [595, 842],
                    "para_blocks": [
                        {
                            "type": "image",
                            "blocks": [
                                {
                                    "type": "image_body",
                                    "lines": [
                                        {"spans": [{"type": "image", "image_path": "missing.png"}]}
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        profile = pmc_middle_profile(payload, Path("/definitely/not/a/source/root"))
        self.assertEqual(profile["image_asset_references"], 1)
        self.assertEqual(profile["unique_image_assets"], 1)
        self.assertEqual(profile["missing_unique_image_assets"], 1)


if __name__ == "__main__":
    unittest.main()
