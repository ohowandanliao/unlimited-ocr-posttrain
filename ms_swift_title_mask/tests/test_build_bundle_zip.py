import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from ms_swift_title_mask.scripts import build_bundle_zip as bundle


class BundleZipContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "bundle"
        self.data = self.root / "data_assets"
        self.data.mkdir(parents=True)
        (self.data / "STATUS_ZH.md").write_text("status\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _minimal_package_files(self):
        for relative in (
            "final_mix_v1/mix_report.json",
            "final_mix_v1/train.jsonl",
            "final_mix_v1/validation.jsonl",
            "final_mix_v1/test.jsonl",
            "readoc_silver_v1/readoc_full/train.jsonl",
            "readoc_silver_v1/readoc_full/validation.jsonl",
            "readoc_silver_v1/readoc_full/test.jsonl",
            "pmc_silver_v2/pmc_full/train.jsonl",
            "pmc_silver_v2/pmc_full/validation.jsonl",
            "pmc_silver_v2/pmc_full/test.jsonl",
            "pmc_silver_v2/pmc_single/train.jsonl",
            "pmc_silver_v2/pmc_single/validation.jsonl",
            "pmc_silver_v2/pmc_single/test.jsonl",
        ):
            path = self.data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        (self.root / "scripts").mkdir()
        (self.root / "scripts/build_bundle_zip.py").write_text("# test\n", encoding="utf-8")

    def test_real_assets_close_gt_jsonl_and_pdf_manifest_contract(self):
        if not (bundle.DATA / "final_mix_v1/train.jsonl").is_file():
            self.skipTest("ignored training payload is not present")
        counts = bundle.validate_contract()
        self.assertEqual(counts["standalone_gt"], 4524)
        self.assertEqual(counts["unique_pdf_docs"], 3038)
        self.assertEqual(counts["train"] + counts["validation"] + counts["test"], 4524)

    def test_checksum_parser_rejects_unsafe_or_duplicate_paths(self):
        digest = "a" * 64
        for content in (
            f"{digest}  ../escape\n",
            f"{digest}  /absolute\n",
            f"{digest}  payload\n{digest}  payload\n",
            "not-a-sha  payload\n",
        ):
            with self.subTest(content=content), self.assertRaises(RuntimeError):
                bundle.parse_checksums(content)

    def test_package_rechecks_every_zipped_payload_checksum(self):
        self._minimal_package_files()
        payload = self.data / "payload.txt"
        payload.write_text("payload\n", encoding="utf-8")
        output = Path(self.temp.name) / "bundle.zip"
        with patch.object(bundle, "ROOT", self.root), patch.object(bundle, "DATA", self.data):
            checksum_file = bundle.write_checksums()
            digest, _size, _files = bundle.package(output)
            self.assertEqual(digest, hashlib.sha256(output.read_bytes()).hexdigest())
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    bundle.verify_archive_checksums(archive),
                    len(checksum_file.read_text(encoding="utf-8").splitlines()),
                )

    def test_archive_verification_fails_on_tampered_payload(self):
        archive_path = Path(self.temp.name) / "tampered.zip"
        digest = hashlib.sha256(b"expected").hexdigest()
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("ms_swift_title_mask/data_assets/DATA_SHA256SUMS", f"{digest}  payload.txt\n")
            archive.writestr("ms_swift_title_mask/data_assets/payload.txt", b"tampered")
        with zipfile.ZipFile(archive_path) as archive, self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            bundle.verify_archive_checksums(archive)


if __name__ == "__main__":
    unittest.main()
