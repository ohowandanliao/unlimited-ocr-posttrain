import unittest

from uocr_train.training_utils import SequenceLengthGuard


class SequenceLengthGuardTest(unittest.TestCase):
    def test_accepts_at_limit_and_tracks_stats(self):
        guard = SequenceLengthGuard(max_length=8, strategy="error")

        self.assertTrue(guard.check("sample-ok", 8))
        self.assertEqual(guard.encoded, 1)
        self.assertEqual(guard.max_observed, 8)
        self.assertEqual(guard.over_limit, 0)

    def test_error_policy_reports_sample_and_lengths(self):
        guard = SequenceLengthGuard(max_length=8, strategy="error")

        with self.assertRaisesRegex(ValueError, "sample-long.*9 tokens.*max_length=8"):
            guard.check("sample-long", 9)

    def test_drop_policy_counts_dropped_and_unique_samples(self):
        guard = SequenceLengthGuard(max_length=8, strategy="drop")

        self.assertFalse(guard.check("sample-long", 9))
        self.assertFalse(guard.check("sample-long", 10))
        self.assertEqual(guard.dropped, 2)
        self.assertEqual(guard.over_limit, 2)
        self.assertEqual(len(guard.over_limit_ids), 1)

    def test_rejects_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "max_length must be positive"):
            SequenceLengthGuard(max_length=0)
        with self.assertRaisesRegex(ValueError, "length_strategy"):
            SequenceLengthGuard(max_length=8, strategy="truncate")


if __name__ == "__main__":
    unittest.main()
