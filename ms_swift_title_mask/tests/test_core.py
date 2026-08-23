import unittest

from ms_swift_title_mask.core import (
    TitleMaskError,
    atx_heading_spans,
    build_sequence_loss_scale,
    canonicalize_target,
    response_token_weights,
    rswa_prefix_length,
)


def char_offsets(text):
    return [(index, index + 1) for index in range(len(text))]


class TitleMaskCoreTest(unittest.TestCase):

    def test_commonmark_atx_only_and_fenced_hash_is_body(self):
        text = "# Real title\n\nBody\n\n```python\n# not a heading\n```\n\nSetext\n======\n"
        spans = atx_heading_spans(text)
        self.assertEqual([(span.level, text[span.start:span.end]) for span in spans], [(1, "# Real title\n")])

        weights, _ = response_token_weights(text, char_offsets(text))
        real_hash = text.index("# Real title")
        code_hash = text.index("# not a heading")
        self.assertEqual(weights[real_hash], 1)
        self.assertEqual(weights[code_hash], 0)
        self.assertEqual(weights[text.index("Body")], 0)
        self.assertEqual(weights[text.index("\n", real_hash)], 1)

    def test_multiple_heading_levels(self):
        text = "intro\n\n## Section\nbody\n\n#### Detail\nmore\n"
        spans = atx_heading_spans(text)
        self.assertEqual([span.level for span in spans], [2, 4])

    def test_no_atx_heading_is_rejected(self):
        with self.assertRaisesRegex(TitleMaskError, "no CommonMark ATX heading"):
            response_token_weights("plain body\n", char_offsets("plain body\n"))

    def test_contentful_boundary_token_is_rejected(self):
        text = "# Heading\nBody\n"
        boundary = text.index("g\nB")
        offsets = [(0, boundary), (boundary, boundary + 3), (boundary + 3, len(text))]
        with self.assertRaisesRegex(TitleMaskError, "crosses heading/body"):
            response_token_weights(text, offsets)

    def test_boundary_token_with_only_outside_whitespace_is_title(self):
        text = "# Heading\n\nBody\n"
        offsets = [(0, len("# Heading\n\n")), (len("# Heading\n\n"), len(text))]
        weights, _ = response_token_weights(text, offsets)
        self.assertEqual(weights, [1, 0])

    def test_sequence_mask_changes_neither_ids_nor_labels(self):
        prefix = [10, 11, 12]
        response_ids = [20, 21, 22, 23]
        eos = [2]
        input_ids = prefix + response_ids + eos
        labels = [-100] * len(prefix) + response_ids + eos
        original_ids = list(input_ids)
        original_labels = list(labels)
        native_prefix = rswa_prefix_length(labels)

        scale = build_sequence_loss_scale(input_ids, labels, response_ids, [1, 1, 0, 0], eos)
        self.assertEqual(scale, [0, 0, 0, 1, 1, 0, 0, 1])
        self.assertEqual(input_ids, original_ids)
        self.assertEqual(labels, original_labels)
        self.assertEqual(rswa_prefix_length(labels), native_prefix)

    def test_response_token_mismatch_is_rejected(self):
        with self.assertRaisesRegex(TitleMaskError, "response token IDs"):
            build_sequence_loss_scale([1, 7, 8, 2], [-100, 7, 9, 2], [7, 8], [1, 0], [2])

    def test_canonical_target_has_exactly_one_final_newline(self):
        self.assertEqual(canonicalize_target("# T\r\n\r\nBody\r\n\r\n"), "# T\n\nBody\n")


if __name__ == "__main__":
    unittest.main()
