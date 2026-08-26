import importlib.util
import types
import unittest
from collections import defaultdict


RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "swift"))


class _MetricRecorder:

    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(value)


@unittest.skipUnless(RUNTIME_AVAILABLE, "requires the server ms-swift runtime")
class PluginIntegrationTest(unittest.TestCase):

    def test_registration_and_single_rank_active_mean(self):
        import torch

        from ms_swift_title_mask.plugin.uocr_title_mask import (
            CALLBACK_NAME,
            LEGACY_WEIGHTED_LOSS_NAME,
            LOSS_NAME,
            UNIFORM_LOSS_NAME,
            UNIFORM_TEMPLATE_NAME,
            WEIGHTED_LOSS_NAME,
            TEMPLATE_NAME,
            WEIGHTED_TEMPLATE_NAME,
            UOCRTitleActiveMean,
            UOCRLegacyTitleWeightedMean,
            UOCRTitleWeightedMean,
            UOCRUniformMean,
        )
        from swift.callbacks import callbacks_map
        from swift.loss import loss_map
        from swift.template.register import TEMPLATE_MAPPING

        self.assertIn(TEMPLATE_NAME, TEMPLATE_MAPPING)
        self.assertIs(loss_map[LOSS_NAME], UOCRTitleActiveMean)
        self.assertIn(WEIGHTED_TEMPLATE_NAME, TEMPLATE_MAPPING)
        self.assertIs(loss_map[LEGACY_WEIGHTED_LOSS_NAME], UOCRLegacyTitleWeightedMean)
        self.assertIs(loss_map[WEIGHTED_LOSS_NAME], UOCRTitleWeightedMean)
        self.assertIn(UNIFORM_TEMPLATE_NAME, TEMPLATE_MAPPING)
        self.assertIs(loss_map[UNIFORM_LOSS_NAME], UOCRUniformMean)
        self.assertIn(CALLBACK_NAME, callbacks_map)

        args = types.SimpleNamespace(
            average_tokens_across_devices=False,
            use_liger_kernel=False,
            enable_dft_loss=False,
        )
        metrics = {
            "train": defaultdict(_MetricRecorder),
            "eval": defaultdict(_MetricRecorder),
        }
        trainer = types.SimpleNamespace(
            template=types.SimpleNamespace(
                sequence_parallel_size=1,
                tokenizer=types.SimpleNamespace(eos_token_id=2),
            ),
            model=types.SimpleNamespace(training=True),
            custom_metrics=metrics,
        )
        loss_fn = UOCRTitleActiveMean(args, trainer)
        outputs = types.SimpleNamespace(
            loss=torch.tensor([0.0, 4.0, 6.0, 0.0], requires_grad=True),
            logits=torch.tensor(
                [[[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0]]]
            ),
        )
        labels = torch.tensor([[-100, 5, 6, 2]])
        aligned_loss_scale = torch.tensor([0.0, 1.0, 1.0, 0.0])
        loss = loss_fn(outputs, labels, loss_scale=aligned_loss_scale)
        self.assertEqual(float(loss), 5.0)
        self.assertEqual(int(metrics["train"]["active_title_tokens"].values[0]), 1)
        self.assertEqual(int(metrics["train"]["active_eos_tokens"].values[0]), 1)


    def test_weighted_mean_uses_native_accumulation_denominator(self):
        import torch

        from ms_swift_title_mask.plugin.uocr_title_mask import (
            TRUSTED_TITLE_WEIGHT,
            UOCRTitleWeightedMean,
        )

        args = types.SimpleNamespace(
            average_tokens_across_devices=False,
            use_liger_kernel=False,
            enable_dft_loss=False,
        )
        metrics = {
            "train": defaultdict(_MetricRecorder),
            "eval": defaultdict(_MetricRecorder),
        }
        trainer = types.SimpleNamespace(
            template=types.SimpleNamespace(
                sequence_parallel_size=1,
                tokenizer=types.SimpleNamespace(eos_token_id=2),
            ),
            model=types.SimpleNamespace(training=True),
            custom_metrics=metrics,
        )
        loss_fn = UOCRTitleWeightedMean(args, trainer)
        outputs = types.SimpleNamespace(
            loss=torch.tensor(
                [0.0, 4.0 * TRUSTED_TITLE_WEIGHT, 6.0, 0.0],
                requires_grad=True,
            ),
            logits=torch.zeros((1, 4, 3)),
        )
        labels = torch.tensor([[-100, 5, 6, 2]])
        aligned_loss_scale = torch.tensor([TRUSTED_TITLE_WEIGHT, 1.0, 1.0, 0.0])
        loss = loss_fn(
            outputs,
            labels,
            num_items_in_batch=torch.tensor(6),
            loss_scale=aligned_loss_scale,
        )
        self.assertAlmostEqual(float(loss), (4.0 * TRUSTED_TITLE_WEIGHT + 6.0) / 6.0)
        self.assertEqual(int(metrics["train"]["active_title_tokens"].values[0]), 1)
        self.assertEqual(int(metrics["train"]["active_body_tokens"].values[0]), 1)
        self.assertEqual(int(metrics["train"]["active_eos_tokens"].values[0]), 1)
        self.assertAlmostEqual(
            float(metrics["train"]["active_weight_sum"].values[0]),
            TRUSTED_TITLE_WEIGHT + 2.0,
        )

    def test_legacy_weighted_mean_uses_local_active_weight_sum(self):
        import torch

        from ms_swift_title_mask.plugin.uocr_title_mask import (
            TRUSTED_TITLE_WEIGHT,
            UOCRLegacyTitleWeightedMean,
        )

        args = types.SimpleNamespace(
            average_tokens_across_devices=False,
            use_liger_kernel=False,
            enable_dft_loss=False,
        )
        metrics = {
            "train": defaultdict(_MetricRecorder),
            "eval": defaultdict(_MetricRecorder),
        }
        trainer = types.SimpleNamespace(
            template=types.SimpleNamespace(
                sequence_parallel_size=1,
                tokenizer=types.SimpleNamespace(eos_token_id=2),
            ),
            model=types.SimpleNamespace(training=True),
            custom_metrics=metrics,
        )
        loss_fn = UOCRLegacyTitleWeightedMean(args, trainer)
        outputs = types.SimpleNamespace(
            loss=torch.tensor(
                [4.0 * TRUSTED_TITLE_WEIGHT, 6.0, 2.0, 0.0],
                requires_grad=True,
            ),
            logits=torch.zeros((1, 4, 3)),
        )
        labels = torch.tensor([[-100, 5, 6, 2]])
        aligned_loss_scale = torch.tensor([TRUSTED_TITLE_WEIGHT, 1.0, 1.0, 0.0])
        loss = loss_fn(
            outputs,
            labels,
            num_items_in_batch=torch.tensor(99),
            loss_scale=aligned_loss_scale,
        )
        expected = (4.0 * TRUSTED_TITLE_WEIGHT + 8.0) / (TRUSTED_TITLE_WEIGHT + 2.0)
        self.assertAlmostEqual(float(loss), expected)

    def test_uniform_mean_tracks_all_non_eos_tokens_as_body(self):
        import torch

        from ms_swift_title_mask.plugin.uocr_title_mask import UOCRUniformMean

        args = types.SimpleNamespace(
            average_tokens_across_devices=False,
            use_liger_kernel=False,
            enable_dft_loss=False,
        )
        metrics = {
            "train": defaultdict(_MetricRecorder),
            "eval": defaultdict(_MetricRecorder),
        }
        trainer = types.SimpleNamespace(
            template=types.SimpleNamespace(
                sequence_parallel_size=1,
                tokenizer=types.SimpleNamespace(eos_token_id=2),
            ),
            model=types.SimpleNamespace(training=True),
            custom_metrics=metrics,
        )
        loss_fn = UOCRUniformMean(args, trainer)
        outputs = types.SimpleNamespace(
            loss=torch.tensor([0.0, 4.0, 6.0, 0.0], requires_grad=True),
            logits=torch.zeros((1, 4, 3)),
        )
        labels = torch.tensor([[-100, 5, 6, 2]])
        aligned_loss_scale = torch.tensor([1.0, 1.0, 1.0, 0.0])
        loss = loss_fn(
            outputs,
            labels,
            num_items_in_batch=torch.tensor(6),
            loss_scale=aligned_loss_scale,
        )
        self.assertAlmostEqual(float(loss), 10.0 / 6.0)
        self.assertEqual(int(metrics["train"]["active_title_tokens"].values[0]), 0)
        self.assertEqual(int(metrics["train"]["active_body_tokens"].values[0]), 2)
        self.assertEqual(int(metrics["train"]["active_eos_tokens"].values[0]), 1)


if __name__ == "__main__":
    unittest.main()
