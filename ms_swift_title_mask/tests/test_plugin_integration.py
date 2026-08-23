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
            LOSS_NAME,
            TEMPLATE_NAME,
            UOCRTitleActiveMean,
        )
        from swift.callbacks import callbacks_map
        from swift.loss import loss_map
        from swift.template.register import TEMPLATE_MAPPING

        self.assertIn(TEMPLATE_NAME, TEMPLATE_MAPPING)
        self.assertIs(loss_map[LOSS_NAME], UOCRTitleActiveMean)
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


if __name__ == "__main__":
    unittest.main()
