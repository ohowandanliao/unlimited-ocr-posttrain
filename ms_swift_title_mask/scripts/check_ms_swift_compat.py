#!/usr/bin/env python3
"""Check the exact ms-swift interfaces used by this external plugin."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import runpy
import subprocess
import sys
from pathlib import Path


TESTED_MS_SWIFT_COMMIT = "1a1ba3ee86488af323ef9b64ca3d34edee90ab11"


def require_text(path: Path, fragments):
    text = path.read_text(encoding="utf-8")
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise RuntimeError(f"{path} no longer satisfies the tested contract; missing {missing}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-swift-root", type=Path, required=True)
    parser.add_argument("--model")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.ms_swift_root.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != TESTED_MS_SWIFT_COMMIT and os.environ.get("ALLOW_UNTESTED_MS_SWIFT") != "1":
        raise RuntimeError(
            f"ms-swift commit is {commit}, tested {TESTED_MS_SWIFT_COMMIT}; "
            "review the diff before setting ALLOW_UNTESTED_MS_SWIFT=1"
        )

    require_text(
        root / "swift/template/templates/deepseek.py",
        ["class UnlimitedOCR", "_build_rswa_attention_mask", "MLLMTemplateType.unlimited_ocr"],
    )
    require_text(
        root / "swift/trainers/seq2seq_trainer.py",
        [
            "loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1).view(-1)",
            "outputs.loss = outputs.loss * loss_scale",
            "compute_loss_func(",
        ],
    )
    require_text(root / "swift/loss/mapping.py", ["loss_map = {"])

    sys.path.insert(0, str(root))
    plugin = Path(__file__).resolve().parents[1] / "plugin/uocr_title_mask.py"
    runpy.run_path(str(plugin), run_name="uocr_title_mask_compat_check")
    from swift.callbacks import callbacks_map
    from swift.loss import loss_map
    from swift.template.register import TEMPLATE_MAPPING

    assert "unlimited_ocr_title_mask" in TEMPLATE_MAPPING
    assert "uocr_title_active_mean" in loss_map
    assert "uocr_lora_target_guard" in callbacks_map

    transformers_version = importlib.metadata.version("transformers")
    if transformers_version != "4.46.3":
        raise RuntimeError(
            f"this ms-swift Unlimited-OCR registration requires transformers==4.46.3; got {transformers_version}"
        )
    if args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("Unlimited-OCR tokenizer is slow and cannot provide safe title offsets")
        probe = tokenizer("# Title\n\nBody\n", add_special_tokens=False, return_offsets_mapping=True)
        if len(probe["input_ids"]) != len(probe["offset_mapping"]):
            raise RuntimeError("Unlimited-OCR tokenizer returned invalid offset_mapping")

    print(
        f"compat_ok ms_swift={commit} transformers={transformers_version} "
        f"markdown_it={importlib.metadata.version('markdown-it-py')}"
    )


if __name__ == "__main__":
    main()
