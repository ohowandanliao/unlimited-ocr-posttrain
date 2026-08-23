"""ms-swift external plugin for reviewed Unlimited-OCR title supervision."""

from __future__ import annotations

import sys
import re
from copy import deepcopy
from pathlib import Path

import torch

# `--external_plugins` imports this file directly, so make the bundle importable
# even when ms-swift was launched outside its repository root.
_BUNDLE_PARENT = Path(__file__).resolve().parents[2]
if str(_BUNDLE_PARENT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_PARENT))

from ms_swift_title_mask.core import (  # noqa: E402
    EXPECTED_CHANNEL,
    TitleMaskError,
    build_sequence_loss_scale,
    response_token_weights,
    validate_conversation,
)
from swift.callbacks import TrainerCallback, callbacks_map  # noqa: E402
from swift.loss import BaseLoss, loss_map  # noqa: E402
from swift.template import register_template  # noqa: E402
from swift.template.register import TEMPLATE_MAPPING  # noqa: E402
from swift.template.templates.deepseek import UnlimitedOCR  # noqa: E402
from swift.utils import get_logger  # noqa: E402


TEMPLATE_NAME = "unlimited_ocr_title_mask"
LOSS_NAME = "uocr_title_active_mean"
CALLBACK_NAME = "uocr_lora_target_guard"
logger = get_logger()

_TARGET_RE = re.compile(
    r"^model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj)"
    r"|mlp\.shared_experts\.(gate_proj|up_proj|down_proj))$"
)


def _tokenize_response(tokenizer, response: str):
    if not getattr(tokenizer, "is_fast", False):
        raise TitleMaskError("title mask requires a fast tokenizer with offset_mapping")
    tokenized = tokenizer(
        response,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in tokenized:
        raise TitleMaskError("tokenizer did not return offset_mapping")
    response_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]
    if response_ids and isinstance(response_ids[0], list):
        raise TitleMaskError("tokenizer unexpectedly returned a batched response")
    if len(response_ids) != len(offsets):
        raise TitleMaskError("response token IDs and offsets have different lengths")
    return list(response_ids), list(offsets)


class ReviewedTitleMaskUnlimitedOCR(UnlimitedOCR):
    """Keep native tokens/labels/R-SWA and replace only the independent loss scale."""

    def _encode(self, inputs):
        if not self.is_training:
            raise TitleMaskError(f"{TEMPLATE_NAME} is training-only; use unlimited_ocr for inference")
        if inputs.channel != EXPECTED_CHANNEL:
            raise TitleMaskError(
                f"{TEMPLATE_NAME} only accepts channel={EXPECTED_CHANNEL!r}; got {inputs.channel!r}"
            )
        if inputs.system not in (None, ""):
            raise TitleMaskError("reviewed title training does not accept a system prompt")

        response = validate_conversation(inputs.messages, image_count=len(inputs.images or []))
        encoded = super()._encode(inputs)
        input_ids = encoded.get("input_ids")
        labels = encoded.get("labels")
        if input_ids is None or labels is None:
            raise TitleMaskError("native Unlimited-OCR template did not produce input_ids and labels")

        response_ids, offsets = _tokenize_response(self.tokenizer, response)
        response_weights, _ = response_token_weights(response, offsets)
        suffix_ids = self._encode_context_list(self.template_meta.suffix)[0]
        if suffix_ids != [self.tokenizer.eos_token_id]:
            raise TitleMaskError(
                f"tested contract requires exactly one EOS suffix; got {suffix_ids!r}"
            )
        encoded["loss_scale"] = build_sequence_loss_scale(
            input_ids,
            labels,
            response_ids,
            response_weights,
            suffix_ids,
        )
        return encoded


class UOCRTitleActiveMean(BaseLoss):
    """Global active-token mean for loss already weighted by Seq2SeqTrainer."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        if getattr(args, "average_tokens_across_devices", False):
            raise ValueError(f"{LOSS_NAME} requires --average_tokens_across_devices false")
        if getattr(args, "use_liger_kernel", False):
            raise ValueError(f"{LOSS_NAME} requires --use_liger_kernel false")
        if getattr(args, "enable_dft_loss", False):
            raise ValueError(f"{LOSS_NAME} is not compatible with DFT loss")
        if getattr(trainer.template, "sequence_parallel_size", 1) != 1:
            raise ValueError(f"{LOSS_NAME} currently supports sequence_parallel_size=1 only")

    def __call__(self, outputs, labels, *, loss_scale=None, **kwargs):
        if loss_scale is None or outputs.loss is None:
            raise RuntimeError(f"{LOSS_NAME} requires aligned loss_scale and per-token loss")
        if loss_scale.ndim != 1 or outputs.loss.ndim != 1 or loss_scale.numel() != outputs.loss.numel():
            raise RuntimeError("aligned loss_scale and per-token loss must be equal-length vectors")
        if not torch.all((loss_scale == 0) | (loss_scale == 1)):
            raise RuntimeError("title loss_scale must remain binary 0/1")

        aligned_labels = torch.roll(labels, shifts=-1, dims=-1).reshape(-1).to(loss_scale.device)
        valid = aligned_labels != -100
        active = (loss_scale > 0) & valid
        local_denom = loss_scale[valid].sum().detach().to(outputs.loss.device)
        if local_denom.item() <= 0:
            raise RuntimeError("local batch has no active title/EOS token")

        self._update_metrics(outputs, aligned_labels, active, valid)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_denom = local_denom.clone()
            torch.distributed.all_reduce(global_denom, op=torch.distributed.ReduceOp.SUM)
            if global_denom.item() <= 0:
                raise RuntimeError("distributed batch has no active title/EOS token")
            world_size = torch.distributed.get_world_size()
            return outputs.loss.sum() * world_size / global_denom
        return outputs.loss.sum() / local_denom

    @torch.no_grad()
    def _update_metrics(self, outputs, aligned_labels, active, valid):
        mode = "train" if self.trainer.model.training else "eval"
        metrics = self.trainer.custom_metrics[mode]
        eos_token_id = self.trainer.template.tokenizer.eos_token_id
        title = active & (aligned_labels != eos_token_id)
        masked_body = valid & ~active

        metrics["active_title_tokens"].update(title.sum())
        metrics["active_eos_tokens"].update((active & ~title).sum())
        metrics["masked_body_tokens"].update(masked_body.sum())
        if title.any():
            metrics["title_token_loss"].update(outputs.loss[title])
            predictions = outputs.logits.argmax(dim=-1).reshape(-1).to(aligned_labels.device)
            metrics["title_token_acc"].update((predictions[title] == aligned_labels[title]).float())


class UOCRLoRATargetGuard(TrainerCallback):
    """Fail before step 1 if ms-swift applies LoRA outside the reviewed target set."""

    @staticmethod
    def _canonical_name(name: str):
        marker = "model.layers."
        index = name.find(marker)
        return name[index:] if index >= 0 else name

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            raise RuntimeError(f"{CALLBACK_NAME} did not receive the model")
        targets = set()
        unexpected = []
        for name, module in model.named_modules():
            lora_a = getattr(module, "lora_A", None)
            if lora_a is None or len(lora_a) == 0:
                continue
            canonical = self._canonical_name(name)
            if not _TARGET_RE.fullmatch(canonical):
                unexpected.append(canonical)
            targets.add(canonical)
        if unexpected:
            raise RuntimeError(f"LoRA was attached outside the approved decoder target set: {unexpected[:10]}")

        counts = {
            "attn": sum(".self_attn." in name for name in targets),
            "dense_mlp": sum(".mlp." in name and ".shared_experts." not in name for name in targets),
            "shared_experts": sum(".shared_experts." in name for name in targets),
            "routed_experts": sum(".experts." in name and ".shared_experts." not in name for name in targets),
        }
        expected = {
            "attn": 48,
            "dense_mlp": 3,
            "shared_experts": 33,
            "routed_experts": 0,
        }
        if len(targets) != 84 or counts != expected:
            raise RuntimeError(f"LoRA target contract changed: total={len(targets)}, counts={counts}, expected={expected}")
        logger.info(f"[UOCR title] verified 84 LoRA modules: {counts}")


def _register():
    if "unlimited_ocr" not in TEMPLATE_MAPPING:
        raise RuntimeError("native unlimited_ocr template must be registered before this plugin")
    if TEMPLATE_NAME not in TEMPLATE_MAPPING:
        meta = deepcopy(TEMPLATE_MAPPING["unlimited_ocr"])
        meta.template_type = TEMPLATE_NAME
        meta.template_cls = ReviewedTitleMaskUnlimitedOCR
        register_template(meta)
    elif TEMPLATE_MAPPING[TEMPLATE_NAME].template_cls is not ReviewedTitleMaskUnlimitedOCR:
        raise RuntimeError(f"template name collision: {TEMPLATE_NAME}")

    existing_loss = loss_map.get(LOSS_NAME)
    if existing_loss not in (None, UOCRTitleActiveMean):
        raise RuntimeError(f"loss name collision: {LOSS_NAME}")
    loss_map[LOSS_NAME] = UOCRTitleActiveMean

    existing_callback = callbacks_map.get(CALLBACK_NAME)
    if existing_callback not in (None, UOCRLoRATargetGuard):
        raise RuntimeError(f"callback name collision: {CALLBACK_NAME}")
    callbacks_map[CALLBACK_NAME] = UOCRLoRATargetGuard


_register()
