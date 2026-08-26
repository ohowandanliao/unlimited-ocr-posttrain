"""ms-swift external plugin for reviewed Unlimited-OCR title supervision."""

from __future__ import annotations

import math
import os
import re
import sys
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
    MULTI_PAGE_PROMPT,
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
WEIGHTED_TEMPLATE_NAME = "unlimited_ocr_title_weighted"
UNIFORM_TEMPLATE_NAME = "unlimited_ocr_uniform_ce"
LOSS_NAME = "uocr_title_active_mean"
LEGACY_WEIGHTED_LOSS_NAME = "uocr_title_weighted_mean"
WEIGHTED_LOSS_NAME = "uocr_title_weighted_token_mean"
UNIFORM_LOSS_NAME = "uocr_uniform_token_mean"
CALLBACK_NAME = "uocr_lora_target_guard"
logger = get_logger()

_TARGET_RE = re.compile(
    r"^model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj)"
    r"|mlp\.shared_experts\.(gate_proj|up_proj|down_proj))$"
)


def _environment_weight(name: str, default: float, *, greater_than: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a finite number; got {raw!r}") from exc
    if not math.isfinite(value) or value <= greater_than:
        raise RuntimeError(f"{name} must be greater than {greater_than}; got {raw!r}")
    return value


TRUSTED_TITLE_WEIGHT = _environment_weight("UOCR_TITLE_WEIGHT", 1.5, greater_than=1.0)
TRUSTED_EOS_WEIGHT = _environment_weight("UOCR_EOS_WEIGHT", 1.0, greater_than=0.0)


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

    template_name = TEMPLATE_NAME
    body_weight = 0.0
    title_weight = 1.0
    eos_weight = 1.0

    def _encode(self, inputs):
        if not self.is_training:
            raise TitleMaskError(f"{self.template_name} is training-only; use unlimited_ocr for inference")
        if inputs.channel != EXPECTED_CHANNEL:
            raise TitleMaskError(
                f"{self.template_name} only accepts channel={EXPECTED_CHANNEL!r}; got {inputs.channel!r}"
            )
        if inputs.system not in (None, ""):
            raise TitleMaskError("reviewed title training does not accept a system prompt")

        runtime_messages = inputs.messages
        image_count = len(inputs.images or [])
        if image_count > 1:
            runtime_messages = deepcopy(inputs.messages)
            runtime_prompt = runtime_messages[0].get("content")
            expanded_prompt = "<image>" * image_count + MULTI_PAGE_PROMPT.removeprefix("<image>")
            if runtime_prompt == expanded_prompt:
                runtime_messages[0]["content"] = MULTI_PAGE_PROMPT
        response = validate_conversation(runtime_messages, image_count=image_count)
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
            body_weight=self.body_weight,
            title_weight=self.title_weight,
            eos_weight=self.eos_weight,
        )
        return encoded


class ReviewedTitleWeightedUnlimitedOCR(ReviewedTitleMaskUnlimitedOCR):
    """Weight trusted headings while preserving full response supervision."""

    template_name = WEIGHTED_TEMPLATE_NAME
    body_weight = 1.0
    title_weight = TRUSTED_TITLE_WEIGHT
    eos_weight = TRUSTED_EOS_WEIGHT


class ReviewedUniformUnlimitedOCR(ReviewedTitleMaskUnlimitedOCR):
    """Use the custom weighted-loss path with every response token at 1x."""

    template_name = UNIFORM_TEMPLATE_NAME
    body_weight = 1.0
    title_weight = 1.0
    eos_weight = 1.0


class _UOCRWeightedTokenMean(BaseLoss):
    """Loss already multiplied by the aligned scale in Seq2SeqTrainer."""

    loss_name = LOSS_NAME
    binary_loss_scale = True
    body_weight = 0.0
    title_weight = 1.0
    native_token_denominator = False

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        if getattr(args, "average_tokens_across_devices", False):
            raise ValueError(f"{self.loss_name} requires --average_tokens_across_devices false")
        if getattr(args, "use_liger_kernel", False):
            raise ValueError(f"{self.loss_name} requires --use_liger_kernel false")
        if getattr(args, "enable_dft_loss", False):
            raise ValueError(f"{self.loss_name} is not compatible with DFT loss")
        if getattr(trainer.template, "sequence_parallel_size", 1) != 1:
            raise ValueError(f"{self.loss_name} currently supports sequence_parallel_size=1 only")

    def __call__(
        self,
        outputs,
        labels,
        *,
        num_items_in_batch=None,
        loss_scale=None,
        **kwargs,
    ):
        if loss_scale is None or outputs.loss is None:
            raise RuntimeError(f"{self.loss_name} requires aligned loss_scale and per-token loss")
        loss_scale = loss_scale.to(outputs.loss.device)
        if loss_scale.ndim != 1 or outputs.loss.ndim != 1 or loss_scale.numel() != outputs.loss.numel():
            raise RuntimeError("aligned loss_scale and per-token loss must be equal-length vectors")
        if not torch.all(torch.isfinite(loss_scale)) or torch.any(loss_scale < 0):
            raise RuntimeError(f"{self.loss_name} requires finite non-negative loss_scale")
        if self.binary_loss_scale and not torch.all((loss_scale == 0) | (loss_scale == 1)):
            raise RuntimeError("title loss_scale must remain binary 0/1")

        aligned_labels = torch.roll(labels, shifts=-1, dims=-1).reshape(-1).to(loss_scale.device)
        valid = aligned_labels != -100
        active = (loss_scale > 0) & valid
        self._update_metrics(outputs, aligned_labels, loss_scale, active, valid)

        if self.native_token_denominator:
            if num_items_in_batch is None:
                denominator = valid.sum()
            else:
                denominator = num_items_in_batch
            denominator = torch.as_tensor(
                denominator,
                device=outputs.loss.device,
                dtype=outputs.loss.dtype,
            )
            if denominator.numel() != 1 or not torch.isfinite(denominator) or denominator.item() <= 0:
                raise RuntimeError(f"{self.loss_name} requires a positive scalar num_items_in_batch")
            return outputs.loss.sum() / denominator

        local_denom = loss_scale[valid].sum().detach().to(outputs.loss.device)
        if local_denom.item() <= 0:
            raise RuntimeError(f"{self.loss_name} local batch has no active token")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_denom = local_denom.clone()
            torch.distributed.all_reduce(global_denom, op=torch.distributed.ReduceOp.SUM)
            if global_denom.item() <= 0:
                raise RuntimeError(f"{self.loss_name} distributed batch has no active token")
            world_size = torch.distributed.get_world_size()
            return outputs.loss.sum() * world_size / global_denom
        return outputs.loss.sum() / local_denom

    @torch.no_grad()
    def _update_metrics(self, outputs, aligned_labels, loss_scale, active, valid):
        mode = "train" if self.trainer.model.training else "eval"
        metrics = self.trainer.custom_metrics[mode]
        eos_token_id = self.trainer.template.tokenizer.eos_token_id
        eos = active & (aligned_labels == eos_token_id)
        if self.title_weight == self.body_weight:
            title = torch.zeros_like(active)
        else:
            title = active & (aligned_labels != eos_token_id) & (loss_scale == self.title_weight)
        body = active & ~title & ~eos
        masked_body = valid & ~active

        metrics["active_title_tokens"].update(title.sum())
        metrics["active_body_tokens"].update(body.sum())
        metrics["active_eos_tokens"].update(eos.sum())
        metrics["active_weight_sum"].update(loss_scale[valid].sum())
        metrics["masked_body_tokens"].update(masked_body.sum())
        if title.any():
            metrics["title_token_loss"].update(outputs.loss[title] / loss_scale[title])
            predictions = outputs.logits.argmax(dim=-1).reshape(-1).to(aligned_labels.device)
            metrics["title_token_acc"].update((predictions[title] == aligned_labels[title]).float())


class UOCRTitleActiveMean(_UOCRWeightedTokenMean):
    """Legacy title+EOS active-token mean with body weight zero."""

    loss_name = LOSS_NAME
    binary_loss_scale = True
    body_weight = 0.0
    title_weight = 1.0


class UOCRTitleWeightedMean(_UOCRWeightedTokenMean):
    """Weighted CE using the same accumulation denominator as native CE."""

    loss_name = WEIGHTED_LOSS_NAME
    binary_loss_scale = False
    body_weight = 1.0
    title_weight = TRUSTED_TITLE_WEIGHT
    native_token_denominator = True


class UOCRLegacyTitleWeightedMean(_UOCRWeightedTokenMean):
    """Historical per-micro-step active-weight mean used by the 2026-08-25 run."""

    loss_name = LEGACY_WEIGHTED_LOSS_NAME
    binary_loss_scale = False
    body_weight = 1.0
    title_weight = TRUSTED_TITLE_WEIGHT
    native_token_denominator = False


class UOCRUniformMean(_UOCRWeightedTokenMean):
    """All-ones control using the same accumulation denominator as native CE."""

    loss_name = UNIFORM_LOSS_NAME
    binary_loss_scale = True
    body_weight = 1.0
    title_weight = 1.0
    native_token_denominator = True


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

    if WEIGHTED_TEMPLATE_NAME not in TEMPLATE_MAPPING:
        meta = deepcopy(TEMPLATE_MAPPING["unlimited_ocr"])
        meta.template_type = WEIGHTED_TEMPLATE_NAME
        meta.template_cls = ReviewedTitleWeightedUnlimitedOCR
        register_template(meta)
    elif TEMPLATE_MAPPING[WEIGHTED_TEMPLATE_NAME].template_cls is not ReviewedTitleWeightedUnlimitedOCR:
        raise RuntimeError(f"template name collision: {WEIGHTED_TEMPLATE_NAME}")

    if UNIFORM_TEMPLATE_NAME not in TEMPLATE_MAPPING:
        meta = deepcopy(TEMPLATE_MAPPING["unlimited_ocr"])
        meta.template_type = UNIFORM_TEMPLATE_NAME
        meta.template_cls = ReviewedUniformUnlimitedOCR
        register_template(meta)
    elif TEMPLATE_MAPPING[UNIFORM_TEMPLATE_NAME].template_cls is not ReviewedUniformUnlimitedOCR:
        raise RuntimeError(f"template name collision: {UNIFORM_TEMPLATE_NAME}")

    existing_loss = loss_map.get(LOSS_NAME)
    if existing_loss not in (None, UOCRTitleActiveMean):
        raise RuntimeError(f"loss name collision: {LOSS_NAME}")
    loss_map[LOSS_NAME] = UOCRTitleActiveMean

    existing_weighted_loss = loss_map.get(WEIGHTED_LOSS_NAME)
    if existing_weighted_loss not in (None, UOCRTitleWeightedMean):
        raise RuntimeError(f"loss name collision: {WEIGHTED_LOSS_NAME}")
    loss_map[WEIGHTED_LOSS_NAME] = UOCRTitleWeightedMean

    existing_legacy_weighted_loss = loss_map.get(LEGACY_WEIGHTED_LOSS_NAME)
    if existing_legacy_weighted_loss not in (None, UOCRLegacyTitleWeightedMean):
        raise RuntimeError(f"loss name collision: {LEGACY_WEIGHTED_LOSS_NAME}")
    loss_map[LEGACY_WEIGHTED_LOSS_NAME] = UOCRLegacyTitleWeightedMean

    existing_uniform_loss = loss_map.get(UNIFORM_LOSS_NAME)
    if existing_uniform_loss not in (None, UOCRUniformMean):
        raise RuntimeError(f"loss name collision: {UNIFORM_LOSS_NAME}")
    loss_map[UNIFORM_LOSS_NAME] = UOCRUniformMean

    existing_callback = callbacks_map.get(CALLBACK_NAME)
    if existing_callback not in (None, UOCRLoRATargetGuard):
        raise RuntimeError(f"callback name collision: {CALLBACK_NAME}")
    callbacks_map[CALLBACK_NAME] = UOCRLoRATargetGuard


_register()
