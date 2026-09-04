# Compatibility contract

Tested public source checkouts:

- [modelscope/ms-swift](https://github.com/modelscope/ms-swift): `1a1ba3ee86488af323ef9b64ca3d34edee90ab11`
- [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR): `d49ff64afffc1f47ab563dc1c589bc2f78808fa4`

The plugin depends on these ms-swift behaviors:

1. `swift/template/templates/deepseek.py` registers `UnlimitedOCR` and builds the R-SWA mask from complete labels.
2. `Seq2SeqTrainer` rolls `loss_scale` by one token, flattens it, and multiplies per-token CE before invoking a custom loss.
3. `Seq2SeqTrainer` passes accumulation-level `num_items_in_batch` into custom loss. New `uniform_ce` and
   `title_weighted` use this native token denominator. The legacy title-only loss keeps its historical active-token mean,
   and `legacy_20260823` maps weighted training to the historical per-micro-batch active-weight mean.
4. `--external_plugins`, `TEMPLATE_MAPPING`, `loss_map`, and `callbacks_map` remain public extension points.
5. Unlimited-OCR training uses `transformers==4.46.3`; the tokenizer must be fast and return character offsets.
6. Sequence parallelism, FSDP, DeepSpeed, packing, Liger CE, and DFT loss are outside this bundle's tested custom-loss contract.

After every ms-swift update, review the changed template and trainer code against this contract and rerun the
integration tests before training. The removed compatibility scripts and `ALLOW_UNTESTED_MS_SWIFT` override are
not part of the current interface.
