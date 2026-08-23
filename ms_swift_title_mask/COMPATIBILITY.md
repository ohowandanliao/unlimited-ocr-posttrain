# Compatibility contract

Tested source checkouts:

- ms-swift: `1a1ba3ee86488af323ef9b64ca3d34edee90ab11`
- Unlimited-OCR repository: `d49ff64afffc1f47ab563dc1c589bc2f78808fa4`

The plugin depends on these ms-swift behaviors:

1. `swift/template/templates/deepseek.py` registers `UnlimitedOCR` and builds the R-SWA mask from complete labels.
2. `Seq2SeqTrainer` rolls `loss_scale` by one token, flattens it, and multiplies per-token CE before invoking a custom loss.
3. `--external_plugins`, `TEMPLATE_MAPPING`, `loss_map`, and `callbacks_map` remain public extension points.
4. Unlimited-OCR training uses `transformers==4.46.3`; the tokenizer must be fast and return character offsets.
5. Sequence parallelism, FSDP, DeepSpeed, packing, Liger CE, and DFT loss are outside this bundle's tested custom-loss contract.

Run `scripts/run_preflight.sh` after every ms-swift update. Do not use
`ALLOW_UNTESTED_MS_SWIFT=1` until the changed template and trainer code have been reviewed and the integration tests rerun.
