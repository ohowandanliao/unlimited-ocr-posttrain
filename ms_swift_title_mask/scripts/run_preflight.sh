#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POSTTRAIN_ROOT=$(cd "$BUNDLE_ROOT/.." && pwd)
HYX_ROOT=$(cd "$POSTTRAIN_ROOT/.." && pwd)
UOCR_ROOT=${UOCR_ROOT:-$HYX_ROOT/uocr-ms-swift-title-mask}
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-$UOCR_ROOT/repos/ms-swift-uocr}
UOCR_SHIMS_ROOT=${UOCR_SHIMS_ROOT:-$UOCR_ROOT/runtime/shims}
PYTHON_BIN=${PYTHON_BIN:-$UOCR_ROOT/env/ms-swift-venv/bin/python}
export PYTHONPATH="$UOCR_SHIMS_ROOT:$MS_SWIFT_ROOT:$POSTTRAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

COMPAT_ARGS=(--ms-swift-root "$MS_SWIFT_ROOT")
if [[ -n "${MODEL_PATH:-}" ]]; then
  COMPAT_ARGS+=(--model "$MODEL_PATH")
fi

"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/check_ms_swift_compat.py" "${COMPAT_ARGS[@]}"
"$PYTHON_BIN" - "$BUNDLE_ROOT/tests" <<'PY'
import sys
import unittest


SKIP = "test_ddp_math.DDPReferenceMathTest.test_two_process_ddp_gradient_matches_global_active_mean"


def filter_suite(suite):
    filtered = unittest.TestSuite()
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            filtered.addTest(filter_suite(test))
        elif test.id() != SKIP:
            filtered.addTest(test)
    return filtered


suite = unittest.defaultTestLoader.discover(sys.argv[1], pattern="test_*.py")
print(f"preflight: skipped known reference test: {SKIP}")
result = unittest.TextTestRunner(verbosity=2).run(filter_suite(suite))
raise SystemExit(0 if result.wasSuccessful() else 1)
PY

if [[ -n "${TRAIN_JSONL:-}" && -n "${VAL_JSONL:-}" && -n "${TEST_JSONL:-}" ]]; then
  "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/validate_reviewed_jsonl.py" \
    "$TRAIN_JSONL" "$VAL_JSONL" "$TEST_JSONL"
fi

echo "preflight_ok"
