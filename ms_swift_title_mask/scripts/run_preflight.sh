#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MS_SWIFT_ROOT=${MS_SWIFT_ROOT:-$(dirname "$BUNDLE_ROOT")}
PYTHON_BIN=${PYTHON_BIN:-python}

COMPAT_ARGS=(--ms-swift-root "$MS_SWIFT_ROOT")
if [[ -n "${MODEL_PATH:-}" ]]; then
  COMPAT_ARGS+=(--model "$MODEL_PATH")
fi

"$PYTHON_BIN" "$BUNDLE_ROOT/scripts/check_ms_swift_compat.py" "${COMPAT_ARGS[@]}"
PYTHONPATH="$(dirname "$BUNDLE_ROOT"):$MS_SWIFT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m unittest discover -s "$BUNDLE_ROOT/tests" -p 'test_*.py' -v

if [[ -n "${TRAIN_JSONL:-}" && -n "${VAL_JSONL:-}" && -n "${TEST_JSONL:-}" ]]; then
  "$PYTHON_BIN" "$BUNDLE_ROOT/scripts/validate_reviewed_jsonl.py" \
    "$TRAIN_JSONL" "$VAL_JSONL" "$TEST_JSONL"
fi

echo "preflight_ok"
