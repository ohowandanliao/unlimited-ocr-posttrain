"""Import shim for an incompatible system-only Apex placeholder.

The training environment uses bf16, so Transformers must not initialize Apex.
This module only satisfies its optional import guard while keeping that backend
unavailable at runtime.
"""


class _UnavailableAmp:
    def __getattr__(self, name):
        raise RuntimeError("Apex AMP is intentionally unavailable; use bf16")


amp = _UnavailableAmp()
