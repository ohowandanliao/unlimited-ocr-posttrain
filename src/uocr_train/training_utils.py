"""Small stateful helpers used by the training loop."""

from dataclasses import dataclass, field


@dataclass
class SequenceLengthGuard:
    """Apply an explicit error-or-drop policy to encoded sequence lengths."""

    max_length: int
    strategy: str = "error"
    encoded: int = field(default=0, init=False)
    max_observed: int = field(default=0, init=False)
    over_limit: int = field(default=0, init=False)
    dropped: int = field(default=0, init=False)
    over_limit_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self):
        self.strategy = self.strategy.lower()
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.strategy not in {"error", "drop"}:
            raise ValueError("length_strategy must be one of: error, drop")

    def check(self, sample_id, sequence_length: int) -> bool:
        """Return whether a sample should be kept, or raise under the error policy."""
        self.encoded += 1
        self.max_observed = max(self.max_observed, sequence_length)
        if sequence_length <= self.max_length:
            return True

        sample_name = str(sample_id) if sample_id is not None else "<unknown>"
        self.over_limit += 1
        self.over_limit_ids.add(sample_name)
        message = (
            f"sample {sample_name!r} encoded to {sequence_length} tokens, "
            f"exceeding max_length={self.max_length}"
        )
        if self.strategy == "error":
            raise ValueError(f"{message}; raise max_length or set length_strategy: drop")

        self.dropped += 1
        return False

    def summary(self) -> str:
        return (
            f"max_length={self.max_length} strategy={self.strategy} encoded={self.encoded} "
            f"max_observed={self.max_observed} over_limit={self.over_limit} "
            f"over_limit_unique={len(self.over_limit_ids)} dropped={self.dropped}"
        )
