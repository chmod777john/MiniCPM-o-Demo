"""Explicit cache-limit errors shared by model and serving layers."""


class CacheLimitExceeded(RuntimeError):
    """Raised before a model call would write beyond a session cache."""

    def __init__(self, cache_name: str, current_length: int, requested_length: int, limit: int):
        self.cache_name = str(cache_name)
        self.current_length = int(current_length)
        self.requested_length = int(requested_length)
        self.limit = int(limit)
        super().__init__(
            f"{self.cache_name} cache limit exceeded: "
            f"current={self.current_length}, requested={self.requested_length}, limit={self.limit}"
        )

    def as_dict(self) -> dict[str, int | str]:
        return {
            "cache_name": self.cache_name,
            "current_length": self.current_length,
            "requested_length": self.requested_length,
            "limit": self.limit,
        }
