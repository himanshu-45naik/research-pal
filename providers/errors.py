
from typing import Literal

from schemas.llm import RateLimitInfo


ErrorKind = Literal[
    "rate_limited",    # 429
    "server_error",    # 5xx
    "client_error",    # other 4xx
    "timeout",         # our deadline or the HTTP client's timeout
    "connection",      # could not reach the provider
]


class LLMProviderError(Exception):
    """An LLM call failed. Carries what the caller needs to decide on a retry."""

    def __init__(
        self,
        message: str,
        *,
        request_id: str,
        kind: ErrorKind,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        rate_limit: RateLimitInfo | None = None,
        elapsed_ms: float | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.kind = kind
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.rate_limit = rate_limit or RateLimitInfo()
        self.elapsed_ms = elapsed_ms

    def __str__(self) -> str:
        status = f" {self.status_code}" if self.status_code else ""
        return f"[{self.kind}{status}] {self.message}"
