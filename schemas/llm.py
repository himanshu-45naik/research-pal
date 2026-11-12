
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    """A message sent to an LLM."""

    role: Literal[
        "system",
        "developer",
        "user",
        "assistant",
        "tool"
    ]

    content: str
    name: str | None = None

    # Required on role="tool": the id of the tool call this message answers.
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_tool_call_id(self) -> "ChatMessage":
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages need a tool_call_id")
        if self.role != "tool" and self.tool_call_id:
            raise ValueError("only tool messages take a tool_call_id")
        return self


class LLMRequest(BaseModel):
    """Input to an LLM provider."""

    request_id: str

    model: str
    messages: list[ChatMessage]

    max_completion_tokens: int = Field(
        default=1000,
        gt=0
    )

    temperature: float = Field(
        default=0.2,
        ge=0,
        le=2
    )

    stream: bool = False
    timeout_s: float = Field(default=30, gt=0)


class TokenUsage(BaseModel):
    """Tokens billed for one LLM call."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class RateLimitInfo(BaseModel):
    """Rate-limit state reported in the provider's response headers.

    Per Groq's docs, the *requests* headers track the per-day request limit
    (RPD) and the *tokens* headers track the per-minute token limit (TPM).
    No header reports requests per minute or tokens per day. Confirm with probes.
    """

    limit_requests: int | None = None        # x-ratelimit-limit-requests
    remaining_requests: int | None = None    # x-ratelimit-remaining-requests
    reset_requests_s: float | None = None    # x-ratelimit-reset-requests

    limit_tokens: int | None = None          # x-ratelimit-limit-tokens
    remaining_tokens: int | None = None      # x-ratelimit-remaining-tokens
    reset_tokens_s: float | None = None      # x-ratelimit-reset-tokens

    retry_after_s: float | None = None       # retry-after, sent only with a 429

    # Header values exactly as received, kept as evidence for the limits table.
    raw_headers: dict[str, str] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Standard response from any LLM provider."""

    request_id: str

    provider: Literal["groq", "stand-in"]
    model: str

    content: str

    usage: TokenUsage = Field(
        default_factory=TokenUsage
    )

    latency_ms: float = Field(ge=0)
    first_token_ms: float | None = Field(
        default=None,
        ge=0
    )

    finish_reason: str | None = None

    rate_limit: RateLimitInfo = Field(
        default_factory=RateLimitInfo
    )

