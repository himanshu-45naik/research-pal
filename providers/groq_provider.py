
import os
import re
import time
import asyncio

import httpx
from dotenv import load_dotenv
from groq import AsyncGroq, APIConnectionError, APIError, APIStatusError, APITimeoutError
from groq.types import CompletionUsage
from loguru import logger

from providers.errors import ErrorKind, LLMProviderError
from schemas.llm import ChatMessage, LLMRequest, LLMResponse, RateLimitInfo, TokenUsage

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Durations look like "2m59.56s", "7.66s", "120ms", or a bare number of seconds.
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNIT_S = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _get_duration_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000


def _token_usage(usage: CompletionUsage | None) -> TokenUsage:
    if usage is None:
        return TokenUsage()

    return TokenUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens
    )


def _parse_duration_s(value: str | None) -> float | None:
    if not value:
        return None

    value = value.strip()

    try:
        return float(value)
    except ValueError:
        pass

    parts = _DURATION_PART.findall(value)
    if not parts:
        return None

    return sum(float(amount) * _DURATION_UNIT_S[unit] for amount, unit in parts)


def _parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _rate_limit_info(headers: httpx.Headers) -> RateLimitInfo:

    raw = {
        name: value
        for name, value in headers.items()
        if name.startswith("x-ratelimit-") or name == "retry-after"
    }

    return RateLimitInfo(
        limit_requests=_parse_int(raw.get("x-ratelimit-limit-requests")),
        remaining_requests=_parse_int(raw.get("x-ratelimit-remaining-requests")),
        reset_requests_s=_parse_duration_s(raw.get("x-ratelimit-reset-requests")),
        limit_tokens=_parse_int(raw.get("x-ratelimit-limit-tokens")),
        remaining_tokens=_parse_int(raw.get("x-ratelimit-remaining-tokens")),
        reset_tokens_s=_parse_duration_s(raw.get("x-ratelimit-reset-tokens")),
        retry_after_s=_parse_duration_s(raw.get("retry-after")),
        raw_headers=raw
    )


def _status_error(
    exc: APIStatusError,
    request: LLMRequest,
    elapsed_ms: float
) -> LLMProviderError:

    kind: ErrorKind
    if exc.status_code == 429:
        kind = "rate_limited"
    elif exc.status_code >= 500:
        kind = "server_error"
    else:
        kind = "client_error"

    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    error = error if isinstance(error, dict) else {}

    return LLMProviderError(
        # For a 429 the message says which limit was hit (e.g. tokens per minute).
        error.get("message") or exc.message,
        request_id=request.request_id,
        kind=kind,
        status_code=exc.status_code,
        error_type=error.get("type"),
        error_code=error.get("code"),
        rate_limit=_rate_limit_info(exc.response.headers),
        elapsed_ms=elapsed_ms
    )


class GroqLLMService:

    def __init__(self) -> None:

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY is missing")

        # Create the client once, reuse it for every request.
        self.client = AsyncGroq(
            api_key=api_key,
            timeout=30.0,
            max_retries=0
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:

        start_time = time.perf_counter()

        try:
            # timeout_s is a deadline for the whole call. httpx's own timeout
            # only limits each read, so a slowly trickling stream would never hit it.
            async with asyncio.timeout(request.timeout_s):
                if request.stream:
                    result = await self._complete_stream(request, start_time)
                else:
                    result = await self._complete_once(request, start_time)

        except APIStatusError as exc:
            raise self._failed(
                _status_error(exc, request, _get_duration_ms(start_time))
            ) from exc

        # The SDK wraps errors while sending the request. Errors while reading
        # a stream arrive unwrapped from httpx, so both forms are caught.
        except (TimeoutError, APITimeoutError, httpx.TimeoutException) as exc:
            raise self._failed(LLMProviderError(
                f"No complete response within {request.timeout_s}s",
                request_id=request.request_id,
                kind="timeout",
                elapsed_ms=_get_duration_ms(start_time)
            )) from exc

        except (APIConnectionError, httpx.TransportError) as exc:
            raise self._failed(LLMProviderError(
                str(exc),
                request_id=request.request_id,
                kind="connection",
                elapsed_ms=_get_duration_ms(start_time)
            )) from exc

        except APIError as exc:
            # An error event sent inside an otherwise successful (200) stream.
            raise self._failed(LLMProviderError(
                exc.message,
                request_id=request.request_id,
                kind="server_error",
                elapsed_ms=_get_duration_ms(start_time)
            )) from exc

        logger.info(
            "LLM request completed | Request: {} | Model: {} | Latency: {:.2f} ms | "
            "First token: {} ms | Tokens: {} | Remaining tokens: {} | Remaining requests: {}",
            result.request_id,
            result.model,
            result.latency_ms,
            None if result.first_token_ms is None else round(result.first_token_ms, 2),
            result.usage.total_tokens,
            result.rate_limit.remaining_tokens,
            result.rate_limit.remaining_requests
        )

        return result

    def _failed(self, error: LLMProviderError) -> LLMProviderError:

        logger.warning(
            "Groq API request failed | Request: {} | {} | After: {:.2f} ms | "
            "Retry after: {} s | Remaining tokens: {} | Remaining requests: {}",
            error.request_id,
            error,
            error.elapsed_ms or 0.0,
            error.rate_limit.retry_after_s,
            error.rate_limit.remaining_tokens,
            error.rate_limit.remaining_requests
        )

        return error

    def _create_params(self, request: LLMRequest) -> dict:

        return {
            "model": request.model,
            "messages": [
                message.model_dump(exclude_none=True)
                for message in request.messages
            ],
            "max_completion_tokens": request.max_completion_tokens,
            "temperature": request.temperature,
            "timeout": request.timeout_s,
        }

    async def _complete_once(
        self,
        request: LLMRequest,
        start_time: float
    ) -> LLMResponse:

        # with_raw_response exposes the HTTP headers alongside the parsed body.
        raw = await self.client.chat.completions.with_raw_response.create(
            **self._create_params(request)
        )
        response = await raw.parse()

        choice = response.choices[0]

        return LLMResponse(
            request_id=request.request_id,
            provider="groq",
            model=response.model,
            content=choice.message.content or "",
            usage=_token_usage(response.usage),
            latency_ms=_get_duration_ms(start_time),
            # Not observable without streaming.
            first_token_ms=None,
            finish_reason=choice.finish_reason,
            rate_limit=_rate_limit_info(raw.headers)
        )

    async def _complete_stream(
        self,
        request: LLMRequest,
        start_time: float
    ) -> LLMResponse:

        stream = await self.client.chat.completions.create(
            **self._create_params(request),
            stream=True
        )

        # Headers arrive before the first chunk.
        rate_limit = _rate_limit_info(stream.response.headers)

        parts: list[str] = []
        first_token_ms: float | None = None
        finish_reason: str | None = None
        usage = None
        model = request.model

        async with stream:
            async for chunk in stream:

                model = chunk.model or model

                # Groq sends usage on the final chunk, either top-level
                # or under x_groq depending on the response.
                if chunk.usage:
                    usage = chunk.usage
                elif chunk.x_groq and chunk.x_groq.usage:
                    usage = chunk.x_groq.usage

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                # First token = first visible answer text. Reasoning deltas
                # (delta.reasoning on gpt-oss) are not counted.
                if choice.delta.content:
                    if first_token_ms is None:
                        first_token_ms = _get_duration_ms(start_time)
                    parts.append(choice.delta.content)

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

        return LLMResponse(
            request_id=request.request_id,
            provider="groq",
            model=model,
            content="".join(parts),
            usage=_token_usage(usage),
            latency_ms=_get_duration_ms(start_time),
            first_token_ms=first_token_ms,
            finish_reason=finish_reason,
            rate_limit=rate_limit
        )


async def main() -> None:

    # Created here, not at import time, so importing this module does not
    # need an API key or open a client.
    llm = GroqLLMService()

    request = LLMRequest(
        request_id="demo-1",
        model=MODEL,
        messages=[
            ChatMessage(
                role="system",
                content="You are a helpful research assistant."
            ),
            ChatMessage(
                role="user",
                content="Explain the purpose of multi-head attention."
            )
        ],
        stream=True
    )

    result = await llm.complete(request)

    print(result.content)
    print("Input tokens:", result.usage.input_tokens)
    print("Output tokens:", result.usage.output_tokens)
    print("First token (ms):", result.first_token_ms)
    print("Latency (ms):", result.latency_ms)
    print("Finish reason:", result.finish_reason)
    print("Rate limits:", result.rate_limit.model_dump(exclude={"raw_headers"}))
    print("Raw rate-limit headers:", result.rate_limit.raw_headers)


if __name__ == "__main__":
    asyncio.run(main())
