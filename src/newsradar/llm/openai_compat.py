"""Client for OpenAI and any OpenAI-compatible endpoint (DeepSeek, Ollama, vLLM, ...)."""

from __future__ import annotations

from typing import Any

import openai
from openai import AsyncOpenAI

from newsradar.llm.base import LLMRequest, LLMResponse, TransientLLMError

_TRANSIENT = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAICompatibleClient:
    """Uses JSON mode, which OpenAI-compatible servers widely support; schema checks happen
    client-side in :class:`~newsradar.llm.base.StructuredLLM`."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        native_openai: bool = False,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._native = native_openai
        self._client = client or AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,  # retries are handled (and accounted for) by StructuredLLM
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "response_format": {"type": "json_object"},
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.reasoning_effort is not None:
            # Sent as a raw body field so older SDK versions pass it through unchanged.
            params["extra_body"] = {"reasoning_effort": request.reasoning_effort}
        # OpenAI's own API renamed the limit; compatible servers still expect max_tokens.
        limit_key = "max_completion_tokens" if self._native else "max_tokens"
        params[limit_key] = request.max_output_tokens
        try:
            response = await self._client.chat.completions.create(**params)
        except _TRANSIENT as exc:
            raise TransientLLMError(str(exc)) from exc
        text = ""
        if response.choices:
            text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
