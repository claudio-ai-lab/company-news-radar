"""Client for Google Gemini via the ``google-genai`` SDK (optional extra ``[gemini]``)."""

from __future__ import annotations

from typing import Any

from newsradar.llm.base import LLMRequest, LLMResponse, TransientLLMError

_TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}


class GeminiClient:
    def __init__(self, model: str, api_key: str | None, provider: str = "gemini") -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Gemini support needs the optional extra: pip install 'company-news-radar[gemini]'"
            ) from exc
        self.provider = provider
        self.model = model
        self._client: Any = genai.Client(api_key=api_key)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=request.system,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            response_mime_type="application/json",
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=request.user, config=config
            )
        except errors.APIError as exc:
            if getattr(exc, "code", None) in _TRANSIENT_CODES:
                raise TransientLLMError(str(exc)) from exc
            raise
        usage = response.usage_metadata
        output_tokens = 0
        input_tokens = 0
        if usage is not None:
            input_tokens = usage.prompt_token_count or 0
            # Thinking tokens are billed as output tokens.
            output_tokens = (usage.candidates_token_count or 0) + (
                getattr(usage, "thoughts_token_count", 0) or 0
            )
        return LLMResponse(
            text=response.text or "", input_tokens=input_tokens, output_tokens=output_tokens
        )
