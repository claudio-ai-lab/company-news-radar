"""Provider-neutral LLM interface plus the structured-output guard rails.

Every LLM call in the pipeline goes through :class:`StructuredLLM`, which

1. refuses to call when the daily budget is used up,
2. tells the model the exact JSON schema it must return,
3. retries transient provider errors with exponential backoff,
4. parses and validates the answer with Pydantic, and on failure gives the model one
   repair round that includes the validation error,
5. records tokens and cost for every attempt, successful or not.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from newsradar.llm.costs import CostTracker

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SCHEMA_INSTRUCTION = (
    "Return exactly one JSON object and nothing else (no prose, no code fences). "
    "It must validate against this JSON Schema:\n{schema}"
)
REPAIR_INSTRUCTION = (
    "\n\nYour previous answer could not be used: {error}\n"
    "Previous answer (truncated): {previous}\n"
    "Answer again with one JSON object that follows the schema exactly."
)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMError(Exception):
    """Base class for LLM failures the pipeline can recover from."""


class TransientLLMError(LLMError):
    """Rate limits, timeouts, 5xx: worth retrying."""


class StructuredOutputError(LLMError):
    """The model did not produce valid structured output, even after repair."""


@dataclass
class LLMRequest:
    stage: str
    system: str
    user: str
    temperature: float | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int = 1500
    # The structured inputs the prompt was rendered from. Real providers ignore it;
    # the offline client uses it so demos and CI need no network and no API key.
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    provider: str
    model: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


def parse_structured(text: str, schema: type[T]) -> T:
    """Extract the JSON object from a model answer and validate it against ``schema``."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the answer")
    data = json.loads(cleaned[start : end + 1])
    return schema.model_validate(data)


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()[:5]
        ]
        return "; ".join(parts)
    return str(exc)[:300]


class StructuredLLM:
    """Wraps an :class:`LLMClient` for one pipeline stage."""

    def __init__(
        self,
        client: LLMClient,
        costs: CostTracker,
        stage: str,
        *,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 1500,
        max_attempts: int = 3,
        repair_rounds: int = 1,
        backoff_s: float = 1.0,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.client = client
        self.costs = costs
        self.stage = stage
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts
        self.repair_rounds = repair_rounds
        self.backoff_s = backoff_s
        self._semaphore = semaphore or asyncio.Semaphore(8)

    @property
    def model_id(self) -> str:
        return f"{self.client.provider}/{self.client.model}"

    async def generate(
        self,
        system: str,
        user: str,
        schema: type[T],
        payload: dict[str, Any] | None = None,
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        full_system = f"{system.rstrip()}\n\n{SCHEMA_INSTRUCTION.format(schema=schema_json)}"
        prompt = user
        last_error: Exception | None = None
        for _ in range(self.repair_rounds + 1):
            request = LLMRequest(
                stage=self.stage,
                system=full_system,
                user=prompt,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                max_output_tokens=self.max_output_tokens,
                payload=payload or {},
            )
            response = await self._call(request)
            try:
                return parse_structured(response.text, schema)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                logger.warning("%s: invalid structured output (%s)", self.stage, _short_error(exc))
                prompt = user + REPAIR_INSTRUCTION.format(
                    error=_short_error(exc), previous=response.text[:1500]
                )
        raise StructuredOutputError(f"{self.stage}: {_short_error(last_error or ValueError())}")

    async def _call(self, request: LLMRequest) -> LLMResponse:
        provider, model = self.client.provider, self.client.model
        for attempt in range(1, self.max_attempts + 1):
            self.costs.ensure_budget()
            try:
                async with self._semaphore:
                    response = await self.client.complete(request)
            except TransientLLMError as exc:
                self.costs.record(self.stage, provider, model, 0, 0, ok=False, error=str(exc))
                if attempt == self.max_attempts:
                    raise
                delay = self.backoff_s * 2 ** (attempt - 1) * (1 + random.random() / 4)
                logger.info("%s: transient error, retry %d in %.1fs", self.stage, attempt, delay)
                await asyncio.sleep(delay)
                continue
            except Exception as exc:
                self.costs.record(self.stage, provider, model, 0, 0, ok=False, error=str(exc))
                raise LLMError(f"{self.stage}: {exc}") from exc
            self.costs.record(
                self.stage, provider, model, response.input_tokens, response.output_tokens, ok=True
            )
            return response
        raise TransientLLMError(f"{self.stage}: retries exhausted")  # pragma: no cover
