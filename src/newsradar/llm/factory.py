"""Build one :class:`StructuredLLM` per pipeline stage from the configuration."""

from __future__ import annotations

import asyncio
import os

from newsradar.config import AppConfig, ConfigError, StageModel
from newsradar.llm.base import LLMClient, StructuredLLM
from newsradar.llm.costs import CostTracker
from newsradar.llm.fake import OfflineLLM


def build_client(config: AppConfig, stage_model: StageModel) -> LLMClient:
    settings = config.llm.provider(stage_model.provider)
    if settings.kind == "fake":
        return OfflineLLM(config)

    api_key = os.environ.get(settings.api_key_env) if settings.api_key_env else None
    if settings.api_key_env and not api_key:
        raise ConfigError(
            f"provider {stage_model.provider!r} needs the environment variable "
            f"{settings.api_key_env} (see .env.example), or run with --offline"
        )
    if settings.kind == "gemini":
        from newsradar.llm.gemini import GeminiClient

        return GeminiClient(stage_model.model, api_key, provider=stage_model.provider)

    from newsradar.llm.openai_compat import OpenAICompatibleClient

    return OpenAICompatibleClient(
        provider=stage_model.provider,
        model=stage_model.model,
        api_key=api_key,
        base_url=settings.base_url,
        timeout_s=config.llm.request_timeout_s,
        native_openai=settings.kind == "openai",
    )


def build_stage_llms(
    config: AppConfig, costs: CostTracker, *, offline: bool = False
) -> dict[str, StructuredLLM]:
    """One wrapper per stage; all share a concurrency limit and the cost tracker."""
    from newsradar.config import STAGES

    semaphore = asyncio.Semaphore(config.llm.max_concurrency)
    llms: dict[str, StructuredLLM] = {}
    clients: dict[tuple[str, str], LLMClient] = {}
    for stage in STAGES:
        stage_model = config.llm.for_stage(stage)
        if offline:
            stage_model = StageModel(provider="fake", model="keyword-heuristic")
        key = (stage_model.provider, stage_model.model)
        if key not in clients:
            clients[key] = build_client(config, stage_model)
        llms[stage] = StructuredLLM(
            clients[key],
            costs,
            stage,
            temperature=stage_model.temperature,
            reasoning_effort=stage_model.reasoning_effort,
            max_output_tokens=stage_model.max_output_tokens,
            semaphore=semaphore,
        )
    return llms
