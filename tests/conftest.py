from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from newsradar.config import AppConfig, load_config
from newsradar.llm.base import LLMRequest, LLMResponse, StructuredLLM
from newsradar.llm.costs import CostTracker
from newsradar.models import Article, ClassificationOutput, Stage, SummaryOutput
from newsradar.prompts import PromptLibrary
from newsradar.storage.sqlite import SQLiteRepository

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "example-ag"

Reply = str | dict[str, Any] | Exception
Responder = Callable[[LLMRequest], Reply]


class ScriptedLLM:
    """Test double for an LLM: replies from a list (in order) or from a function."""

    provider = "scripted"
    model = "test-model"

    def __init__(self, replies: list[Reply] | Responder) -> None:
        self._replies = replies
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        reply = self._replies(request) if callable(self._replies) else self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text, input_tokens=100, output_tokens=20)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    cfg = load_config(EXAMPLE_DIR / "offline.yaml")
    cfg.storage.url = f"sqlite:///{(tmp_path / 'radar.db').as_posix()}"
    cfg.output.dir = str(tmp_path / "out")
    return cfg


@pytest.fixture
def repo() -> SQLiteRepository:
    repository = SQLiteRepository(":memory:")
    yield repository
    repository.close()


@pytest.fixture
def costs() -> CostTracker:
    return CostTracker({}, daily_budget_usd=100.0, run_id="test")


@pytest.fixture
def prompts(config: AppConfig) -> PromptLibrary:
    return PromptLibrary(config)


def structured(
    client: ScriptedLLM, costs: CostTracker, stage: str = "classification"
) -> StructuredLLM:
    return StructuredLLM(client, costs, stage, backoff_s=0.0, semaphore=asyncio.Semaphore(4))


_ids = iter(range(1, 10_000))


def make_article(**overrides: Any) -> Article:
    article_id = overrides.pop("id", next(_ids))
    data: dict[str, Any] = {
        "id": article_id,
        "url": f"https://example.com/news/story-{article_id}",
        "url_hash": f"hash-{article_id}",
        "title": f"Story {article_id}",
        "source": "Example Wire",
        "collected_at": datetime(2026, 10, 16, 8, 0, tzinfo=UTC),
        "published_at": datetime(2026, 10, 16, 7, 0, tzinfo=UTC),
        "stage": Stage.FETCHED,
    }
    data.update(overrides)
    return Article(**data)


def classification(**overrides: Any) -> ClassificationOutput:
    data: dict[str, Any] = {
        "relevant": True,
        "topic": "input_costs",
        "relevance": 7,
        "confidence": 0.9,
        "reason": "test",
    }
    data.update(overrides)
    return ClassificationOutput(**data)


def summary(**overrides: Any) -> SummaryOutput:
    data: dict[str, Any] = {
        "title": "Aluminium prices jump",
        "summary": "Aluminium rose **6%** this week.",
        "key_points": ["Smelter cuts output by a third"],
        "impact": "negative",
    }
    data.update(overrides)
    return SummaryOutput(**data)
