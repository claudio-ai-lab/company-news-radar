from datetime import datetime

import pytest

from newsradar.config import Price
from newsradar.llm.base import (
    LLMError,
    StructuredOutputError,
    TransientLLMError,
    parse_structured,
)
from newsradar.llm.costs import BudgetExceededError, CostTracker
from newsradar.models import ClassificationOutput
from newsradar.storage.sqlite import SQLiteRepository
from tests.conftest import ScriptedLLM, structured

VALID = {
    "relevant": True,
    "topic": "input_costs",
    "relevance": 7,
    "confidence": 0.8,
    "reason": "ok",
}


def test_parse_structured_tolerates_fences_and_prose() -> None:
    text = 'Sure, here it is:\n```json\n{"relevant": false, "topic": "none", "relevance": 1, "confidence": 0.9, "reason": "x"}\n```'
    assert parse_structured(text, ClassificationOutput).relevant is False


async def test_generate_validates_and_sends_schema(costs: CostTracker) -> None:
    client = ScriptedLLM([VALID])
    result = await structured(client, costs).generate("system", "user", ClassificationOutput)
    assert result.relevance == 7
    assert '"relevance"' in client.requests[0].system  # schema is part of the prompt
    assert costs.run_calls == 1


async def test_invalid_output_gets_one_repair_round(costs: CostTracker) -> None:
    client = ScriptedLLM([{**VALID, "relevance": 42}, VALID])
    result = await structured(client, costs).generate("system", "user", ClassificationOutput)
    assert result.relevance == 7
    assert len(client.requests) == 2
    assert "could not be used" in client.requests[1].user
    assert "relevance" in client.requests[1].user


async def test_persistent_invalid_output_raises(costs: CostTracker) -> None:
    client = ScriptedLLM(["not json", "still not json"])
    with pytest.raises(StructuredOutputError):
        await structured(client, costs).generate("s", "u", ClassificationOutput)


async def test_transient_errors_are_retried(costs: CostTracker) -> None:
    client = ScriptedLLM([TransientLLMError("429"), TransientLLMError("503"), VALID])
    result = await structured(client, costs).generate("s", "u", ClassificationOutput)
    assert result.topic == "input_costs"
    assert costs.run_calls == 3
    assert costs.run_failed_calls == 2


async def test_other_errors_become_llm_errors(costs: CostTracker) -> None:
    client = ScriptedLLM([RuntimeError("bad request")])
    with pytest.raises(LLMError, match="bad request"):
        await structured(client, costs).generate("s", "u", ClassificationOutput)


async def test_budget_guard_blocks_calls() -> None:
    tracker = CostTracker({"test-model": Price(input_per_mtok=1e6, output_per_mtok=0)}, 1.0, "t")
    client = ScriptedLLM([VALID, VALID])
    llm = structured(client, tracker)
    await llm.generate("s", "u", ClassificationOutput)  # 100 input tokens -> $100
    with pytest.raises(BudgetExceededError):
        await llm.generate("s", "u", ClassificationOutput)
    assert len(client.requests) == 1


def test_cost_tracker_prices_and_persists() -> None:
    repo = SQLiteRepository(":memory:")
    tracker = CostTracker(
        {"m": Price(input_per_mtok=0.15, output_per_mtok=0.60)}, 1.0, "run-1", ledger=repo
    )
    cost = tracker.record("summarization", "openai", "m", 1_000_000, 500_000, ok=True)
    assert cost == pytest.approx(0.45)
    assert tracker.record("dedup", "openai", "unpriced", 10, 10, ok=True) == 0.0
    # a new tracker sees what was already spent today
    again = CostTracker({}, 1.0, "run-2", ledger=repo)
    assert again.spent_today_usd == pytest.approx(0.45)
    rows = repo.cost_report(since=datetime(2000, 1, 1))
    assert {r["stage"] for r in rows} == {"summarization", "dedup"}
