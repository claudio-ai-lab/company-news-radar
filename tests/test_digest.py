from datetime import date

from newsradar.config import AppConfig
from newsradar.llm.base import LLMError
from newsradar.llm.costs import CostTracker
from newsradar.models import GroupOutput
from newsradar.prompts import PromptLibrary
from newsradar.stages.digest import DigestBuilder, clean_label, validate_groups
from tests.conftest import ScriptedLLM, classification, make_article, structured, summary


def groups(*specs: tuple[str, list[int]]) -> list[GroupOutput]:
    return [GroupOutput(label=label, article_numbers=numbers) for label, numbers in specs]


def test_validate_groups_repairs_the_partition() -> None:
    result = validate_groups(
        groups(("Metals", [1, 2, 2, 99]), ("Logistics", [2, 3]), ("Empty", [0])),
        article_count=5,
        max_groups=8,
        other_label="Other news",
    )
    assert result == [("Metals", [1, 2]), ("Logistics", [3]), ("Other news", [4, 5])]


def test_validate_groups_merges_surplus_groups() -> None:
    proposal = groups(("A", [1, 2, 3]), ("B", [4, 5]), ("C", [6]), ("D", [7]))
    result = validate_groups(proposal, article_count=7, max_groups=3, other_label="Other")
    assert result == [("A", [1, 2, 3]), ("B", [4, 5]), ("Other", [6, 7])]
    assert sorted(n for _, nums in result for n in nums) == list(range(1, 8))


def test_clean_label() -> None:
    assert clean_label("<b>Metals</b> &#9888; 📈 & energy") == "Metals & energy"
    assert len(clean_label("x" * 100)) <= 60


def articles():
    metal = make_article(
        title="Aluminium jumps",
        classification=classification(topic="input_costs", relevance=7),
        summary=summary(title="Aluminium prices jump 6%"),
    )
    port = make_article(
        title="Port strike",
        classification=classification(topic="supply_chain", relevance=9),
        summary=summary(title="Port strike halts traffic", impact="negative"),
        content_origin="feed",
    )
    dup = make_article(title="Port strike, day two", source="Ledger")
    return metal, port, dup


async def test_build_uses_model_grouping_and_orders_by_relevance(
    config: AppConfig, costs: CostTracker, prompts: PromptLibrary
) -> None:
    metal, port, dup = articles()
    reply = {
        "executive_summary": ["Dock workers **strike**; aluminium rises.", " "],
        # Numbers follow the ranking (most relevant first): 1 = port, 2 = metal.
        "groups": [
            {"label": "Metals", "article_numbers": [2]},
            {"label": "Logistics", "article_numbers": [1]},
        ],
    }
    client = ScriptedLLM([reply])
    builder = DigestBuilder(structured(client, costs, "digest"), prompts, config)
    digest = await builder.build([metal, port], {port.url_hash: [dup]}, date(2026, 10, 16))
    assert [g.label for g in digest.groups] == ["Logistics", "Metals"]
    assert digest.groups[0].icon == config.topic("supply_chain").icon
    item = digest.groups[0].items[0]
    assert item.also_covered_by[0].source == "Ledger"
    assert item.based_on_feed_text and item.topic == "Supply chain & logistics"
    assert digest.executive_summary == ["Dock workers strike; aluminium rises."]
    assert not digest.analysis_fallback
    assert "1. Port strike halts traffic" in client.requests[0].user


async def test_build_falls_back_to_topics_when_the_model_fails(
    config: AppConfig, costs: CostTracker, prompts: PromptLibrary
) -> None:
    metal, port, _ = articles()
    client = ScriptedLLM([LLMError("down")])
    builder = DigestBuilder(structured(client, costs, "digest"), prompts, config)
    digest = await builder.build([metal, port], {}, date(2026, 10, 16))
    assert digest.analysis_fallback
    assert digest.executive_summary == []
    assert {g.label for g in digest.groups} == {"Supply chain & logistics", "Materials & energy"}
    assert digest.article_count == 2
