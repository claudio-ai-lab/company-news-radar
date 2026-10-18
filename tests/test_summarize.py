from newsradar.config import AppConfig
from newsradar.llm.costs import CostTracker
from newsradar.prompts import PromptLibrary
from newsradar.stages.summarize import Summarizer
from tests.conftest import ScriptedLLM, make_article, structured, summary

REPLY = {
    "title": "**Aluminium** prices jump 6%",
    "summary": "Aluminium prices rose **6%** after a smelter cut output.",
    "key_points": [
        "Aluminium prices rose 6% after a smelter cut output",  # restates the summary
        "- Mills plan *surcharges* for November",
        "Mills plan surcharges for November deliveries",  # near-duplicate of the previous
        "Premiums expected to stay high until January",
        "Electricity costs doubled since summer",
        "Traders expect volatility",
    ],
    "impact": "negative",
    "entities": {"companies": ["Smelter Co", "smelter co", " "], "people": [], "places": ["EU"]},
}


def summarizer(
    client: ScriptedLLM, config: AppConfig, costs: CostTracker, prompts: PromptLibrary, repo=None
) -> Summarizer:
    return Summarizer(structured(client, costs, "summarization"), prompts, config, repo)


async def test_postprocessing_enforces_the_rules(
    config: AppConfig, costs: CostTracker, prompts: PromptLibrary
) -> None:
    config.pipeline.summary.max_key_points = 3
    article = make_article(content="Long article text about aluminium.", language="en")
    result = await summarizer(ScriptedLLM([REPLY]), config, costs, prompts).summarize(article)
    assert result.title == "Aluminium prices jump 6%"
    assert result.key_points == [
        "Mills plan *surcharges* for November",
        "Premiums expected to stay high until January",
        "Electricity costs doubled since summer",
    ]
    assert result.entities.companies == ["Smelter Co"]
    assert result.impact == "negative"


async def test_prompt_contains_text_and_output_language(
    config: AppConfig, costs: CostTracker, prompts: PromptLibrary
) -> None:
    client = ScriptedLLM([summary().model_dump()])
    article = make_article(
        title="Hütte drosselt Produktion", content="Deutscher Text.", language="de"
    )
    await summarizer(client, config, costs, prompts).summarize(article)
    request = client.requests[0]
    assert "Always write in English" in request.system
    assert "Deutscher Text." in request.user
    assert "German" in request.user  # source language is named


async def test_summaries_are_cached_by_content(
    config: AppConfig, costs: CostTracker, prompts: PromptLibrary, repo
) -> None:
    client = ScriptedLLM([summary().model_dump(), summary().model_dump()])
    s = summarizer(client, config, costs, prompts, repo)
    await s.summarize(make_article(title="Smelter", content="Same text."))
    await s.summarize(make_article(title="Smelter", content="Same text."))  # other URL
    assert len(client.requests) == 1
    await s.summarize(make_article(title="Smelter", content="Changed text."))
    assert len(client.requests) == 2


async def test_falls_back_to_feed_text(
    config: AppConfig, costs: CostTracker, prompts: PromptLibrary
) -> None:
    s = summarizer(ScriptedLLM([]), config, costs, prompts)
    assert s.source_text(make_article(content=None, feed_content="Feed body")) == "Feed body"
    assert s.source_text(make_article(content=None, teaser="Teaser")) == "Teaser"
