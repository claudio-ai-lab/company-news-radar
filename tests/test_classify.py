from newsradar.config import AppConfig
from newsradar.llm.base import TransientLLMError
from newsradar.llm.costs import CostTracker
from newsradar.prompts import PromptLibrary
from newsradar.stages.classify import Classifier
from newsradar.storage.sqlite import SQLiteRepository
from tests.conftest import ScriptedLLM, classification, structured

ARTICLE = {
    "title": "Aluminium prices jump 6%",
    "url": "https://example.com/a?utm_source=x",
    "source": "Wire",
    "teaser": "Smelter cuts output.",
}


def reply(**kw: object) -> dict:
    return classification(**kw).model_dump()


async def test_classifier_prompt_carries_profile_and_article(
    config: AppConfig, prompts: PromptLibrary, costs: CostTracker
) -> None:
    client = ScriptedLLM([reply()])
    classifier = Classifier(structured(client, costs), prompts, config)
    await classifier.classify(**ARTICLE)
    request = client.requests[0]
    assert "Example AG" in request.system
    assert "input_costs" in request.system  # topic ids from the config
    assert "Nordwerk Castings" in request.system  # watchlist from the config
    assert "Aluminium prices jump 6%" in request.user


async def test_gate_and_cache(
    config: AppConfig, prompts: PromptLibrary, costs: CostTracker, repo: SQLiteRepository
) -> None:
    client = ScriptedLLM([reply(relevance=6)])
    classifier = Classifier(structured(client, costs), prompts, config, repo)
    first = await classifier.classify(**ARTICLE)
    assert classifier.accepts(first)

    # Raising the threshold changes the decision without re-asking the model.
    config.relevance.threshold = 7
    second = await classifier.classify(**{**ARTICLE, "url": "https://example.com/a"})
    assert second == first
    assert not classifier.accepts(second)
    assert len(client.requests) == 1


async def test_gate_rules(config: AppConfig, prompts: PromptLibrary, costs: CostTracker) -> None:
    classifier = Classifier(structured(ScriptedLLM([]), costs), prompts, config)
    config.relevance.min_confidence = 0.5
    assert classifier.accepts(classification(relevance=8))
    assert not classifier.accepts(classification(relevant=False, relevance=8))
    assert not classifier.accepts(classification(topic="none", relevance=8))
    assert not classifier.accepts(classification(relevance=5))
    assert not classifier.accepts(classification(relevance=9, confidence=0.4))


async def test_unknown_topic_is_rejected(
    config: AppConfig, prompts: PromptLibrary, costs: CostTracker
) -> None:
    client = ScriptedLLM([reply(topic="celebrity_gossip", relevance=9)])
    result = await Classifier(structured(client, costs), prompts, config).classify(**ARTICLE)
    assert result.topic == "none" and not result.relevant


async def test_failures_fail_closed_and_are_not_cached(
    config: AppConfig, prompts: PromptLibrary, costs: CostTracker, repo: SQLiteRepository
) -> None:
    client = ScriptedLLM([TransientLLMError("503")] * 3 + [reply()])
    classifier = Classifier(structured(client, costs), prompts, config, repo)
    failed = await classifier.classify(**ARTICLE)
    assert not classifier.accepts(failed) and "failed" in failed.reason
    retried = await classifier.classify(**ARTICLE)
    assert classifier.accepts(retried)
