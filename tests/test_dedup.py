from datetime import UTC, datetime, timedelta

from newsradar.config import AppConfig, DedupConfig
from newsradar.llm.costs import CostTracker
from newsradar.models import Article
from newsradar.prompts import PromptLibrary
from newsradar.stages.dedup import Deduplicator, choose_canonical, cluster_pairs
from tests.conftest import ScriptedLLM, make_article, structured

SAME = {"same_story": True, "confidence": 0.9, "reason": "same closure"}


def pair() -> tuple[Article, Article]:
    a = make_article(
        title="Nordwerk Castings to close Saxony foundry",
        url="https://wire.example.com/business/nordwerk-castings-close-saxony-foundry",
        source="Wire",
        priority=2,
    )
    b = make_article(
        title="Nordwerk Castings confirms closure of Saxony foundry",
        url="https://ledger.example.org/2026/10/nordwerk-castings-closure-saxony-foundry",
        source="Ledger",
        priority=3,
    )
    return a, b


def unrelated() -> Article:
    return make_article(
        title="Celebrity chef opens restaurant", url="https://wire.example.com/life/chef-opens"
    )


def dedup(
    client: ScriptedLLM, costs: CostTracker, prompts: PromptLibrary, **cfg: object
) -> Deduplicator:
    return Deduplicator(structured(client, costs, "dedup"), prompts, DedupConfig(**cfg))


async def test_only_candidate_pairs_reach_the_llm(
    costs: CostTracker, prompts: PromptLibrary
) -> None:
    a, b = pair()
    client = ScriptedLLM(lambda request: SAME)
    result = await dedup(client, costs, prompts).run([a, b, unrelated()])
    assert result.pairs_considered == 3
    assert result.candidate_pairs == 1
    assert len(client.requests) == 1
    assert result.duplicates == {b.id: a.url_hash}  # higher-priority source is canonical
    assert {"url_slug", "title_jaccard"} & set(result.signal_hits)


async def test_low_confidence_or_negative_verdicts_keep_both(
    costs: CostTracker, prompts: PromptLibrary
) -> None:
    a, b = pair()
    unsure = ScriptedLLM([{**SAME, "confidence": 0.5}])
    assert (await dedup(unsure, costs, prompts).run([a, b])).duplicates == {}
    different = ScriptedLLM([{**SAME, "same_story": False}])
    assert (await dedup(different, costs, prompts).run([a, b])).duplicates == {}


async def test_already_published_article_stays_canonical(
    costs: CostTracker, prompts: PromptLibrary
) -> None:
    a, b = pair()
    # b was summarised in an earlier run; a is new and comes from a better source
    result = await dedup(ScriptedLLM([SAME]), costs, prompts).run([a], reference=[b])
    assert result.duplicates == {a.id: b.url_hash}


async def test_time_window_and_pair_cap(costs: CostTracker, prompts: PromptLibrary) -> None:
    a, b = pair()
    late = b.model_copy(update={"published_at": a.timestamp + timedelta(days=5)})
    client = ScriptedLLM([])
    assert (await dedup(client, costs, prompts).run([a, late])).candidate_pairs == 0
    capped = await dedup(client, costs, prompts, max_llm_pairs=0).run([a, b])
    assert capped.candidate_pairs == 1 and capped.duplicates == {} and not client.requests


async def test_embeddings_catch_translations(costs: CostTracker, prompts: PromptLibrary) -> None:
    english = make_article(title="Smelter cuts output by a third", url="https://a.example/x")
    german = make_article(
        title="Hütte drosselt Produktion um ein Drittel", url="https://b.example/y"
    )

    class FakeEmbedder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    detector = Deduplicator(
        structured(ScriptedLLM([SAME]), costs, "dedup"),
        prompts,
        DedupConfig(),
        embedder=FakeEmbedder(),
    )
    result = await detector.run([english, german])
    assert result.signal_hits.get("embedding") == 1
    assert len(result.duplicates) == 1


async def test_verdicts_are_cached(
    config: AppConfig, costs: CostTracker, prompts: PromptLibrary, repo
) -> None:
    a, b = pair()
    client = ScriptedLLM([SAME])
    detector = Deduplicator(structured(client, costs, "dedup"), prompts, DedupConfig(), repo)
    await detector.run([a, b])
    await detector.run([a, b])
    assert len(client.requests) == 1


def test_clustering_is_transitive() -> None:
    a, b, c, d = (make_article() for _ in range(4))
    clusters = cluster_pairs([(a, b), (b, c)])
    assert len(clusters) == 1 and {x.id for x in clusters[0]} == {a.id, b.id, c.id}
    assert d.id not in {x.id for x in clusters[0]}


def test_canonical_choice() -> None:
    early = make_article(priority=2, published_at=datetime(2026, 10, 16, 6, tzinfo=UTC))
    late_better = make_article(priority=1, published_at=datetime(2026, 10, 16, 9, tzinfo=UTC))
    assert choose_canonical([early, late_better], {early.id, late_better.id}) is late_better
    assert choose_canonical([early, late_better], {late_better.id}) is early
