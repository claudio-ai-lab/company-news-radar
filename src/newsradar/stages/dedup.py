"""Stage 3: collapse articles from different outlets that report the same story.

Comparing every pair with an LLM is quadratic in cost. Instead, cheap signals pick
candidate pairs and only those are verified by the model:

1. URL slug overlap (syndicated copies often share the slug)
2. Title token Jaccard similarity
3. Shared named entities / numbers in the title
4. SimHash distance of the article text
5. Optional multilingual sentence embeddings (catches translations)

Confirmed pairs are clustered with union-find. Already-published articles win as the
canonical copy, so a story that appeared yesterday is not repeated today.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

from newsradar.config import DedupConfig
from newsradar.llm.base import LLMError, StructuredLLM
from newsradar.models import Article, DuplicateVerdict
from newsradar.prompts import PromptLibrary
from newsradar.storage.base import Repository
from newsradar.text import entities, hamming, jaccard, sha256, simhash, slug_tokens, tokens

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "dedup"


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Wraps ``sentence-transformers`` (optional extra ``[embeddings]``)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass
class DedupResult:
    duplicates: dict[int, str] = field(default_factory=dict)  # article id -> canonical hash
    pairs_considered: int = 0
    candidate_pairs: int = 0
    llm_verified_pairs: int = 0
    signal_hits: dict[str, int] = field(default_factory=dict)


class Deduplicator:
    def __init__(
        self,
        llm: StructuredLLM,
        prompts: PromptLibrary,
        config: DedupConfig,
        repo: Repository | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.config = config
        self.repo = repo
        self.embedder = embedder
        self._system = prompts.system("dedup")

    # ------------------------------------------------------------- candidate search

    def signals(
        self, a: Article, b: Article, vectors: dict[int, list[float]] | None = None
    ) -> list[str]:
        cfg = self.config
        fired: list[str] = []
        if jaccard(slug_tokens(a.url), slug_tokens(b.url)) >= cfg.url_slug_jaccard:
            fired.append("url_slug")
        if jaccard(tokens(a.title), tokens(b.title)) >= cfg.title_jaccard:
            fired.append("title_jaccard")
        if len(entities(a.title) & entities(b.title)) >= cfg.min_shared_entities:
            fired.append("entities")
        text_a, text_b = _body(a), _body(b)
        if text_a and text_b:
            hash_a, hash_b = simhash(text_a), simhash(text_b)
            if hash_a and hash_b and hamming(hash_a, hash_b) <= cfg.simhash_max_distance:
                fired.append("simhash")
        if (
            vectors
            and a.id in vectors
            and b.id in vectors
            and cosine(vectors[a.id], vectors[b.id]) >= cfg.embedding_similarity
        ):
            fired.append("embedding")
        return fired

    def candidate_pairs(
        self, new: Sequence[Article], reference: Sequence[Article]
    ) -> tuple[list[tuple[Article, Article, list[str]]], int]:
        window = timedelta(hours=self.config.window_hours)
        pool = list(new) + [r for r in reference if r.id not in {n.id for n in new}]
        vectors = self._embed(pool)
        pairs: list[tuple[Article, Article, list[str]]] = []
        considered = 0
        for i, a in enumerate(new):
            for b in [*new[i + 1 :], *pool[len(new) :]]:
                if abs(a.timestamp - b.timestamp) > window:
                    continue
                considered += 1
                fired = self.signals(a, b, vectors)
                if fired:
                    pairs.append((a, b, fired))
        pairs.sort(key=lambda p: -len(p[2]))
        return pairs, considered

    def _embed(self, articles: Sequence[Article]) -> dict[int, list[float]] | None:
        if self.embedder is None or not articles:
            return None
        try:
            vectors = self.embedder.encode([a.title for a in articles])
        except Exception as exc:  # embeddings are an optional booster, never a hard failure
            logger.warning("embedding failed, continuing without: %s", exc)
            return None
        return {a.id: v for a, v in zip(articles, vectors, strict=True)}

    # ----------------------------------------------------------------- verification

    async def verify(self, a: Article, b: Article) -> DuplicateVerdict | None:
        first, second = sorted((a, b), key=lambda x: x.url_hash)
        key = sha256(self._system, self.llm.model_id, first.url_hash, second.url_hash)
        if self.repo is not None and (cached := self.repo.cache_get(CACHE_NAMESPACE, key)):
            return DuplicateVerdict.model_validate_json(cached)
        payload = {"a": _describe(first), "b": _describe(second)}
        try:
            verdict = await self.llm.generate(
                self._system, self.prompts.user("dedup", **payload), DuplicateVerdict, payload
            )
        except LLMError as exc:
            logger.warning("duplicate check failed: %s", exc)
            return None
        if self.repo is not None:
            self.repo.cache_put(CACHE_NAMESPACE, key, verdict.model_dump_json())
        return verdict

    async def run(self, new: Sequence[Article], reference: Sequence[Article] = ()) -> DedupResult:
        result = DedupResult()
        if not new:
            return result
        pairs, result.pairs_considered = self.candidate_pairs(new, reference)
        result.candidate_pairs = len(pairs)
        for _, _, fired in pairs:
            for name in fired:
                result.signal_hits[name] = result.signal_hits.get(name, 0) + 1
        to_verify = pairs[: self.config.max_llm_pairs]
        if len(pairs) > len(to_verify):
            logger.info("dedup: verifying %d of %d candidate pairs", len(to_verify), len(pairs))
        verdicts = await asyncio.gather(*(self.verify(a, b) for a, b, _ in to_verify))
        result.llm_verified_pairs = len(to_verify)
        confirmed = [
            (a, b)
            for (a, b, _), verdict in zip(to_verify, verdicts, strict=True)
            if verdict is not None
            and verdict.same_story
            and verdict.confidence >= self.config.min_confidence
        ]
        new_ids = {a.id for a in new}
        for cluster in cluster_pairs(confirmed):
            canonical = choose_canonical(cluster, new_ids)
            for article in cluster:
                if article.id in new_ids and article.id != canonical.id:
                    result.duplicates[article.id] = canonical.url_hash
        logger.info(
            "dedup: %d pairs considered, %d candidates, %d duplicates",
            result.pairs_considered,
            result.candidate_pairs,
            len(result.duplicates),
        )
        return result


def cluster_pairs(pairs: Sequence[tuple[Article, Article]]) -> list[list[Article]]:
    """Union-find over confirmed duplicate pairs."""
    parent: dict[int, int] = {}
    by_id: dict[int, Article] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        by_id[a.id], by_id[b.id] = a, b
        root_a, root_b = find(a.id), find(b.id)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    clusters: dict[int, list[Article]] = {}
    for article_id, article in by_id.items():
        clusters.setdefault(find(article_id), []).append(article)
    return [members for members in clusters.values() if len(members) > 1]


def choose_canonical(cluster: Sequence[Article], new_ids: set[int]) -> Article:
    """Prefer an already-published article, then the higher-priority source, then the earliest."""
    return min(cluster, key=lambda a: (a.id in new_ids, a.priority, a.timestamp, a.id))


def _body(article: Article) -> str:
    return article.content or article.feed_content or ""


def _describe(article: Article) -> dict[str, str]:
    snippet = _body(article)[:300].strip() or article.teaser[:300]
    return {
        "title": article.title,
        "source": article.source,
        "language": article.language,
        "date": article.timestamp.date().isoformat(),
        "snippet": snippet,
    }
