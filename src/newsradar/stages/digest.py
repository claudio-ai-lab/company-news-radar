"""Stage 5: executive summary and thematic grouping for the day's entries.

The model proposes groups by article number; the code validates them. Unknown or
repeated numbers are dropped, articles the model forgot land in a catch-all group, and
surplus groups are merged. If the call fails (or the budget is spent) the digest is
still produced, grouped by topic and without an executive summary.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime

from newsradar.config import AppConfig
from newsradar.llm.base import LLMError, StructuredLLM
from newsradar.llm.costs import BudgetExceededError
from newsradar.markup import strip_inline
from newsradar.models import (
    Article,
    CoverageLink,
    Digest,
    DigestGroup,
    DigestItem,
    DigestOutput,
    GroupOutput,
    SourceStat,
)
from newsradar.prompts import PromptLibrary
from newsradar.text import strip_html, truncate

logger = logging.getLogger(__name__)


def clean_label(label: str, limit: int = 60) -> str:
    """Plain-text heading: no tags, entities, emoji or other symbol characters."""
    text = strip_html(label)
    text = re.sub(r"&#x?[0-9a-fA-F]+;", "", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in {"So", "Sk", "Cs", "Co"})
    return truncate(re.sub(r"\s+", " ", text).strip(" -:|"), limit)


def validate_groups(
    groups: Sequence[GroupOutput], article_count: int, max_groups: int, other_label: str
) -> list[tuple[str, list[int]]]:
    """Make the model's grouping a partition of 1..article_count with at most max_groups parts."""
    assigned: set[int] = set()
    result: list[tuple[str, list[int]]] = []
    for group in groups:
        numbers = [
            n
            for n in dict.fromkeys(group.article_numbers)
            if 1 <= n <= article_count and n not in assigned
        ]
        if not numbers:
            continue
        assigned.update(numbers)
        result.append((clean_label(group.label) or other_label, numbers))

    missing = [n for n in range(1, article_count + 1) if n not in assigned]
    if missing:
        result.append((other_label, missing))

    if len(result) > max_groups:
        ranked = sorted(result, key=lambda g: -len(g[1]))
        keep, merge = ranked[: max_groups - 1], ranked[max_groups - 1 :]
        result = [*keep, (other_label, sorted(n for _, nums in merge for n in nums))]

    # Merge several catch-all groups into one.
    merged: list[tuple[str, list[int]]] = []
    other: list[int] = []
    for label, numbers in result:
        if label == other_label:
            other.extend(numbers)
        else:
            merged.append((label, numbers))
    if other:
        merged.append((other_label, sorted(other)))
    return merged


class DigestBuilder:
    def __init__(self, llm: StructuredLLM, prompts: PromptLibrary, config: AppConfig) -> None:
        self.llm = llm
        self.prompts = prompts
        self.config = config
        self._system = prompts.system("digest")

    def _item(self, article: Article, duplicates: Sequence[Article]) -> DigestItem:
        assert article.summary is not None and article.classification is not None
        topic = self.config.topic(article.classification.topic)
        return DigestItem(
            article_id=article.id,
            title=article.summary.title or article.title,
            url=article.url,
            source=article.source,
            published_at=article.published_at,
            summary=article.summary.summary,
            key_points=article.summary.key_points,
            impact=article.summary.impact,
            topic=topic.label if topic else article.classification.topic,
            relevance=article.relevance,
            based_on_feed_text=article.content_origin == "feed",
            also_covered_by=[CoverageLink(source=d.source, url=d.url) for d in duplicates],
        )

    async def build(
        self,
        articles: Sequence[Article],
        duplicates: dict[str, list[Article]],
        digest_date: date,
        stats: dict[str, float | int] | None = None,
        sources: Sequence[SourceStat] = (),
    ) -> Digest:
        cfg = self.config.pipeline.digest
        ranked = sorted(
            (a for a in articles if a.summary and a.classification),
            key=lambda a: (-a.relevance, a.priority, -a.timestamp.timestamp(), a.id),
        )[: cfg.max_articles]
        items = [self._item(a, duplicates.get(a.url_hash, [])) for a in ranked]
        item_by_id = {item.article_id: item for item in items}

        analysis, fallback = await self._analyse(items)
        if fallback or not analysis.groups:
            proposals = self._groups_by_topic(ranked)
        else:
            proposals = analysis.groups
        partition = validate_groups(
            proposals, len(items), cfg.max_groups, self.config.other_group_label
        )

        groups: list[DigestGroup] = []
        for label, numbers in partition:
            members = sorted((ranked[n - 1] for n in numbers), key=lambda a: -a.relevance)
            groups.append(
                DigestGroup(
                    label=label,
                    icon=self._icon(members),
                    items=[item_by_id[a.id] for a in members],
                )
            )
        groups.sort(key=lambda g: -max(item.relevance for item in g.items))

        summary_sentences = [
            truncate(strip_inline(s).strip(), 400) for s in analysis.executive_summary if s.strip()
        ][: cfg.executive_summary_max]

        return Digest(
            title=self.config.digest_title,
            company=self.config.profile.company,
            digest_date=digest_date,
            generated_at=datetime.now(UTC),
            executive_summary=summary_sentences,
            groups=groups,
            language=self.config.language,
            locale=self.config.locale,
            timezone=self.config.output.timezone,
            negative_label=self.config.negative_label,
            positive_label=self.config.positive_label,
            analysis_fallback=fallback,
            stats=stats or {},
            sources=list(sources),
        )

    async def _analyse(self, items: Sequence[DigestItem]) -> tuple[DigestOutput, bool]:
        if not items:
            return DigestOutput(), False
        entries = [
            {
                "n": i,
                "title": item.title,
                "source": item.source,
                "topic": item.topic,
                "relevance": item.relevance,
                "summary": strip_inline(item.summary),
            }
            for i, item in enumerate(items, start=1)
        ]
        try:
            output = await self.llm.generate(
                self._system,
                self.prompts.user("digest", articles=entries),
                DigestOutput,
                payload={"articles": entries},
            )
        except (LLMError, BudgetExceededError) as exc:
            logger.warning("digest analysis unavailable, using topic grouping: %s", exc)
            return DigestOutput(), True
        return output, False

    def _groups_by_topic(self, ranked: Sequence[Article]) -> list[GroupOutput]:
        by_topic: dict[str, list[int]] = {}
        for n, article in enumerate(ranked, start=1):
            assert article.classification is not None
            by_topic.setdefault(article.classification.topic, []).append(n)
        groups = []
        for topic_id, numbers in by_topic.items():
            topic = self.config.topic(topic_id)
            groups.append(
                GroupOutput(label=topic.label if topic else topic_id, article_numbers=numbers)
            )
        return groups

    def _icon(self, members: Sequence[Article]) -> str:
        counts = Counter(a.classification.topic for a in members if a.classification)
        for topic_id, _ in counts.most_common():
            topic = self.config.topic(topic_id)
            if topic is not None:
                return topic.icon
        return "•"
