"""Stage 4: one grounded briefing entry per article.

Summaries are cached by a hash of (prompt, model, title, text): unchanged articles are
never paid for twice, and a prompt or config change invalidates the cache automatically.
A deterministic post-processing step enforces what prompts can only ask for.
"""

from __future__ import annotations

import logging

from newsradar.config import AppConfig
from newsradar.llm.base import StructuredLLM
from newsradar.markup import balance_markers, strip_inline
from newsradar.models import Article, Entities, SummaryOutput
from newsradar.prompts import PromptLibrary
from newsradar.storage.base import Repository
from newsradar.text import jaccard, sha256, tokens, truncate

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "summary"
REDUNDANCY_THRESHOLD = 0.6


class Summarizer:
    def __init__(
        self,
        llm: StructuredLLM,
        prompts: PromptLibrary,
        config: AppConfig,
        repo: Repository | None = None,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.config = config
        self.repo = repo
        self._system = prompts.system("summarize")

    def source_text(self, article: Article) -> str:
        text = article.content or article.feed_content or article.teaser or article.title
        return truncate(text, self.config.pipeline.scrape.max_chars)

    async def summarize(self, article: Article) -> SummaryOutput:
        """Raises :class:`~newsradar.llm.base.LLMError` if no valid summary can be produced."""
        text = self.source_text(article)
        key = sha256(self._system, self.llm.model_id, article.title, text)
        if self.repo is not None and (cached := self.repo.cache_get(CACHE_NAMESPACE, key)):
            return SummaryOutput.model_validate_json(cached)
        payload = {"title": article.title, "source": article.source, "text": text}
        result = await self.llm.generate(
            self._system,
            self.prompts.user("summarize", article=article, text=text),
            SummaryOutput,
            payload=payload,
        )
        result = self.postprocess(result)
        if self.repo is not None:
            self.repo.cache_put(CACHE_NAMESPACE, key, result.model_dump_json())
        return result

    def postprocess(self, result: SummaryOutput) -> SummaryOutput:
        cfg = self.config.pipeline.summary
        summary = balance_markers(result.summary.strip())
        if len(strip_inline(summary)) > cfg.max_summary_chars * 1.5:
            summary = balance_markers(truncate(summary, int(cfg.max_summary_chars * 1.5)))
        summary_tokens = tokens(strip_inline(summary))

        key_points: list[str] = []
        seen: list[set[str]] = []
        for raw_point in result.key_points:
            point = balance_markers(raw_point.strip().lstrip("-•* ").strip())
            point_tokens = tokens(strip_inline(point))
            if not point or not point_tokens:
                continue
            # Drop points that merely restate the summary or an earlier point.
            covered = len(point_tokens & summary_tokens) / len(point_tokens)
            if covered >= REDUNDANCY_THRESHOLD:
                continue
            if any(jaccard(point_tokens, other) >= REDUNDANCY_THRESHOLD for other in seen):
                continue
            key_points.append(point)
            seen.append(point_tokens)
            if len(key_points) >= cfg.max_key_points:
                break

        return SummaryOutput(
            title=truncate(strip_inline(result.title).strip(), 200),
            summary=summary,
            key_points=key_points,
            impact=result.impact,
            entities=Entities(
                companies=_unique(result.entities.companies)[:3],
                people=_unique(result.entities.people)[:5],
                places=_unique(result.entities.places)[:5],
            ),
        )


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            out.append(clean)
    return out
