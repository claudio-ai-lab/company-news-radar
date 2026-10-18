"""Orchestrates one run: collect, screen, fetch, deduplicate, summarise, build the digest.

Every stage reads its input from the repository and writes its result back, so a run can
be interrupted at any point and the next run continues where this one stopped. Failures
are contained per item (a broken feed, an unreachable page, one invalid model answer)
and never abort the run. When the daily LLM budget is reached, the remaining LLM work
is deferred to the next run and the digest is built from what is ready.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from newsradar.config import AppConfig
from newsradar.delivery import send_email, write_outputs
from newsradar.llm.base import LLMError
from newsradar.llm.costs import BudgetExceededError, CostTracker
from newsradar.llm.factory import build_stage_llms
from newsradar.models import Article, Digest, SourceStat, Stage
from newsradar.prompts import PromptLibrary
from newsradar.render import render_json
from newsradar.scraping import Fetcher
from newsradar.sources import collect_all
from newsradar.stages import Classifier, Deduplicator, DigestBuilder, Summarizer
from newsradar.stages.dedup import Embedder, SentenceTransformerEmbedder
from newsradar.storage.base import Repository

logger = logging.getLogger(__name__)

MIN_FEED_TEXT_CHARS = 60


@dataclass
class RunOptions:
    offline: bool = False
    skip_collect: bool = False
    scrape: bool = True
    limit: int | None = None
    redo_digest: bool = False
    send_email: bool = True
    digest_date: date | None = None


@dataclass
class RunReport:
    run_id: str
    status: str
    stats: dict[str, Any] = field(default_factory=dict)
    outputs: list[Path] = field(default_factory=list)
    digest: Digest | None = None


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        repo: Repository,
        *,
        http_client: httpx.AsyncClient | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config
        self.repo = repo
        self._http_client = http_client
        self._embedder = embedder

    async def run(self, options: RunOptions | None = None) -> RunReport:
        options = options or RunOptions()
        run_id = new_run_id()
        self.repo.start_run(run_id)
        costs = CostTracker(
            self.config.llm.pricing, self.config.llm.daily_budget_usd, run_id, ledger=self.repo
        )
        llms = build_stage_llms(self.config, costs, offline=options.offline)
        prompts = PromptLibrary(self.config)
        report = RunReport(run_id=run_id, status="ok")
        stats: dict[str, Any] = report.stats
        source_stats: list[SourceStat] = []

        client = self._http_client or httpx.AsyncClient(
            timeout=self.config.pipeline.collect_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": self.config.pipeline.scrape.user_agent},
        )
        try:
            if not options.skip_collect:
                source_stats = await self._collect(client, run_id, stats)
            try:
                classifier = Classifier(llms["classification"], prompts, self.config, self.repo)
                await self._classify(classifier, options, stats)
                await self._fetch(client, options, stats)
                if self.config.pipeline.dedup.enabled:
                    dedup = Deduplicator(
                        llms["dedup"],
                        prompts,
                        self.config.pipeline.dedup,
                        self.repo,
                        embedder=self._embedder or self._default_embedder(),
                    )
                    await self._deduplicate(dedup, stats)
                summarizer = Summarizer(llms["summarization"], prompts, self.config, self.repo)
                await self._summarize(summarizer, stats)
            except BudgetExceededError as exc:
                logger.warning("%s; remaining LLM work is deferred to the next run", exc)
                stats["budget_exhausted"] = True
                report.status = "partial"

            builder = DigestBuilder(llms["digest"], prompts, self.config)
            await self._digest(builder, options, costs, stats, source_stats, report)
        except Exception:
            self.repo.finish_run(run_id, "failed", stats)
            raise
        finally:
            if self._http_client is None:
                await client.aclose()

        stats.update(
            llm_calls=costs.run_calls,
            llm_failed_calls=costs.run_failed_calls,
            llm_cost_usd=round(costs.run_cost_usd, 6),
        )
        self.repo.finish_run(run_id, report.status, stats)
        return report

    # ---------------------------------------------------------------------- stages

    async def _collect(
        self, client: httpx.AsyncClient, run_id: str, stats: dict[str, Any]
    ) -> list[SourceStat]:
        metas, source_stats = await collect_all(self.config, client)
        stats["sources_ok"] = sum(s.ok for s in source_stats)
        stats["sources_failed"] = sum(not s.ok for s in source_stats)
        stats["collected"] = len(metas)
        stats["new"] = self.repo.add_articles(metas, run_id)
        logger.info("collected %d items, %d new", len(metas), stats["new"])
        return source_stats

    async def _classify(
        self, classifier: Classifier, options: RunOptions, stats: dict[str, Any]
    ) -> None:
        limit = self.config.pipeline.max_classifications_per_run
        if options.limit is not None:
            limit = min(limit, options.limit)
        pending = self.repo.articles_in_stage(Stage.COLLECTED, limit=limit)
        results = await asyncio.gather(
            *(
                classifier.classify(
                    title=a.title, url=a.url, source=a.source, teaser=a.teaser, language=a.language
                )
                for a in pending
            ),
            return_exceptions=True,
        )
        accepted = rejected = 0
        budget_hit: BudgetExceededError | None = None
        for article, result in zip(pending, results, strict=True):
            if isinstance(result, BudgetExceededError):
                budget_hit = result  # article stays "collected" for the next run
                continue
            if isinstance(result, BaseException):
                raise result
            ok = classifier.accepts(result)
            self.repo.save_classification(article.id, result, ok)
            accepted += ok
            rejected += not ok
        stats.update(classified=accepted + rejected, accepted=accepted, rejected=rejected)
        logger.info("classified %d: %d accepted", accepted + rejected, accepted)
        if budget_hit is not None:
            raise budget_hit

    async def _fetch(
        self, client: httpx.AsyncClient, options: RunOptions, stats: dict[str, Any]
    ) -> None:
        accepted = self.repo.articles_in_stage(Stage.ACCEPTED)
        scrape = options.scrape and not options.offline and self.config.pipeline.scrape.enabled
        texts: list[str | None] = [None] * len(accepted)
        errors: list[str | None] = [None] * len(accepted)
        if scrape and accepted:
            fetcher = Fetcher(self.config.pipeline.scrape, client)
            results = await asyncio.gather(*(fetcher.fetch(a.url) for a in accepted))
            texts = [r.text for r in results]
            errors = [r.error for r in results]
        fetched = from_feed = failed = 0
        for article, text, error in zip(accepted, texts, errors, strict=True):
            if text:
                self.repo.save_content(article.id, text, "scraped")
                fetched += 1
                continue
            fallback = article.feed_content or article.teaser
            if len(fallback) >= MIN_FEED_TEXT_CHARS:
                self.repo.save_content(article.id, fallback, "feed")
                from_feed += 1
            else:
                self.repo.mark_fetch_failed(article.id, error or "no text available")
                failed += 1
        stats.update(fetched=fetched, from_feed_text=from_feed, fetch_failed=failed)

    async def _deduplicate(self, dedup: Deduplicator, stats: dict[str, Any]) -> None:
        fresh = self.repo.articles_in_stage(Stage.FETCHED)
        since = datetime.now(UTC) - timedelta(hours=self.config.pipeline.dedup.window_hours)
        reference = self.repo.summarized_since(since)
        result = await dedup.run(fresh, reference)
        for article_id, canonical in result.duplicates.items():
            self.repo.mark_duplicate(article_id, canonical)
        stats.update(
            dedup_candidate_pairs=result.candidate_pairs,
            duplicates=len(result.duplicates),
        )

    async def _summarize(self, summarizer: Summarizer, stats: dict[str, Any]) -> None:
        pending = self.repo.articles_in_stage(Stage.FETCHED)
        results = await asyncio.gather(
            *(summarizer.summarize(a) for a in pending), return_exceptions=True
        )
        done = failed = 0
        budget_hit: BudgetExceededError | None = None
        for article, result in zip(pending, results, strict=True):
            if isinstance(result, BudgetExceededError):
                budget_hit = result
            elif isinstance(result, LLMError):
                logger.warning("summary failed for %r: %s", article.title[:60], result)
                self.repo.mark_summary_failed(article.id, str(result))
                failed += 1
            elif isinstance(result, BaseException):
                raise result
            else:
                self.repo.save_summary(article.id, result)
                done += 1
        stats.update(summarized=done, summary_failed=failed)
        if budget_hit is not None:
            raise budget_hit

    async def _digest(
        self,
        builder: DigestBuilder,
        options: RunOptions,
        costs: CostTracker,
        stats: dict[str, Any],
        source_stats: list[SourceStat],
        report: RunReport,
    ) -> None:
        digest_date = options.digest_date or date.today()
        if options.redo_digest:
            self.repo.reset_digest(digest_date)
        candidates: list[Article] = self.repo.digest_candidates()
        if not candidates:
            logger.info("no new relevant articles; no digest written")
            stats["digest_articles"] = 0
            return
        duplicates = self.repo.duplicates_of([a.url_hash for a in candidates])
        # Keys are translated by the renderer (see newsradar.i18n).
        footer = {
            "stat_new": stats.get("new", 0),
            "stat_screened": stats.get("classified", 0),
            "stat_relevant": stats.get("accepted", 0),
            "stat_duplicates": stats.get("duplicates", 0),
            "stat_llm_calls": costs.run_calls,
            "stat_llm_cost": round(costs.run_cost_usd, 4),
        }
        digest = await builder.build(candidates, duplicates, digest_date, footer, source_stats)
        report.digest = digest
        report.outputs = write_outputs(digest, self.config.output)
        self.repo.save_digest(digest_date, report.run_id, render_json(digest))
        # Everything considered is marked, including articles below the top-N cut,
        # so that a busy day does not spill stale stories into tomorrow's digest.
        self.repo.mark_digested([a.id for a in candidates], digest_date)
        stats["digest_articles"] = digest.article_count

        email = self.config.output.email
        if email.enabled and options.send_email and not options.offline:
            try:
                send_email(digest, email)
                stats["email_sent"] = True
            except Exception as exc:
                logger.error("e-mail delivery failed: %s", exc)
                stats["email_sent"] = False
                report.status = "partial"

    def _default_embedder(self) -> Embedder | None:
        cfg = self.config.pipeline.dedup
        if not cfg.use_embeddings:
            return None
        try:
            return SentenceTransformerEmbedder(cfg.embedding_model)
        except ImportError:
            logger.warning(
                "use_embeddings is on but sentence-transformers is not installed "
                "(pip install 'company-news-radar[embeddings]'); continuing without"
            )
            return None
