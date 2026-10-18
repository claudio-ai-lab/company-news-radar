"""Storage interface. The pipeline only talks to this protocol, so another backend
(PostgreSQL, a document store, ...) can be added without touching the stages."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Protocol

from newsradar.models import Article, ArticleMeta, ClassificationOutput, Stage, SummaryOutput


class Repository(Protocol):
    # articles ------------------------------------------------------------------------
    def add_articles(self, metas: Sequence[ArticleMeta], run_id: str) -> int:
        """Insert unseen articles (by URL hash); return how many were new."""
        ...

    def articles_in_stage(self, stage: Stage, limit: int | None = None) -> list[Article]: ...

    def save_classification(
        self, article_id: int, result: ClassificationOutput, accepted: bool
    ) -> None: ...

    def save_content(self, article_id: int, content: str, origin: str) -> None: ...

    def mark_fetch_failed(self, article_id: int, error: str) -> None: ...

    def mark_duplicate(self, article_id: int, canonical_hash: str) -> None: ...

    def save_summary(self, article_id: int, summary: SummaryOutput) -> None: ...

    def mark_summary_failed(self, article_id: int, error: str) -> None: ...

    def summarized_since(self, since: datetime) -> list[Article]: ...

    def digest_candidates(self) -> list[Article]: ...

    def duplicates_of(self, canonical_hashes: Sequence[str]) -> dict[str, list[Article]]: ...

    def mark_digested(self, article_ids: Sequence[int], digest_date: date) -> None: ...

    def reset_digest(self, digest_date: date) -> int: ...

    # caches --------------------------------------------------------------------------
    def cache_get(self, namespace: str, key: str) -> str | None: ...

    def cache_put(self, namespace: str, key: str, value: str) -> None: ...

    # accounting & runs ---------------------------------------------------------------
    def record_llm_call(
        self,
        run_id: str,
        stage: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        ok: bool,
        error: str | None,
    ) -> None: ...

    def llm_cost_since(self, since: datetime) -> float: ...

    def cost_report(self, since: datetime) -> list[dict[str, Any]]: ...

    def start_run(self, run_id: str) -> None: ...

    def finish_run(self, run_id: str, status: str, stats: dict[str, Any]) -> None: ...

    def save_digest(self, digest_date: date, run_id: str, payload_json: str) -> None: ...

    def close(self) -> None: ...
