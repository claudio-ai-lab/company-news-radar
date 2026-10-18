"""Domain models. The ``*Output`` models double as JSON schemas for structured LLM output."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Stage(StrEnum):
    """Lifecycle of an article in the store."""

    COLLECTED = "collected"
    REJECTED = "rejected"
    ACCEPTED = "accepted"
    FETCHED = "fetched"
    FETCH_FAILED = "fetch_failed"
    DUPLICATE = "duplicate"
    SUMMARIZED = "summarized"
    SUMMARY_FAILED = "summary_failed"


class ArticleMeta(BaseModel):
    """What a collector knows about an article before any LLM touches it."""

    url: str
    url_hash: str
    title: str = Field(min_length=1)
    source: str
    language: str = "en"
    priority: int = 2
    published_at: datetime | None = None
    teaser: str = ""
    feed_content: str = ""


# --------------------------------------------------------------- structured LLM outputs


class ClassificationOutput(BaseModel):
    """Headline screening result."""

    relevant: bool = Field(description="Whether the article belongs in the briefing")
    topic: str = Field(description="One of the configured topic ids, or 'none'")
    relevance: int = Field(ge=1, le=10, description="Relevance for the company on the 1-10 rubric")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this judgement")
    reason: str = Field(max_length=300, description="One short sentence")


class Entities(BaseModel):
    companies: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)


class SummaryOutput(BaseModel):
    """Grounded briefing entry for one article."""

    title: str = Field(max_length=200, description="Factual plain-text title")
    summary: str = Field(min_length=1, description="One or two short sentences, key fact first")
    key_points: list[str] = Field(
        default_factory=list, description="Additional facts not already in the summary"
    )
    impact: Literal["negative", "neutral", "positive"] = Field(
        default="neutral", description="Direction of the concrete effect on the reader"
    )
    entities: Entities = Field(default_factory=Entities)


class DuplicateVerdict(BaseModel):
    same_story: bool = Field(description="True if both articles report the same event")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=300)


class GroupOutput(BaseModel):
    label: str = Field(description="Short section heading")
    article_numbers: list[int] = Field(description="1-based numbers from the article list")


class DigestOutput(BaseModel):
    executive_summary: list[str] = Field(
        default_factory=list, description="Factual sentences that connect related stories"
    )
    groups: list[GroupOutput] = Field(default_factory=list)


# ------------------------------------------------------------------------ stored rows


class Article(BaseModel):
    id: int
    url: str
    url_hash: str
    title: str
    source: str
    language: str = "en"
    priority: int = 2
    published_at: datetime | None = None
    collected_at: datetime
    teaser: str = ""
    feed_content: str = ""
    stage: Stage = Stage.COLLECTED
    classification: ClassificationOutput | None = None
    content: str | None = None
    content_origin: Literal["scraped", "feed"] | None = None
    duplicate_of: str | None = None
    summary: SummaryOutput | None = None
    digest_date: date | None = None

    @property
    def relevance(self) -> int:
        return self.classification.relevance if self.classification else 0

    @property
    def timestamp(self) -> datetime:
        return self.published_at or self.collected_at


# ----------------------------------------------------------------------------- digest


class CoverageLink(BaseModel):
    source: str
    url: str


class DigestItem(BaseModel):
    article_id: int
    title: str
    url: str
    source: str
    published_at: datetime | None = None
    summary: str
    key_points: list[str] = Field(default_factory=list)
    impact: Literal["negative", "neutral", "positive"] = "neutral"
    topic: str
    relevance: int
    based_on_feed_text: bool = False
    also_covered_by: list[CoverageLink] = Field(default_factory=list)


class DigestGroup(BaseModel):
    label: str
    icon: str = "•"
    items: list[DigestItem]


class SourceStat(BaseModel):
    name: str
    ok: bool
    items: int = 0
    error: str | None = None


class Digest(BaseModel):
    title: str
    company: str
    digest_date: date
    generated_at: datetime
    executive_summary: list[str] = Field(default_factory=list)
    groups: list[DigestGroup] = Field(default_factory=list)
    language: str = "en"
    locale: str = "en"
    timezone: str = "UTC"
    negative_label: str = "Risk"
    positive_label: str = "Opportunity"
    analysis_fallback: bool = False
    stats: dict[str, float | int] = Field(default_factory=dict)
    sources: list[SourceStat] = Field(default_factory=list)

    @property
    def article_count(self) -> int:
        return sum(len(group.items) for group in self.groups)
