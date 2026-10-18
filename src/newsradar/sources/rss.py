"""RSS/Atom collector."""

from __future__ import annotations

import calendar
from datetime import UTC, datetime, timedelta
from typing import Any

import feedparser

from newsradar.config import Source
from newsradar.models import ArticleMeta
from newsradar.text import normalize_url, strip_html, truncate, url_hash


def _entry_datetime(entry: Any) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            return datetime.fromtimestamp(calendar.timegm(parsed), tz=UTC)
    return None


def parse_feed(raw: bytes | str, source: Source, now: datetime | None = None) -> list[ArticleMeta]:
    """Turn raw feed XML into article metadata, applying the source's age limit."""
    now = now or datetime.now(UTC)
    feed = feedparser.parse(raw)
    cutoff = now - timedelta(hours=source.max_age_hours) if source.max_age_hours else None
    articles: list[ArticleMeta] = []
    seen: set[str] = set()
    for entry in feed.entries[: source.max_items]:
        link = (entry.get("link") or "").strip()
        title = strip_html(entry.get("title"))
        if not link.startswith(("http://", "https://")) or not title:
            continue
        published = _entry_datetime(entry)
        if cutoff and published and published < cutoff:
            continue
        digest = url_hash(link)
        if digest in seen:
            continue
        seen.add(digest)
        content = ""
        if entry.get("content"):
            content = strip_html(" ".join(part.get("value", "") for part in entry.content))
        articles.append(
            ArticleMeta(
                url=normalize_url(link),
                url_hash=digest,
                title=truncate(title, 300),
                source=source.name,
                language=source.language,
                priority=source.priority,
                published_at=published,
                teaser=truncate(strip_html(entry.get("summary")), 800),
                feed_content=content,
            )
        )
    return articles
