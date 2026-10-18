"""Google-News-style sitemap collector (``<news:news>`` entries)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from xml.etree import ElementTree

from newsradar.config import Source
from newsradar.models import ArticleMeta
from newsradar.text import normalize_url, strip_html, truncate, url_hash

_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "news": "http://www.google.com/schemas/sitemap-news/0.9",
}


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_sitemap(
    raw: bytes | str, source: Source, now: datetime | None = None
) -> list[ArticleMeta]:
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=source.max_age_hours) if source.max_age_hours else None
    root = ElementTree.fromstring(raw)
    articles: list[ArticleMeta] = []
    for node in root.findall("sm:url", _NS)[: source.max_items]:
        loc = (node.findtext("sm:loc", default="", namespaces=_NS) or "").strip()
        title = strip_html(node.findtext("news:news/news:title", default="", namespaces=_NS))
        if not loc.startswith(("http://", "https://")) or not title:
            continue
        published = _parse_date(
            node.findtext("news:news/news:publication_date", default="", namespaces=_NS)
        )
        if cutoff and published and published < cutoff:
            continue
        articles.append(
            ArticleMeta(
                url=normalize_url(loc),
                url_hash=url_hash(loc),
                title=truncate(title, 300),
                source=source.name,
                language=source.language,
                priority=source.priority,
                published_at=published,
            )
        )
    return articles
