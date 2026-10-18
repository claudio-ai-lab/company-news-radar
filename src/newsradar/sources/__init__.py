"""Collect article metadata from all enabled sources, isolating per-source failures."""

from __future__ import annotations

import asyncio
import logging

import httpx

from newsradar.config import AppConfig, Source
from newsradar.models import ArticleMeta, SourceStat
from newsradar.sources.rss import parse_feed
from newsradar.sources.sitemap import parse_sitemap

logger = logging.getLogger(__name__)


async def _read_source(source: Source, config: AppConfig, client: httpx.AsyncClient) -> bytes:
    if source.path is not None:
        return config.resolve(source.path).read_bytes()
    assert source.url is not None
    response = await client.get(source.url)
    response.raise_for_status()
    return response.content


async def collect_source(
    source: Source, config: AppConfig, client: httpx.AsyncClient
) -> tuple[list[ArticleMeta], SourceStat]:
    try:
        raw = await _read_source(source, config, client)
        parser = parse_feed if source.type == "rss" else parse_sitemap
        items = parser(raw, source)
    except Exception as exc:  # one broken feed must not stop the run
        logger.warning("source %r failed: %s", source.name, exc)
        return [], SourceStat(name=source.name, ok=False, error=str(exc)[:200])
    logger.info("source %r: %d items", source.name, len(items))
    return items, SourceStat(name=source.name, ok=True, items=len(items))


async def collect_all(
    config: AppConfig, client: httpx.AsyncClient
) -> tuple[list[ArticleMeta], list[SourceStat]]:
    sources = [s for s in config.sources if s.enabled]
    results = await asyncio.gather(*(collect_source(s, config, client) for s in sources))
    metas: list[ArticleMeta] = []
    seen: set[str] = set()
    for items, _ in results:
        for item in items:
            if item.url_hash not in seen:
                seen.add(item.url_hash)
                metas.append(item)
    return metas, [stat for _, stat in results]


__all__ = ["collect_all", "collect_source", "parse_feed", "parse_sitemap"]
