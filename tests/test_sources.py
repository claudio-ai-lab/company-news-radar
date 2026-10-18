from datetime import UTC, datetime

import httpx

from newsradar.config import AppConfig, Source
from newsradar.sources import collect_all
from newsradar.sources.rss import parse_feed
from newsradar.sources.sitemap import parse_sitemap
from tests.conftest import EXAMPLE_DIR

WIRE = (EXAMPLE_DIR / "fixtures" / "example-wire.xml").read_bytes()


def test_parse_feed_extracts_metadata() -> None:
    source = Source(name="Wire", path="x.xml", max_age_hours=None, priority=1, language="en")
    items = parse_feed(WIRE, source)
    assert len(items) == 8
    first = items[0]
    assert first.title.startswith("Nordwerk Castings")
    assert "utm_source" not in first.url
    assert first.published_at == datetime(2026, 10, 16, 7, 10, tzinfo=UTC)
    assert "240 job cuts" in first.teaser
    assert first.feed_content.startswith("Nordwerk Castings will close")
    assert first.priority == 1 and first.source == "Wire"


def test_parse_feed_applies_age_limit() -> None:
    source = Source(name="Wire", path="x.xml", max_age_hours=24)
    assert parse_feed(WIRE, source, now=datetime(2026, 10, 16, 20, tzinfo=UTC))
    assert parse_feed(WIRE, source, now=datetime(2026, 12, 1, tzinfo=UTC)) == []


def test_parse_sitemap() -> None:
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
            xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
      <url><loc>https://example.com/a</loc>
        <news:news><news:publication_date>2026-10-16T08:00:00Z</news:publication_date>
        <news:title>Smelter cuts output</news:title></news:news></url>
      <url><loc>https://example.com/no-title</loc></url>
    </urlset>"""
    items = parse_sitemap(xml, Source(name="Map", type="sitemap", path="m.xml", max_age_hours=None))
    assert [i.title for i in items] == ["Smelter cuts output"]
    assert items[0].published_at == datetime(2026, 10, 16, 8, tzinfo=UTC)


async def test_collect_all_isolates_failing_sources(config: AppConfig) -> None:
    config.sources = [
        Source(name="Local", path="fixtures/example-wire.xml", max_age_hours=None),
        Source(name="Missing", path="fixtures/does-not-exist.xml"),
        Source(name="Remote", url="https://feeds.example.com/rss", max_age_hours=None),
        Source(name="Broken", url="https://broken.example.com/rss"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "feeds.example.com":
            return httpx.Response(200, content=WIRE)
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        metas, stats = await collect_all(config, client)
    by_name = {s.name: s for s in stats}
    assert by_name["Local"].ok and by_name["Local"].items == 8
    assert by_name["Remote"].ok
    assert not by_name["Missing"].ok and not by_name["Broken"].ok
    assert len(metas) == 8  # the same URLs from two sources are collected once
