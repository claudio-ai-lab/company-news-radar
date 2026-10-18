"""Polite full-text fetching: robots.txt, a per-host delay and main-text extraction."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from newsradar.config import ScrapeConfig

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    text: str | None
    error: str | None = None


class Fetcher:
    def __init__(self, config: ScrapeConfig, client: httpx.AsyncClient) -> None:
        self.config = config
        self.client = client
        self._robots: dict[str, RobotFileParser | None] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def fetch(self, url: str) -> FetchResult:
        host = urlparse(url).netloc
        if self.config.respect_robots_txt and not await self._allowed(url, host):
            return FetchResult(None, "disallowed by robots.txt")
        async with self._semaphore, self._host_lock(host):
            await self._wait_for_host(host)
            try:
                response = await self.client.get(
                    url, headers={"User-Agent": self.config.user_agent}
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                return FetchResult(None, f"http error: {exc}"[:200])
            finally:
                self._last_request[host] = time.monotonic()
        if "html" not in response.headers.get("content-type", "html"):
            return FetchResult(None, "not an HTML page")
        text = await asyncio.to_thread(_extract, response.text, url)
        if not text or len(text) < self.config.min_chars:
            return FetchResult(None, "no article text found")
        return FetchResult(text[: self.config.max_chars])

    def _host_lock(self, host: str) -> asyncio.Lock:
        return self._host_locks.setdefault(host, asyncio.Lock())

    async def _wait_for_host(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            remaining = self.config.delay_s - (time.monotonic() - last)
            if remaining > 0:
                await asyncio.sleep(remaining)

    async def _allowed(self, url: str, host: str) -> bool:
        if host not in self._robots:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{host}/robots.txt"
            parser: RobotFileParser | None = RobotFileParser()
            try:
                response = await self.client.get(
                    robots_url, headers={"User-Agent": self.config.user_agent}
                )
                if response.status_code >= 400:
                    parser = None  # no robots.txt: everything allowed
                else:
                    assert parser is not None
                    parser.parse(response.text.splitlines())
            except httpx.HTTPError:
                parser = None
            self._robots[host] = parser
        robots = self._robots[host]
        return robots is None or robots.can_fetch(self.config.user_agent, url)


def _extract(html: str, url: str) -> str | None:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    return str(text) if text else None
