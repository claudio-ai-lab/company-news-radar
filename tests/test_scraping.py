import httpx

from newsradar.config import ScrapeConfig
from newsradar.scraping import Fetcher

ARTICLE = (
    "<html><head><title>Plant closure</title></head><body><nav>Menu Home About</nav>"
    "<article><h1>Foundry closes</h1>"
    + "".join(
        f"<p>Paragraph {i}: the casting supplier confirmed that production of aluminium "
        f"housings will move to another plant, and customers will receive transition "
        f"plans before the end of the year.</p>"
        for i in range(6)
    )
    + "</article><footer>Imprint</footer></body></html>"
)


def make_client(robots: str | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=robots) if robots else httpx.Response(404)
        if request.url.path == "/missing":
            return httpx.Response(404)
        if request.url.path == "/file.pdf":
            return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})
        return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_extracts_main_text() -> None:
    async with make_client() as client:
        result = await Fetcher(ScrapeConfig(delay_s=0, min_chars=200), client).fetch(
            "https://news.example.com/foundry"
        )
    assert result.error is None and result.text
    assert "aluminium housings" in result.text
    assert "Imprint" not in result.text


async def test_fetch_respects_robots_txt() -> None:
    robots = "User-agent: *\nDisallow: /private/\n"
    async with make_client(robots) as client:
        fetcher = Fetcher(ScrapeConfig(delay_s=0, min_chars=200), client)
        blocked = await fetcher.fetch("https://news.example.com/private/story")
        allowed = await fetcher.fetch("https://news.example.com/public/story")
    assert blocked.text is None and "robots" in (blocked.error or "")
    assert allowed.text


async def test_fetch_reports_http_errors_and_non_html() -> None:
    async with make_client() as client:
        fetcher = Fetcher(ScrapeConfig(delay_s=0), client)
        missing = await fetcher.fetch("https://news.example.com/missing")
        pdf = await fetcher.fetch("https://news.example.com/file.pdf")
    assert missing.text is None and "http error" in (missing.error or "")
    assert pdf.text is None and "HTML" in (pdf.error or "")
