import json
from datetime import UTC, date, datetime

import pytest

from newsradar import i18n
from newsradar.models import CoverageLink, Digest, DigestGroup, DigestItem
from newsradar.render import render_html, render_json, render_markdown


def digest(language: str = "en", locale: str = "en_GB") -> Digest:
    item = DigestItem(
        article_id=1,
        title='Port <script>alert("x")</script> strike',
        url="https://example.com/port",
        source="Wire",
        published_at=datetime(2026, 10, 16, 7, 10, tzinfo=UTC),
        summary="Traffic halted for **two days** <img src=x>.",
        key_points=["About *40* vessels waiting"],
        impact="negative",
        topic="Supply chain",
        relevance=9,
        also_covered_by=[CoverageLink(source="Ledger", url="https://example.org/port")],
    )
    return Digest(
        title="Example AG · Daily Radar",
        company="Example AG",
        digest_date=date(2026, 10, 16),
        generated_at=datetime(2026, 10, 16, 8, 0, tzinfo=UTC),
        executive_summary=["Dock workers strike."],
        groups=[DigestGroup(label="Logistics", icon="⛟", items=[item])],
        language=language,
        locale=locale,
        timezone="Europe/Berlin",
        stats={"stat_new": 1234, "stat_llm_cost": 0.0123},
    )


def test_html_escapes_untrusted_text_and_keeps_emphasis() -> None:
    html = render_html(digest())
    assert "<script>" not in html and "<img src=x>" not in html
    assert "<strong>two days</strong>" in html
    assert "<em>40</em>" in html
    assert "Also reported by" in html and "▼ Risk" in html
    assert "Friday, 16 October 2026" in html
    assert "1,234" in html  # en_GB number format


def test_markdown_and_json() -> None:
    md = render_markdown(digest())
    assert "## In brief" in md
    assert '[Port alert("x") strike](https://example.com/port)' not in md  # tags kept as text
    assert "https://example.com/port" in md
    data = json.loads(render_json(digest()))
    assert data["groups"][0]["items"][0]["relevance"] == 9


def test_german_output_uses_german_strings_and_locale() -> None:
    d = digest(language="de", locale="de_CH")
    d.negative_label = i18n.strings("de")["negative_label"]
    md = render_markdown(d)
    assert "## Das Wichtigste in Kürze" in md
    assert "Freitag, 16. Oktober 2026" in md
    assert "Ebenfalls berichtet von" in md
    assert "▼ Risiko" in md
    html = render_html(d)
    assert "1’234" in html  # Swiss grouping separator
    assert "16.10.26, 09:10" in html  # converted to Europe/Berlin


def test_i18n_helpers() -> None:
    assert i18n.language_name("de", "en") == "German"
    assert i18n.language_name("en", "de") == "Englisch"
    assert i18n.strings("xx") == i18n.strings("en")  # unknown language falls back
    assert i18n.Formatter("de_DE").number(1234567.891) == "1.234.567,89"
    with pytest.raises(ValueError):
        i18n.validate_locale("zz_ZZ_bogus")
    with pytest.raises(ValueError):
        i18n.validate_timezone("Mars/Olympus")
