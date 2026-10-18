"""Language support: UI strings, default rubric/impact texts and locale-aware formatting.

Two languages are configured separately:

* the **output language** (``output.language``) selects the prompt set and the strings in
  the rendered digest; the model is told to write in it;
* each **source language** (``sources[].language``) tells the prompts which language an
  article is written in, so a German briefing can be built from English feeds.

Dates and numbers are formatted with Babel according to ``output.locale`` (for example
``de_CH`` gives ``1’234.5``, ``en_US`` gives ``1,234.5``).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from babel import Locale, UnknownLocaleError
from babel.dates import format_date, format_datetime
from babel.numbers import format_decimal

FALLBACK_LANGUAGE = "en"

STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "in_brief": "In brief",
        "articles_one": "article",
        "articles_many": "articles",
        "relevance": "relevance",
        "also_reported_by": "Also reported by",
        "feed_text_note": "Summarised from the feed text; the full article could not be fetched.",
        "fallback_note": "The overview could not be generated today; articles are grouped by topic.",
        "sources_unavailable": "Sources unavailable this run",
        "generated_by": "Generated {when} by company-news-radar. "
        "Summaries are machine-generated from the linked articles.",
        "other_news": "Other news",
        "negative_label": "Risk",
        "positive_label": "Opportunity",
        "stat_new": "new articles",
        "stat_screened": "screened",
        "stat_relevant": "relevant",
        "stat_duplicates": "duplicates merged",
        "stat_llm_calls": "LLM calls",
        "stat_llm_cost": "LLM cost (USD)",
    },
    "de": {
        "in_brief": "Das Wichtigste in Kürze",
        "articles_one": "Artikel",
        "articles_many": "Artikel",
        "relevance": "Relevanz",
        "also_reported_by": "Ebenfalls berichtet von",
        "feed_text_note": "Zusammengefasst aus dem Feed-Text; der vollständige Artikel war nicht abrufbar.",
        "fallback_note": "Die Übersicht konnte heute nicht erstellt werden; die Artikel sind nach Thema gruppiert.",
        "sources_unavailable": "Nicht erreichbare Quellen in diesem Lauf",
        "generated_by": "Erstellt {when} von company-news-radar. "
        "Die Zusammenfassungen sind maschinell aus den verlinkten Artikeln erzeugt.",
        "other_news": "Weitere Meldungen",
        "negative_label": "Risiko",
        "positive_label": "Chance",
        "stat_new": "neue Artikel",
        "stat_screened": "geprüft",
        "stat_relevant": "relevant",
        "stat_duplicates": "Duplikate zusammengeführt",
        "stat_llm_calls": "LLM-Aufrufe",
        "stat_llm_cost": "LLM-Kosten (USD)",
    },
}

DEFAULT_RUBRIC: dict[str, list[dict[str, str]]] = {
    "en": [
        {
            "score": "9-10",
            "meaning": "Direct, near-term effect on the company or a watchlist entity.",
        },
        {
            "score": "6-8",
            "meaning": "Concrete event in the company's markets, supply chain or rules it must follow.",
        },
        {"score": "3-5", "meaning": "Indirect or background signal for the company's industry."},
        {"score": "1-2", "meaning": "Unrelated to the company's business."},
    ],
    "de": [
        {
            "score": "9-10",
            "meaning": "Direkte, kurzfristige Auswirkung auf das Unternehmen oder eine Organisation der Beobachtungsliste.",
        },
        {
            "score": "6-8",
            "meaning": "Konkretes Ereignis in den Märkten, der Lieferkette oder den Regeln, die für das Unternehmen gelten.",
        },
        {
            "score": "3-5",
            "meaning": "Indirektes Signal oder Hintergrund zur Branche des Unternehmens.",
        },
        {"score": "1-2", "meaning": "Ohne Bezug zum Geschäft des Unternehmens."},
    ],
}

DEFAULT_IMPACT: dict[str, dict[str, str]] = {
    "en": {
        "negative": "a concrete adverse effect is described (disruption, higher cost, lost business)",
        "positive": "a concrete favourable effect is described (new business, lower cost, relief)",
    },
    "de": {
        "negative": "eine konkrete nachteilige Wirkung wird beschrieben (Störung, höhere Kosten, Geschäftsverlust)",
        "positive": "eine konkrete vorteilhafte Wirkung wird beschrieben (neues Geschäft, niedrigere Kosten, Entlastung)",
    },
}


def supported(language: str) -> str:
    """The language itself if built-in strings exist for it, else the fallback."""
    return language if language in STRINGS else FALLBACK_LANGUAGE


def strings(language: str) -> dict[str, str]:
    return STRINGS[supported(language)]


def validate_locale(value: str) -> str:
    try:
        Locale.parse(value)
    except (UnknownLocaleError, ValueError) as exc:
        raise ValueError(f"unknown locale {value!r}") from exc
    return value


def validate_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown time zone {value!r}") from exc
    return value


def language_name(code: str, in_language: str = "en") -> str:
    """Human-readable language name, e.g. ``language_name("de", "en") == "German"``."""
    try:
        name = Locale.parse(code).get_display_name(in_language)
    except (UnknownLocaleError, ValueError):
        return code
    return name or code


class Formatter:
    """Locale-aware formatting used by the renderers."""

    def __init__(self, locale: str, timezone: str = "UTC") -> None:
        self.locale = locale
        self.tz = ZoneInfo(timezone)

    def date(self, value: date, style: str = "full") -> str:
        return str(format_date(value, format=style, locale=self.locale))

    def datetime(self, value: datetime, style: str = "short") -> str:
        return str(format_datetime(value, format=style, tzinfo=self.tz, locale=self.locale))

    def number(self, value: Any, digits: int | None = None) -> str:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float) and digits is None:
            digits = 4 if abs(value) < 1 else 2
        pattern = "#,##0" + (f".{'0' * digits}" if digits else "")
        return str(format_decimal(value, format=pattern, locale=self.locale))
