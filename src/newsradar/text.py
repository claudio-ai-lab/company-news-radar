"""Text helpers: URL normalisation, hashing, tokenisation and cheap similarity signals."""

from __future__ import annotations

import hashlib
import html
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "cmpid",
        "ref",
    }
)

# Small multilingual stop-word list (EN/DE/FR/IT). Good enough for overlap signals;
# the LLM makes the final call on anything these signals flag.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from has have he her his in into is it its
    more new not of on or out over says said she than that the their them they this to
    up was were what when which who will with after amid about
    der die das den dem des ein eine einer eines und oder ist sind im in mit von zu zum
    zur für auf bei nach aus als auch sich nicht noch wird werden hat haben über
    le la les l un une des du de et ou est sont en pour sur dans par au aux qui que
    il lo gli i una di da del della e per con su che non
    """.split()  # noqa: SIM905 - compact on purpose
)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[^\W\d_]{3,}|\d+(?:[.,]\d+)?", re.UNICODE)
_ENTITY = re.compile(r"\b(?:[A-ZÄÖÜ][\w&'-]*[\w]|[A-Z]{2,})\b|\d+(?:[.,]\d+)?%?")
# A sentence ends at . ! ? followed by a capital, except after a number ("15. Dezember").
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!\d\.)\s+(?=[\"'A-ZÄÖÜ0-9])")
_SLUG_NOISE = frozenset({"html", "htm", "php", "aspx", "news", "article", "articles", "story"})


def normalize_url(url: str) -> str:
    """Lower-case scheme/host, drop tracking parameters, fragments and trailing slashes."""
    parsed = urlparse(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (parsed.scheme.lower(), parsed.netloc.lower(), path, "", urlencode(query), "")
    )


def sha256(*parts: str) -> str:
    """Stable SHA-256 over several string parts (separator-safe)."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def url_hash(url: str) -> str:
    """Identity of an article: hash of its normalised URL."""
    return sha256(normalize_url(url))


def strip_html(text: str | None) -> str:
    """Remove tags, unescape entities and collapse whitespace."""
    if not text:
        return ""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", text))).strip()


def truncate(text: str, limit: int) -> str:
    """Cut text at a word boundary, adding an ellipsis when shortened."""
    if len(text) <= limit:
        return text
    cut = text[: max(limit - 1, 0)].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + "…"


def tokens(text: str) -> set[str]:
    """Lower-cased content words (stop words removed)."""
    return {t.lower() for t in _WORD.findall(text) if t.lower() not in STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def slug_tokens(url: str) -> set[str]:
    """Meaningful tokens from a URL path, e.g. /business/acme-closes-plant-1234."""
    path = urlparse(url).path.lower()
    return {
        token
        for token in re.split(r"[/\-_.]+", path)
        if len(token) >= 3 and not token.isdigit() and token not in _SLUG_NOISE
    }


def entities(text: str) -> set[str]:
    """Capitalised words, acronyms and numbers: a rough proxy for named entities."""
    found: set[str] = set()
    for match in _ENTITY.findall(text):
        low = match.lower()
        if len(low) >= 2 and low not in STOPWORDS:
            found.add(low)
    return found


def simhash(text: str, bits: int = 64) -> int:
    """Locality-sensitive fingerprint over word 3-grams; similar texts differ in few bits."""
    words = _WS.sub(" ", text.lower()).split()
    if len(words) < 3:
        return 0
    vector = [0] * bits
    for i in range(len(words) - 2):
        shingle = " ".join(words[i : i + 3]).encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big")
        for bit in range(bits):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    return sum(1 << bit for bit in range(bits) if vector[bit] > 0)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def contains_term(text: str, term: str) -> bool:
    """Case-insensitive whole-word/phrase match."""
    pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None
