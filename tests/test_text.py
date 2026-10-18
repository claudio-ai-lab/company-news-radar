from newsradar.text import (
    contains_term,
    entities,
    hamming,
    jaccard,
    normalize_url,
    simhash,
    slug_tokens,
    split_sentences,
    strip_html,
    tokens,
    truncate,
    url_hash,
)


def test_normalize_url_drops_tracking_fragment_and_trailing_slash() -> None:
    url = "HTTPS://Example.COM/News/Item/?utm_source=rss&id=7&fbclid=x#comments"
    assert normalize_url(url) == "https://example.com/News/Item?id=7"


def test_url_hash_is_stable_across_tracking_variants() -> None:
    assert url_hash("https://example.com/a?utm_medium=x") == url_hash("https://example.com/a/")
    assert url_hash("https://example.com/a") != url_hash("https://example.com/b")


def test_strip_html_and_truncate() -> None:
    assert strip_html("<p>Steel &amp; <b>aluminium</b></p>\n  prices") == "Steel & aluminium prices"
    assert truncate("one two three four", 10) == "one two…"
    assert truncate("short", 10) == "short"


def test_tokens_skip_stopwords_and_jaccard() -> None:
    a = tokens("The port strike halts container traffic")
    b = tokens("Container traffic halted by port strike")
    assert "the" not in a
    assert jaccard(a, b) > 0.5
    assert jaccard(set(), b) == 0.0


def test_simhash_separates_near_duplicates_from_unrelated_text() -> None:
    base = (
        "Nordwerk Castings will close its foundry in Saxony by the end of March and move "
        "production of aluminium housings to its plant in Poland, affecting 240 employees."
    )
    near = base.replace("240 employees", "about 240 employees")
    other = (
        "The regional football club won the cup final on penalties after a draw, its "
        "first trophy in twelve years, and supporters celebrated late into the night."
    )
    assert hamming(simhash(base), simhash(near)) < hamming(simhash(base), simhash(other))
    assert simhash("too short") == 0


def test_slug_tokens_and_entities() -> None:
    assert slug_tokens("https://example.com/news/2026/acme-closes-plant-123.html") == {
        "acme",
        "closes",
        "plant",
    }
    assert {"ecb", "2.0%"} <= entities("ECB keeps deposit rate at 2.0%")


def test_contains_term_matches_whole_words_only() -> None:
    assert contains_term("Dock workers strike again", "strike")
    assert not contains_term("Strikers score twice", "strike")
    assert contains_term("New export controls announced", "export controls")


def test_split_sentences() -> None:
    assert split_sentences("Prices rose 6%. Mills warned. Traders agree.") == [
        "Prices rose 6%.",
        "Mills warned.",
        "Traders agree.",
    ]
