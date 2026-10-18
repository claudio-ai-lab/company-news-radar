from datetime import UTC, date, datetime, timedelta

from newsradar.models import ArticleMeta, Stage
from newsradar.storage.sqlite import SQLiteRepository, sqlite_path_from_url
from tests.conftest import classification, summary


def meta(n: int, **kw: object) -> ArticleMeta:
    data = {
        "url": f"https://example.com/{n}",
        "url_hash": f"h{n}",
        "title": f"Title {n}",
        "source": "Wire",
        "published_at": datetime(2026, 10, 16, tzinfo=UTC),
    }
    data.update(kw)
    return ArticleMeta(**data)


def test_sqlite_url_parsing() -> None:
    assert sqlite_path_from_url("sqlite:///data/x.db") == "data/x.db"
    assert sqlite_path_from_url("sqlite:///:memory:") == ":memory:"


def test_add_articles_is_idempotent(repo: SQLiteRepository) -> None:
    assert repo.add_articles([meta(1), meta(2)], "r1") == 2
    assert repo.add_articles([meta(2), meta(3)], "r2") == 1
    collected = repo.articles_in_stage(Stage.COLLECTED)
    assert [a.url_hash for a in collected] == ["h1", "h2", "h3"]
    assert collected[0].published_at == datetime(2026, 10, 16, tzinfo=UTC)


def test_article_lifecycle(repo: SQLiteRepository) -> None:
    repo.add_articles([meta(1), meta(2), meta(3)], "r1")
    a1, a2, a3 = repo.articles_in_stage(Stage.COLLECTED)
    repo.save_classification(a1.id, classification(relevance=8), accepted=True)
    repo.save_classification(a2.id, classification(relevance=3), accepted=False)
    repo.save_classification(a3.id, classification(relevance=7), accepted=True)
    assert [a.id for a in repo.articles_in_stage(Stage.ACCEPTED)] == [a1.id, a3.id]
    assert repo.articles_in_stage(Stage.REJECTED)[0].relevance == 3

    repo.save_content(a1.id, "Full text", "scraped")
    repo.mark_fetch_failed(a3.id, "timeout")
    fetched = repo.articles_in_stage(Stage.FETCHED)
    assert fetched[0].content == "Full text" and fetched[0].content_origin == "scraped"

    repo.save_summary(a1.id, summary())
    candidates = repo.digest_candidates()
    assert [a.id for a in candidates] == [a1.id]
    assert candidates[0].summary is not None
    assert repo.summarized_since(datetime.now(UTC) - timedelta(hours=1))[0].id == a1.id

    today = date(2026, 10, 16)
    repo.mark_digested([a1.id], today)
    assert repo.digest_candidates() == []
    assert repo.reset_digest(today) == 1
    assert len(repo.digest_candidates()) == 1


def test_duplicates_of(repo: SQLiteRepository) -> None:
    repo.add_articles([meta(1), meta(2), meta(3)], "r1")
    a1, a2, a3 = repo.articles_in_stage(Stage.COLLECTED)
    repo.mark_duplicate(a2.id, a1.url_hash)
    repo.mark_duplicate(a3.id, a1.url_hash)
    dupes = repo.duplicates_of([a1.url_hash, "unknown"])
    assert [a.id for a in dupes[a1.url_hash]] == [a2.id, a3.id]
    assert repo.duplicates_of([]) == {}


def test_cache_and_runs(repo: SQLiteRepository) -> None:
    assert repo.cache_get("ns", "k") is None
    repo.cache_put("ns", "k", "v1")
    repo.cache_put("ns", "k", "v2")
    assert repo.cache_get("ns", "k") == "v2"
    assert repo.cache_get("other", "k") is None
    repo.start_run("r1")
    repo.finish_run("r1", "ok", {"new": 3})
    repo.save_digest(date(2026, 10, 16), "r1", "{}")
    row = repo.conn.execute("SELECT status, stats_json FROM runs").fetchone()
    assert row["status"] == "ok" and '"new": 3' in row["stats_json"]
