"""SQLite implementation of :class:`~newsradar.storage.base.Repository` (standard library only)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from newsradar.models import Article, ArticleMeta, ClassificationOutput, Stage, SummaryOutput

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    url_hash            TEXT NOT NULL UNIQUE,
    url                 TEXT NOT NULL,
    title               TEXT NOT NULL,
    source              TEXT NOT NULL,
    language            TEXT NOT NULL DEFAULT 'en',
    priority            INTEGER NOT NULL DEFAULT 2,
    published_at        TEXT,
    collected_at        TEXT NOT NULL,
    teaser              TEXT NOT NULL DEFAULT '',
    feed_content        TEXT NOT NULL DEFAULT '',
    stage               TEXT NOT NULL DEFAULT 'collected',
    classification_json TEXT,
    relevance           INTEGER,
    topic               TEXT,
    content             TEXT,
    content_origin      TEXT,
    error               TEXT,
    duplicate_of        TEXT,
    summary_json        TEXT,
    summarized_at       TEXT,
    digest_date         TEXT,
    run_id              TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_stage ON articles(stage);
CREATE INDEX IF NOT EXISTS idx_articles_digest_date ON articles(digest_date);
CREATE INDEX IF NOT EXISTS idx_articles_duplicate_of ON articles(duplicate_of);

CREATE TABLE IF NOT EXISTS cache (
    namespace  TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (namespace, key)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT,
    stage         TEXT NOT NULL,
    provider      TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd      REAL NOT NULL,
    ok            INTEGER NOT NULL,
    error         TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,
    stats_json  TEXT
);

CREATE TABLE IF NOT EXISTS digests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    digest_date  TEXT NOT NULL,
    run_id       TEXT,
    payload_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def sqlite_path_from_url(url: str) -> str:
    """``sqlite:///data/x.db`` -> ``data/x.db``; ``sqlite:///:memory:`` -> ``:memory:``."""
    if not url.startswith("sqlite:///"):
        raise ValueError(f"unsupported storage url {url!r} (expected sqlite:///...)")
    return url[len("sqlite:///") :]


class SQLiteRepository:
    def __init__(self, path: str | Path) -> None:
        path_str = str(path)
        if path_str != ":memory:":
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path_str)
        self.conn.row_factory = sqlite3.Row
        if path_str != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    # ------------------------------------------------------------------ articles

    def add_articles(self, metas: Sequence[ArticleMeta], run_id: str) -> int:
        before = self.conn.total_changes
        now = _now()
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO articles
                (url_hash, url, title, source, language, priority, published_at,
                 collected_at, teaser, feed_content, stage, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'collected', ?)
            """,
            [
                (
                    m.url_hash,
                    m.url,
                    m.title,
                    m.source,
                    m.language,
                    m.priority,
                    _iso(m.published_at),
                    now,
                    m.teaser,
                    m.feed_content,
                    run_id,
                )
                for m in metas
            ],
        )
        self.conn.commit()
        return self.conn.total_changes - before

    def articles_in_stage(self, stage: Stage, limit: int | None = None) -> list[Article]:
        sql = "SELECT * FROM articles WHERE stage = ? ORDER BY priority, collected_at, id"
        params: tuple[Any, ...] = (stage.value,)
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        return [self._to_article(row) for row in self.conn.execute(sql, params)]

    def save_classification(
        self, article_id: int, result: ClassificationOutput, accepted: bool
    ) -> None:
        stage = Stage.ACCEPTED if accepted else Stage.REJECTED
        self.conn.execute(
            """UPDATE articles SET classification_json = ?, relevance = ?, topic = ?, stage = ?
               WHERE id = ?""",
            (result.model_dump_json(), result.relevance, result.topic, stage.value, article_id),
        )
        self.conn.commit()

    def save_content(self, article_id: int, content: str, origin: str) -> None:
        self.conn.execute(
            "UPDATE articles SET content = ?, content_origin = ?, stage = ?, error = NULL WHERE id = ?",
            (content, origin, Stage.FETCHED.value, article_id),
        )
        self.conn.commit()

    def mark_fetch_failed(self, article_id: int, error: str) -> None:
        self._set_error(article_id, Stage.FETCH_FAILED, error)

    def mark_duplicate(self, article_id: int, canonical_hash: str) -> None:
        self.conn.execute(
            "UPDATE articles SET duplicate_of = ?, stage = ? WHERE id = ?",
            (canonical_hash, Stage.DUPLICATE.value, article_id),
        )
        self.conn.commit()

    def save_summary(self, article_id: int, summary: SummaryOutput) -> None:
        self.conn.execute(
            "UPDATE articles SET summary_json = ?, summarized_at = ?, stage = ? WHERE id = ?",
            (summary.model_dump_json(), _now(), Stage.SUMMARIZED.value, article_id),
        )
        self.conn.commit()

    def mark_summary_failed(self, article_id: int, error: str) -> None:
        self._set_error(article_id, Stage.SUMMARY_FAILED, error)

    def summarized_since(self, since: datetime) -> list[Article]:
        rows = self.conn.execute(
            "SELECT * FROM articles WHERE stage = ? AND summarized_at >= ? ORDER BY id",
            (Stage.SUMMARIZED.value, _iso(since)),
        )
        return [self._to_article(row) for row in rows]

    def digest_candidates(self) -> list[Article]:
        rows = self.conn.execute(
            """SELECT * FROM articles WHERE stage = ? AND digest_date IS NULL
               ORDER BY relevance DESC, priority, id""",
            (Stage.SUMMARIZED.value,),
        )
        return [self._to_article(row) for row in rows]

    def duplicates_of(self, canonical_hashes: Sequence[str]) -> dict[str, list[Article]]:
        result: dict[str, list[Article]] = {}
        if not canonical_hashes:
            return result
        placeholders = ",".join("?" for _ in canonical_hashes)
        rows = self.conn.execute(
            f"SELECT * FROM articles WHERE duplicate_of IN ({placeholders}) ORDER BY priority, id",
            tuple(canonical_hashes),
        )
        for row in rows:
            article = self._to_article(row)
            result.setdefault(article.duplicate_of or "", []).append(article)
        return result

    def mark_digested(self, article_ids: Sequence[int], digest_date: date) -> None:
        self.conn.executemany(
            "UPDATE articles SET digest_date = ? WHERE id = ?",
            [(digest_date.isoformat(), article_id) for article_id in article_ids],
        )
        self.conn.commit()

    def reset_digest(self, digest_date: date) -> int:
        cursor = self.conn.execute(
            "UPDATE articles SET digest_date = NULL WHERE digest_date = ?",
            (digest_date.isoformat(),),
        )
        self.conn.commit()
        return cursor.rowcount

    # -------------------------------------------------------------------- caches

    def cache_get(self, namespace: str, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM cache WHERE namespace = ? AND key = ?", (namespace, key)
        ).fetchone()
        return str(row["value"]) if row else None

    def cache_put(self, namespace: str, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (namespace, key, value, created_at) VALUES (?, ?, ?, ?)",
            (namespace, key, value, _now()),
        )
        self.conn.commit()

    # -------------------------------------------------------- accounting & runs

    def record_llm_call(
        self,
        run_id: str,
        stage: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        ok: bool,
        error: str | None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO llm_calls (run_id, stage, provider, model, input_tokens,
                   output_tokens, cost_usd, ok, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, stage, provider, model, input_tokens, output_tokens, cost_usd, int(ok),
             error, _now()),
        )  # fmt: skip
        self.conn.commit()

    def llm_cost_since(self, since: datetime) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM llm_calls WHERE created_at >= ?",
            (_iso(since),),
        ).fetchone()
        return float(row["total"])

    def cost_report(self, since: datetime) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT substr(created_at, 1, 10) AS day, stage, provider, model,
                      COUNT(*) AS calls, SUM(1 - ok) AS failed,
                      SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
                      SUM(cost_usd) AS cost_usd
               FROM llm_calls WHERE created_at >= ?
               GROUP BY day, stage, provider, model ORDER BY day, stage""",
            (_iso(since),),
        )
        return [dict(row) for row in rows]

    def start_run(self, run_id: str) -> None:
        self.conn.execute(
            "INSERT INTO runs (run_id, started_at, status) VALUES (?, ?, 'running')",
            (run_id, _now()),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str, stats: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, stats_json = ? WHERE run_id = ?",
            (_now(), status, json.dumps(stats, default=str), run_id),
        )
        self.conn.commit()

    def save_digest(self, digest_date: date, run_id: str, payload_json: str) -> None:
        self.conn.execute(
            "INSERT INTO digests (digest_date, run_id, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (digest_date.isoformat(), run_id, payload_json, _now()),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------- helpers

    def _set_error(self, article_id: int, stage: Stage, error: str) -> None:
        self.conn.execute(
            "UPDATE articles SET stage = ?, error = ? WHERE id = ?",
            (stage.value, error[:500], article_id),
        )
        self.conn.commit()

    @staticmethod
    def _to_article(row: sqlite3.Row) -> Article:
        classification = (
            ClassificationOutput.model_validate_json(row["classification_json"])
            if row["classification_json"]
            else None
        )
        summary = (
            SummaryOutput.model_validate_json(row["summary_json"]) if row["summary_json"] else None
        )
        collected = _parse_dt(row["collected_at"])
        assert collected is not None
        return Article(
            id=row["id"],
            url=row["url"],
            url_hash=row["url_hash"],
            title=row["title"],
            source=row["source"],
            language=row["language"],
            priority=row["priority"],
            published_at=_parse_dt(row["published_at"]),
            collected_at=collected,
            teaser=row["teaser"],
            feed_content=row["feed_content"],
            stage=Stage(row["stage"]),
            classification=classification,
            content=row["content"],
            content_origin=row["content_origin"],
            duplicate_of=row["duplicate_of"],
            summary=summary,
            digest_date=date.fromisoformat(row["digest_date"]) if row["digest_date"] else None,
        )
