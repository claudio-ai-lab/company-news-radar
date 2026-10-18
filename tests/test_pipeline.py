"""End-to-end runs of the offline example profiles (no network, no API key)."""

from pathlib import Path

from newsradar.config import AppConfig, load_config
from newsradar.delivery import build_email
from newsradar.models import Stage
from newsradar.pipeline import Pipeline, RunOptions
from newsradar.storage.sqlite import SQLiteRepository
from tests.conftest import ROOT


async def test_offline_example_end_to_end(config: AppConfig, tmp_path: Path) -> None:
    repo = SQLiteRepository(tmp_path / "radar.db")
    report = await Pipeline(config, repo).run(RunOptions(offline=True))

    assert report.status == "ok"
    stats = report.stats
    assert stats["collected"] == stats["new"] == 14
    assert stats["accepted"] == 9 and stats["rejected"] == 5
    assert stats["duplicates"] == 1
    assert stats["llm_cost_usd"] == 0

    digest = report.digest
    assert digest is not None and digest.article_count == 8
    titles = [item.title for group in digest.groups for item in group.items]
    assert not any("football" in t.lower() or "restaurant" in t.lower() for t in titles)
    nordwerk = next(
        item for g in digest.groups for item in g.items if item.title.startswith("Nordwerk")
    )
    assert [link.source for link in nordwerk.also_covered_by] == ["Industry Ledger"]
    assert {p.suffix for p in report.outputs} == {".html", ".md", ".json"}
    assert all(p.is_file() for p in report.outputs)

    # A second run finds nothing new and writes no digest.
    again = await Pipeline(config, repo).run(RunOptions(offline=True))
    assert again.stats["new"] == 0 and again.stats["digest_articles"] == 0

    # ...unless today's digest is rebuilt on purpose.
    rebuilt = await Pipeline(config, repo).run(
        RunOptions(offline=True, skip_collect=True, redo_digest=True)
    )
    assert rebuilt.stats["digest_articles"] == 8
    repo.close()


async def test_german_offline_example(tmp_path: Path) -> None:
    config = load_config(ROOT / "examples" / "stadtwerke-musterstadt" / "offline.yaml")
    config.output.dir = str(tmp_path / "out")
    repo = SQLiteRepository(":memory:")
    report = await Pipeline(config, repo).run(RunOptions(offline=True))
    assert report.digest is not None and report.digest.language == "de"
    assert report.stats["accepted"] == 4
    markdown = next(p for p in report.outputs if p.suffix == ".md").read_text(encoding="utf-8")
    assert "Das Wichtigste in Kürze" in markdown
    assert "Heimsieg" not in markdown
    repo.close()


async def test_budget_exhaustion_defers_work(config: AppConfig, tmp_path: Path) -> None:
    repo = SQLiteRepository(":memory:")
    config.llm.daily_budget_usd = 0.01
    repo.record_llm_call("earlier", "summarization", "x", "y", 1, 1, 5.0, True, None)
    report = await Pipeline(config, repo).run(RunOptions(offline=True))
    assert report.status == "partial"
    assert report.stats["budget_exhausted"] is True
    assert report.digest is None
    assert len(repo.articles_in_stage(Stage.COLLECTED)) == 14  # retried next run
    repo.close()


def test_email_message(config: AppConfig) -> None:
    from datetime import UTC, date, datetime

    from newsradar.models import Digest

    config.output.email.sender = "radar@example.com"
    config.output.email.to = ["a@example.com", "b@example.com"]
    digest = Digest(
        title="Radar",
        company="Example AG",
        digest_date=date(2026, 10, 16),
        generated_at=datetime(2026, 10, 16, tzinfo=UTC),
    )
    message = build_email(digest, config.output.email)
    assert message["Subject"] == "Radar — 2026-10-16"
    assert message["To"] == "a@example.com, b@example.com"
    assert {part.get_content_type() for part in message.iter_parts()} == {
        "text/plain",
        "text/html",
    }
