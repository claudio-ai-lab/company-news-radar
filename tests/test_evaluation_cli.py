from pathlib import Path

from click.testing import CliRunner

from newsradar.cli import main
from newsradar.config import load_config
from newsradar.evaluation import format_report, load_dataset, score, sweep, topic_accuracy
from tests.conftest import EXAMPLE_DIR, ROOT, classification

DATASET = ROOT / "evals" / "relevance" / "example-ag.jsonl"


def test_dataset_is_consistent_with_the_example_profile() -> None:
    items = load_dataset(DATASET)
    config = load_config(EXAMPLE_DIR / "config.yaml")
    topic_ids = {t.id for t in config.topics}
    assert len(items) == 40
    assert sum(i.relevant for i in items) == 20
    assert all(i.topic in topic_ids for i in items if i.relevant)
    assert all(i.topic is None for i in items if not i.relevant)


def test_metrics() -> None:
    items = load_dataset(DATASET)[:4]  # r01-r04, all relevant
    preds = [
        classification(relevance=8, topic="supplier_customer_health"),
        classification(relevance=5),
        classification(relevance=9, topic="input_costs"),  # wrong topic
        classification(relevant=False, relevance=2, topic="none"),
    ]
    m = score(items, preds, threshold=6)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 0, 2, 0)
    assert m.precision == 1.0 and m.recall == 0.5
    assert topic_accuracy(items, preds, 6) == 0.5
    assert [s.tp for s in sweep(items, preds)][:6] == [3, 3, 3, 3, 3, 2]


def test_format_report_lists_misses() -> None:
    config = load_config(EXAMPLE_DIR / "config.yaml")
    items = load_dataset(DATASET)[:2]
    preds = [classification(relevance=9), classification(relevant=False, relevance=1, topic="none")]
    report = format_report(config, "scripted/test", "set.jsonl", items, preds, 6)
    assert "| Threshold |" in report and "**(configured)**" in report
    assert "| r02 | relevant | 1 | none |" in report


def test_cli_check_run_eval_costs(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    offline = str(EXAMPLE_DIR / "offline.yaml")
    monkeypatch.chdir(tmp_path)  # relative storage/output paths land in tmp_path
    check = runner.invoke(main, ["check", "-c", offline, "--show-prompts"])
    assert check.exit_code == 0, check.output
    assert "Example AG" in check.output and "===== classify" in check.output

    run = runner.invoke(main, ["--log-level", "WARNING", "run", "-c", offline, "--offline"])
    assert run.exit_code == 0, run.output
    assert "digest_articles: 8" in run.output
    assert Path("out/example-ag-offline/radar.db").is_file()

    costs = runner.invoke(main, ["costs", "-c", offline])
    assert costs.exit_code == 0 and "classification" in costs.output

    report = Path("eval.md")
    ok = runner.invoke(
        main,
        ["eval", "-c", offline, "-d", str(DATASET), "--offline", "--out", str(report)],
    )
    assert ok.exit_code == 0, ok.output
    assert report.read_text(encoding="utf-8").startswith("# Relevance evaluation")

    strict = runner.invoke(
        main, ["eval", "-c", offline, "-d", str(DATASET), "--offline", "--min-f1", "0.99"]
    )
    assert strict.exit_code != 0


def test_cli_reports_missing_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    result = CliRunner().invoke(
        main, ["eval", "-c", str(EXAMPLE_DIR / "config.yaml"), "-d", str(DATASET)]
    )
    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output
