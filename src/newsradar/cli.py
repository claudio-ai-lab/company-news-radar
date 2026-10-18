"""Command-line interface: ``newsradar run | check | eval | costs``."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from newsradar import __version__
from newsradar.config import (
    DEFAULT_PROVIDERS,
    STAGES,
    AppConfig,
    ConfigError,
    StageModel,
    load_config,
    load_dotenv,
)
from newsradar.evaluation import evaluate_runs, format_report, load_dataset, mean_f1
from newsradar.llm.costs import BudgetExceededError, CostTracker
from newsradar.llm.factory import build_stage_llms
from newsradar.pipeline import Pipeline, RunOptions
from newsradar.prompts import PromptLibrary
from newsradar.storage import open_repository

CONFIG_OPTION = click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Profile YAML, e.g. examples/example-ag/config.yaml",
)


def _load(config_path: Path) -> AppConfig:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


@click.group()
@click.version_option(__version__, prog_name="newsradar")
@click.option("--log-level", default="INFO", show_default=True, help="DEBUG, INFO, WARNING, ...")
def main(log_level: str) -> None:
    """Company News Radar: turn news feeds into a company-specific daily briefing."""
    logging.basicConfig(
        level=log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()


@main.command()
@CONFIG_OPTION
@click.option("--offline", is_flag=True, help="Deterministic offline LLM; no scraping, no e-mail.")
@click.option("--skip-collect", is_flag=True, help="Process what is already in the database.")
@click.option(
    "--no-scrape", is_flag=True, help="Summarise from feed text instead of fetching pages."
)
@click.option("--limit", type=click.IntRange(min=1), help="Classify at most N new articles.")
@click.option("--redo-digest", is_flag=True, help="Rebuild today's digest, including sent items.")
@click.option("--no-email", is_flag=True, help="Write files but do not send the e-mail.")
def run(
    config_path: Path,
    offline: bool,
    skip_collect: bool,
    no_scrape: bool,
    limit: int | None,
    redo_digest: bool,
    no_email: bool,
) -> None:
    """Run the full pipeline once (schedule this daily)."""
    config = _load(config_path)
    repo = open_repository(config.storage.url)
    options = RunOptions(
        offline=offline,
        skip_collect=skip_collect,
        scrape=not no_scrape,
        limit=limit,
        redo_digest=redo_digest,
        send_email=not no_email,
    )
    try:
        report = asyncio.run(Pipeline(config, repo).run(options))
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        repo.close()
    click.echo(f"run {report.run_id}: {report.status}")
    for key, value in report.stats.items():
        click.echo(f"  {key}: {value}")
    for path in report.outputs:
        click.echo(f"  wrote {path}")


@main.command()
@CONFIG_OPTION
@click.option("--show-prompts", is_flag=True, help="Print the rendered system prompts.")
def check(config_path: Path, show_prompts: bool) -> None:
    """Validate a configuration and show what it will do."""
    config = _load(config_path)
    prompts = PromptLibrary(config)
    enabled = [s for s in config.sources if s.enabled]
    click.echo(f"profile:   {config.profile.company}")
    sources_lang = sorted({s.language for s in enabled}) or ["-"]
    click.echo(
        f"language:  output {config.language} (locale {config.locale}, {config.output.timezone}); "
        f"sources {', '.join(sources_lang)}"
    )
    click.echo(f"topics:    {', '.join(t.id for t in config.topics)}")
    click.echo(f"watchlist: {len(config.profile.watchlist)} entries")
    click.echo(f"sources:   {len(enabled)} enabled of {len(config.sources)}")
    click.echo(f"gate:      relevance >= {config.relevance.threshold}")
    for stage in STAGES:
        model = config.llm.for_stage(stage)
        click.echo(f"llm.{stage:<14} {model.provider}/{model.model}")
    click.echo(f"budget:    ${config.llm.daily_budget_usd:.2f}/day")
    click.echo(f"storage:   {config.storage.url}")
    click.echo(f"output:    {config.output.dir} ({', '.join(config.output.formats)})")
    if show_prompts:
        for name in ("classify", "summarize", "dedup", "digest"):
            click.echo(f"\n===== {name} (system) =====\n{prompts.system(name)}")


def _classification_override(
    config: AppConfig,
    model: str | None,
    temperature: float | None,
    reasoning_effort: str | None,
) -> StageModel:
    """The classification stage with command-line overrides applied."""
    stage = config.llm.for_stage("classification")
    update: dict[str, object] = {}
    if model:
        provider, sep, name = model.partition("/")
        if sep and provider in {**DEFAULT_PROVIDERS, **config.llm.providers}:
            update.update(provider=provider, model=name)
        else:
            update.update(model=model)
        # Settings tuned for the configured model do not carry over to another one.
        update.update(temperature=None, reasoning_effort=None)
    if temperature is not None:
        update["temperature"] = temperature
    if reasoning_effort is not None:
        update["reasoning_effort"] = reasoning_effort
    return stage.model_copy(update=update)


def _settings_label(stage: StageModel) -> str:
    temperature = "provider default" if stage.temperature is None else f"{stage.temperature:g}"
    effort = stage.reasoning_effort or "provider default"
    return f"temperature {temperature}, reasoning effort {effort}"


@main.command("eval")
@CONFIG_OPTION
@click.option(
    "-d",
    "--dataset",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Labelled JSON Lines file.",
)
@click.option("--offline", is_flag=True, help="Evaluate the keyword baseline instead of the LLM.")
@click.option(
    "--model",
    "model_name",
    help="Evaluate another classification model: [provider/]model, e.g. openai/gpt-6.1-sol.",
)
@click.option(
    "--reasoning-effort", help="Reasoning effort for the classification model, e.g. none or low."
)
@click.option(
    "--temperature",
    type=click.FloatRange(0, 2),
    help="Sampling temperature for the classification model (default: provider default).",
)
@click.option(
    "--repeat",
    default=1,
    show_default=True,
    type=click.IntRange(1, 10),
    help="Evaluate N times in a row; models without a fixed temperature vary between runs.",
)
@click.option(
    "--budget",
    type=click.FloatRange(min=0, min_open=True),
    help="Spending cap in USD for this evaluation (default: llm.daily_budget_usd).",
)
@click.option("--threshold", type=click.IntRange(1, 10), help="Override relevance.threshold.")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), help="Write report here.")
@click.option("--min-f1", type=click.FloatRange(0, 1), help="Exit non-zero if mean F1 is lower.")
def evaluate(
    config_path: Path,
    dataset: Path,
    offline: bool,
    model_name: str | None,
    reasoning_effort: str | None,
    temperature: float | None,
    repeat: int,
    budget: float | None,
    threshold: int | None,
    out: Path | None,
    min_f1: float | None,
) -> None:
    """Measure the relevance gate on a labelled set of headlines."""
    config = _load(config_path)
    stage = _classification_override(config, model_name, temperature, reasoning_effort)
    config.llm.classification = stage
    items = load_dataset(dataset)
    costs = CostTracker(config.llm.pricing, budget or config.llm.daily_budget_usd, run_id="eval")
    try:
        llm = build_stage_llms(config, costs, offline=offline)["classification"]
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        runs = asyncio.run(evaluate_runs(config, items, llm, costs, repeat))
    except BudgetExceededError as exc:
        raise click.ClickException(f"{exc}; raise it with --budget") from exc
    gate = threshold or config.relevance.threshold
    report = format_report(
        config,
        llm.model_id,
        dataset.as_posix(),
        items,
        runs,
        gate,
        settings="" if offline else _settings_label(stage),
        priced=offline or stage.model in config.llm.pricing,
    )
    click.echo(report)
    click.echo(f"LLM calls: {costs.run_calls}, cost: ${costs.run_cost_usd:.4f}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        click.echo(f"wrote {out}")
    f1 = mean_f1(items, runs, gate, config.relevance.min_confidence)
    if min_f1 is not None and f1 < min_f1:
        raise click.ClickException(f"F1 {f1:.2f} is below the required {min_f1:.2f}")


@main.command()
@CONFIG_OPTION
@click.option("--days", default=30, show_default=True, type=click.IntRange(min=1))
def costs(config_path: Path, days: int) -> None:
    """Show LLM calls, tokens and cost per day and stage."""
    config = _load(config_path)
    repo = open_repository(config.storage.url)
    try:
        rows = repo.cost_report(datetime.now(UTC) - timedelta(days=days))
    finally:
        repo.close()
    if not rows:
        click.echo("no LLM calls recorded in this period")
        return
    click.echo(
        f"{'day':<11} {'stage':<14} {'model':<28} {'calls':>6} {'tokens in/out':>17} {'USD':>9}"
    )
    total = 0.0
    for row in rows:
        total += row["cost_usd"] or 0.0
        tokens = f"{row['input_tokens']}/{row['output_tokens']}"
        model = f"{row['provider']}/{row['model']}"
        click.echo(
            f"{row['day']:<11} {row['stage']:<14} {model:<28} {row['calls']:>6} "
            f"{tokens:>17} {row['cost_usd'] or 0:>9.4f}"
        )
    click.echo(f"total: ${total:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
