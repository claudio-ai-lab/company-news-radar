"""Offline evaluation of the relevance gate on a small labelled set.

The dataset is JSON Lines; each line is one headline with the expected decision::

    {"id": "s01", "title": "...", "teaser": "...", "relevant": true, "topic": "input_costs"}

The evaluator runs the real classification stage (same prompt, same schema, same gate)
without caches, then reports precision/recall/F1 at the configured threshold, a threshold
sweep computed from the raw scores, topic accuracy, the misclassified items, and the time,
tokens and cost of every run. Models that only accept their default temperature answer
slightly differently each time, so the set can be evaluated several times in a row.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from newsradar.config import AppConfig
from newsradar.llm.base import StructuredLLM
from newsradar.llm.costs import CostTracker
from newsradar.models import ClassificationOutput
from newsradar.prompts import PromptLibrary
from newsradar.stages.classify import Classifier


class EvalItem(BaseModel):
    id: str
    title: str
    teaser: str = ""
    url: str = "https://example.com/"
    source: str = "synthetic"
    language: str = "en"
    relevant: bool
    topic: str | None = None
    note: str = ""


@dataclass
class Metrics:
    threshold: int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0

    def __add__(self, other: Metrics) -> Metrics:
        return Metrics(
            self.threshold,
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.tn + other.tn,
        )


@dataclass
class EvalRun:
    """One pass over the dataset: the predictions plus what they cost."""

    predictions: list[ClassificationOutput]
    seconds: float = 0.0
    cost_usd: float = 0.0
    calls: int = 0
    failed_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def load_dataset(path: str | Path) -> list[EvalItem]:
    items = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            items.append(EvalItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item ids")
    return items


async def predict(
    config: AppConfig, items: list[EvalItem], llm: StructuredLLM
) -> list[ClassificationOutput]:
    classifier = Classifier(llm, PromptLibrary(config), config, repo=None)
    return list(
        await asyncio.gather(
            *(
                classifier.classify(
                    title=item.title,
                    url=item.url,
                    source=item.source,
                    teaser=item.teaser,
                    language=item.language,
                )
                for item in items
            )
        )
    )


async def evaluate_runs(
    config: AppConfig,
    items: list[EvalItem],
    llm: StructuredLLM,
    costs: CostTracker,
    repeat: int = 1,
) -> list[EvalRun]:
    """Classify the whole set ``repeat`` times in one event loop, timing and pricing each run."""
    runs = []
    for _ in range(repeat):
        cost, calls, failed = costs.run_cost_usd, costs.run_calls, costs.run_failed_calls
        tokens_in, tokens_out = costs.run_input_tokens, costs.run_output_tokens
        started = time.perf_counter()
        predictions = await predict(config, items, llm)
        runs.append(
            EvalRun(
                predictions,
                seconds=time.perf_counter() - started,
                cost_usd=costs.run_cost_usd - cost,
                calls=costs.run_calls - calls,
                failed_calls=costs.run_failed_calls - failed,
                input_tokens=costs.run_input_tokens - tokens_in,
                output_tokens=costs.run_output_tokens - tokens_out,
            )
        )
    return runs


def accepted(prediction: ClassificationOutput, threshold: int, min_confidence: float = 0.0) -> bool:
    return (
        prediction.relevant
        and prediction.topic != "none"
        and prediction.relevance >= threshold
        and prediction.confidence >= min_confidence
    )


def score(
    items: list[EvalItem],
    predictions: list[ClassificationOutput],
    threshold: int,
    min_confidence: float = 0.0,
) -> Metrics:
    tp = fp = fn = tn = 0
    for item, prediction in zip(items, predictions, strict=True):
        predicted = accepted(prediction, threshold, min_confidence)
        if item.relevant and predicted:
            tp += 1
        elif predicted:
            fp += 1
        elif item.relevant:
            fn += 1
        else:
            tn += 1
    return Metrics(threshold, tp, fp, fn, tn)


def topic_accuracy(
    items: list[EvalItem], predictions: list[ClassificationOutput], threshold: int
) -> float | None:
    """Share of correctly accepted items that also got the expected topic."""
    pairs = [
        (item, p)
        for item, p in zip(items, predictions, strict=True)
        if item.relevant and item.topic and accepted(p, threshold)
    ]
    if not pairs:
        return None
    return sum(item.topic == p.topic for item, p in pairs) / len(pairs)


def sweep(items: list[EvalItem], predictions: list[ClassificationOutput]) -> list[Metrics]:
    return [score(items, predictions, t) for t in range(1, 11)]


def pooled_sweep(items: list[EvalItem], runs: list[EvalRun]) -> list[Metrics]:
    """Threshold sweep with the confusion counts of all runs added up."""
    per_run = [sweep(items, run.predictions) for run in runs]
    return [sum(column[1:], column[0]) for column in zip(*per_run, strict=True)]


def mean_f1(
    items: list[EvalItem], runs: list[EvalRun], threshold: int, min_confidence: float = 0.0
) -> float:
    return sum(score(items, r.predictions, threshold, min_confidence).f1 for r in runs) / len(runs)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def format_report(
    config: AppConfig,
    model_id: str,
    dataset: str,
    items: list[EvalItem],
    runs: list[EvalRun],
    threshold: int,
    *,
    settings: str = "",
    priced: bool = True,
) -> str:
    """Markdown report. ``priced=False`` marks costs as unknown (no price configured)."""
    min_confidence = config.relevance.min_confidence
    positives = sum(item.relevant for item in items)
    lines = [
        f"# Relevance evaluation: {config.profile.company}",
        "",
        f"- Dataset: `{dataset}` ({len(items)} items, {positives} labelled relevant)",
        f"- Classifier: `{model_id}`",
    ]
    if settings:
        lines.append(f"- Settings: {settings}")
    lines += [
        f"- Gate: relevance >= {threshold}",
        f"- Runs: {len(runs)}, at most {config.llm.max_concurrency} requests in parallel",
        "",
        "| Run | Precision | Recall | F1 | Accuracy | Topic accuracy "
        "| LLM calls (failed) | Cost (USD) | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def cost(value: float) -> str:
        return f"{value:.4f}" if priced else "n/a"

    def opt(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "n/a"

    rows = []
    for number, run in enumerate(runs, start=1):
        m = score(items, run.predictions, threshold, min_confidence)
        topic_acc = topic_accuracy(items, run.predictions, threshold)
        rows.append((m, topic_acc))
        lines.append(
            f"| {number} | {m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} | {m.accuracy:.2f} "
            f"| {opt(topic_acc)} | {run.calls} ({run.failed_calls}) | {cost(run.cost_usd)} "
            f"| {run.seconds:.1f} |"
        )
    if len(runs) > 1:
        topic_values = [t for _, t in rows if t is not None]
        lines.append(
            f"| **Mean** | {_mean([m.precision for m, _ in rows]):.2f} "
            f"| {_mean([m.recall for m, _ in rows]):.2f} | {_mean([m.f1 for m, _ in rows]):.2f} "
            f"| {_mean([m.accuracy for m, _ in rows]):.2f} "
            f"| {opt(_mean(topic_values) if topic_values else None)} "
            f"| {_mean([float(r.calls) for r in runs]):.1f} "
            f"({_mean([float(r.failed_calls) for r in runs]):.1f}) "
            f"| {cost(_mean([r.cost_usd for r in runs]))} "
            f"| {_mean([r.seconds for r in runs]):.1f} |"
        )
    per_item = 1000 / len(items)
    mean_cost = _mean([r.cost_usd for r in runs]) * per_item
    lines += [
        "",
        "Per 1,000 headlines (mean over runs): "
        + (f"${mean_cost:.2f}, " if priced else "cost n/a, ")
        + f"{_mean([float(r.input_tokens) for r in runs]) * per_item:,.0f} input and "
        f"{_mean([float(r.output_tokens) for r in runs]) * per_item:,.0f} output tokens. "
        "Costs apply the configured list prices to every token (no cached-input discount).",
        "",
        "## Threshold sweep" + (f" (all {len(runs)} runs pooled)" if len(runs) > 1 else ""),
        "",
        "| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in pooled_sweep(items, runs):
        marker = " **(configured)**" if m.threshold == threshold else ""
        lines.append(
            f"| {m.threshold}{marker} | {m.tp} | {m.fp} | {m.fn} | {m.tn} | "
            f"{m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} |"
        )

    misses = []
    for index, item in enumerate(items):
        answers = [run.predictions[index] for run in runs]
        wrong = [p for p in answers if item.relevant != accepted(p, threshold, min_confidence)]
        if wrong:
            misses.append((item, answers, wrong))
    lines += [
        "",
        f"## Misclassified at threshold {threshold} ({len(misses)} of {len(items)} items)",
        "",
    ]
    if not misses:
        lines.append("None.")
    else:
        lines += [
            "| Item | Expected | Wrong in | Scores | Topic | Title | Reason |",
            "|---|---|---:|---|---|---|---|",
        ]
        for item, answers, wrong in misses:
            expected = "relevant" if item.relevant else "irrelevant"
            scores = ", ".join(str(p.relevance) for p in answers)
            title = item.title.replace("|", "/")
            reason = wrong[0].reason.replace("|", "/")
            lines.append(
                f"| {item.id} | {expected} | {len(wrong)}/{len(answers)} | {scores} "
                f"| {wrong[0].topic} | {title} | {reason} |"
            )
    return "\n".join(lines) + "\n"
