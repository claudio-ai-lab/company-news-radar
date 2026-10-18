"""Offline evaluation of the relevance gate on a small labelled set.

The dataset is JSON Lines; each line is one headline with the expected decision::

    {"id": "s01", "title": "...", "teaser": "...", "relevant": true, "topic": "input_costs"}

The evaluator runs the real classification stage (same prompt, same schema, same gate)
without caches, then reports precision/recall/F1 at the configured threshold, a threshold
sweep computed from the raw scores, topic accuracy and the misclassified items.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from newsradar.config import AppConfig
from newsradar.llm.base import StructuredLLM
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


def format_report(
    config: AppConfig,
    model_id: str,
    dataset: str,
    items: list[EvalItem],
    predictions: list[ClassificationOutput],
    threshold: int,
) -> str:
    main = score(items, predictions, threshold, config.relevance.min_confidence)
    topic_acc = topic_accuracy(items, predictions, threshold)
    positives = sum(item.relevant for item in items)
    lines = [
        f"# Relevance evaluation: {config.profile.company}",
        "",
        f"- Dataset: `{dataset}` ({len(items)} items, {positives} labelled relevant)",
        f"- Classifier: `{model_id}`",
        f"- Gate: relevance >= {threshold}",
        "",
        "| Precision | Recall | F1 | Accuracy | Topic accuracy |",
        "|---:|---:|---:|---:|---:|",
        f"| {main.precision:.2f} | {main.recall:.2f} | {main.f1:.2f} | {main.accuracy:.2f} | "
        + (f"{topic_acc:.2f} |" if topic_acc is not None else "n/a |"),
        "",
        "## Threshold sweep",
        "",
        "| Threshold | TP | FP | FN | TN | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in sweep(items, predictions):
        marker = " **(configured)**" if m.threshold == threshold else ""
        lines.append(
            f"| {m.threshold}{marker} | {m.tp} | {m.fp} | {m.fn} | {m.tn} | "
            f"{m.precision:.2f} | {m.recall:.2f} | {m.f1:.2f} |"
        )
    misses = [
        (item, p)
        for item, p in zip(items, predictions, strict=True)
        if item.relevant != accepted(p, threshold, config.relevance.min_confidence)
    ]
    lines += ["", f"## Misclassified at threshold {threshold} ({len(misses)})", ""]
    if not misses:
        lines.append("None.")
    else:
        lines += [
            "| Item | Expected | Score | Topic | Title | Reason |",
            "|---|---|---:|---|---|---|",
        ]
        for item, p in misses:
            expected = "relevant" if item.relevant else "irrelevant"
            title = item.title.replace("|", "/")
            reason = p.reason.replace("|", "/")
            lines.append(
                f"| {item.id} | {expected} | {p.relevance} | {p.topic} | {title} | {reason} |"
            )
    return "\n".join(lines) + "\n"
