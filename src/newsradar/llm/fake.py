"""Deterministic offline stand-in for an LLM.

It answers every pipeline stage with simple keyword heuristics driven by the same config
(topics, keywords, watchlist). That makes the quickstart and CI work without network or
API keys, and it doubles as the *keyword baseline* in the relevance evaluation: the gap
between this client and a real model is what the LLM is actually buying you.
"""

from __future__ import annotations

import json
from typing import Any

from newsradar.config import AppConfig
from newsradar.llm.base import LLMRequest, LLMResponse
from newsradar.text import contains_term, jaccard, split_sentences, tokens, truncate

NEGATIVE_TERMS = (
    "insolvency", "bankruptcy", "bankrupt", "closure", "closes", "layoffs", "job cuts",
    "strike", "shortage", "delay", "delays", "disruption", "halts", "recall", "warning",
    "tariff", "tariffs", "sanctions", "surge", "soar", "jump", "price increase", "fire",
    "cuts", "lowered", "fell", "slow", "surcharges", "postponing",
)  # fmt: skip
POSITIVE_TERMS = (
    "wins", "expands", "expansion", "record orders", "growth", "lower prices",
    "eases", "relief", "agreement", "lifts", "cut tariffs",
)  # fmt: skip


class OfflineLLM:
    provider = "fake"
    model = "keyword-heuristic"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def complete(self, request: LLMRequest) -> LLMResponse:
        handlers = {
            "classification": self._classify,
            "dedup": self._dedup,
            "summarization": self._summarize,
            "digest": self._digest,
        }
        data = handlers[request.stage](request.payload)
        text = json.dumps(data, ensure_ascii=False)
        # No tokens are consumed; report zero so cost reports stay honest.
        return LLMResponse(text=text, input_tokens=0, output_tokens=0)

    # ------------------------------------------------------------------ stages

    def _classify(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = f"{payload.get('title', '')} {payload.get('teaser', '')}"
        best_topic, best_hits = "none", 0
        for topic in self.config.topics:
            hits = sum(contains_term(text, kw) for kw in topic.keywords)
            if hits > best_hits:
                best_topic, best_hits = topic.id, hits
        watched = [
            w.name
            for w in self.config.profile.watchlist
            if any(contains_term(text, term) for term in w.terms)
        ]
        score = 1
        if best_hits or watched:
            score = min(10, int(3 + 1.5 * best_hits + 0.5) + 3 * bool(watched))
        if watched and best_topic == "none":
            best_topic = self.config.topics[0].id
        relevant = best_topic != "none" and score >= 3
        reason = (
            f"keyword hits: {best_hits}" + (f"; watchlist: {', '.join(watched)}" if watched else "")
            if relevant
            else "no configured keyword or watchlist entity found"
        )
        return {
            "relevant": relevant,
            "topic": best_topic,
            "relevance": score,
            "confidence": 0.9 if best_hits >= 2 or watched else 0.6,
            "reason": reason,
        }

    def _dedup(self, payload: dict[str, Any]) -> dict[str, Any]:
        a, b = payload.get("a", {}), payload.get("b", {})
        similarity = jaccard(tokens(a.get("title", "")), tokens(b.get("title", "")))
        same = similarity >= 0.35
        return {
            "same_story": same,
            "confidence": round(min(0.99, 0.55 + similarity), 2),
            "reason": f"title token overlap {similarity:.2f}",
        }

    def _summarize(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = payload.get("text") or payload.get("title", "")
        sentences = split_sentences(text) or [text]
        signal_text = f"{payload.get('title', '')} {text}"
        max_chars = self.config.pipeline.summary.max_summary_chars
        max_points = self.config.pipeline.summary.max_key_points
        lowered = signal_text.lower()
        negative = sum(contains_term(lowered, t) for t in NEGATIVE_TERMS)
        positive = sum(contains_term(lowered, t) for t in POSITIVE_TERMS)
        impact = "neutral"
        if negative > positive:
            impact = "negative"
        elif positive > negative:
            impact = "positive"
        companies = [
            w.name
            for w in self.config.profile.watchlist
            if any(contains_term(text, term) for term in w.terms)
        ]
        return {
            "title": truncate(payload.get("title", ""), 180),
            "summary": truncate(sentences[0], max_chars),
            "key_points": [truncate(s, 160) for s in sentences[1 : 1 + min(max_points, 2)]],
            "impact": impact,
            "entities": {"companies": companies[:3], "people": [], "places": []},
        }

    def _digest(self, payload: dict[str, Any]) -> dict[str, Any]:
        articles: list[dict[str, Any]] = payload.get("articles", [])
        by_topic: dict[str, list[int]] = {}
        for item in articles:
            by_topic.setdefault(item.get("topic", "none"), []).append(int(item["n"]))
        groups = []
        for topic_id, numbers in by_topic.items():
            topic = self.config.topic(topic_id)
            groups.append({"label": topic.label if topic else topic_id, "article_numbers": numbers})
        top = sorted(articles, key=lambda a: -int(a.get("relevance", 0)))[:3]
        return {
            "executive_summary": [f"{a['title']} ({a['source']})." for a in top],
            "groups": groups,
        }
