"""Stage 1: screen headlines and decide what is worth fetching.

Only title, teaser and URL go to the model, which keeps this high-volume stage cheap.
Results are cached by (prompt, model, title, URL). The relevance gate is applied when the
result is *used*, not when it is cached, so changing the threshold takes effect on the
next run without invalidating anything.
"""

from __future__ import annotations

import logging

from newsradar.config import AppConfig
from newsradar.llm.base import LLMError, StructuredLLM
from newsradar.models import ClassificationOutput
from newsradar.prompts import PromptLibrary
from newsradar.storage.base import Repository
from newsradar.text import normalize_url, sha256

logger = logging.getLogger(__name__)

CACHE_NAMESPACE = "classification"


class Classifier:
    def __init__(
        self,
        llm: StructuredLLM,
        prompts: PromptLibrary,
        config: AppConfig,
        repo: Repository | None = None,
    ) -> None:
        self.llm = llm
        self.prompts = prompts
        self.config = config
        self.repo = repo
        self._system = prompts.system("classify")
        self._topic_ids = {t.id for t in config.topics}

    def cache_key(self, title: str, url: str) -> str:
        return sha256(self._system, self.llm.model_id, title, normalize_url(url))

    async def classify(
        self, *, title: str, url: str, source: str, teaser: str = "", language: str = "en"
    ) -> ClassificationOutput:
        key = self.cache_key(title, url)
        if self.repo is not None:
            cached = self.repo.cache_get(CACHE_NAMESPACE, key)
            if cached is not None:
                return ClassificationOutput.model_validate_json(cached)

        article = {
            "title": title,
            "url": url,
            "source": source,
            "teaser": teaser,
            "language": language,
        }
        try:
            result = await self.llm.generate(
                self._system,
                self.prompts.user("classify", article=article),
                ClassificationOutput,
                payload=article,
            )
        except LLMError as exc:
            # Fail closed and do not cache: the article is retried on the next run.
            logger.warning("classification failed for %r: %s", title[:60], exc)
            return ClassificationOutput(
                relevant=False,
                topic="none",
                relevance=1,
                confidence=0.0,
                reason=f"classification failed: {exc}"[:300],
            )

        result = self._normalise(result)
        if self.repo is not None:
            self.repo.cache_put(CACHE_NAMESPACE, key, result.model_dump_json())
        return result

    def _normalise(self, result: ClassificationOutput) -> ClassificationOutput:
        if result.topic not in self._topic_ids and result.topic != "none":
            logger.info("unknown topic %r returned; treating as 'none'", result.topic)
            return result.model_copy(
                update={"topic": "none", "relevant": False, "reason": result.reason[:250]}
            )
        return result

    def accepts(self, result: ClassificationOutput) -> bool:
        """The relevance gate."""
        gate = self.config.relevance
        return (
            result.relevant
            and result.topic != "none"
            and result.relevance >= gate.threshold
            and result.confidence >= gate.min_confidence
        )
