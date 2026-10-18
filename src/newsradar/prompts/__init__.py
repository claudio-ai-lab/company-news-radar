"""Prompt templates, one set per language.

The built-in templates are generic: they describe the task and the guard rails, while
every company-specific detail (profile, topics, rubric, watchlist) is filled in from the
configuration. The set is chosen by ``output.language`` (``prompts/<lang>/``); languages
without a built-in set use the English templates, which instruct the model to write in
the configured language.

A profile can replace any template with its own file of the same name
(``classify.yaml``, ``summarize.yaml``, ``dedup.yaml``, ``digest.yaml``) placed in
``prompts.dir/<lang>/`` or directly in ``prompts.dir``. That directory belongs next to
your private config, not in git.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from newsradar import i18n
from newsradar.config import AppConfig, ConfigError

PROMPT_NAMES = ("classify", "summarize", "dedup", "digest")
BUILTIN_LANGUAGES = ("en", "de")


class PromptLibrary:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.language = config.language if config.language in BUILTIN_LANGUAGES else "en"
        self._env = Environment(
            undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True, autoescape=False
        )
        override_dir = config.resolve(config.prompts.dir) if config.prompts.dir else None
        self._templates = {name: self._load(name, override_dir) for name in PROMPT_NAMES}

    def _load(self, name: str, override_dir: Path | None) -> dict[str, str]:
        filename = f"{name}.yaml"
        raw: str | None = None
        if override_dir is not None:
            for candidate in (
                override_dir / self.config.language / filename,
                override_dir / filename,
            ):
                if candidate.is_file():
                    raw = candidate.read_text(encoding="utf-8")
                    break
        if raw is None:
            raw = (
                resources.files("newsradar.prompts")
                .joinpath(self.language, filename)
                .read_text(encoding="utf-8")
            )
        data = yaml.safe_load(raw)
        if not isinstance(data, dict) or not {"system", "user"} <= data.keys():
            raise ConfigError(f"prompt template {filename} needs 'system' and 'user' keys")
        return {"system": str(data["system"]), "user": str(data["user"])}

    def language_name(self, code: str) -> str:
        """Name of a language, written in the prompt language (e.g. "German" / "Deutsch")."""
        return i18n.language_name(code, self.language)

    def _context(self, extra: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        return {
            "profile": cfg.profile,
            "topics": cfg.topics,
            "relevance": cfg.relevance,
            "rubric": cfg.rubric,
            "impact": {
                "perspective": cfg.impact.perspective,
                "negative": cfg.impact_negative,
                "positive": cfg.impact_positive,
            },
            "summary": cfg.pipeline.summary,
            "digest": cfg.pipeline.digest,
            "output_language": self.language_name(cfg.language),
            "language_name": self.language_name,
            **extra,
        }

    def system(self, name: str) -> str:
        """System prompt; depends only on the configuration (used in cache keys)."""
        template = self._env.from_string(self._templates[name]["system"])
        return template.render(self._context({})).strip()

    def user(self, name: str, **variables: Any) -> str:
        template = self._env.from_string(self._templates[name]["user"])
        return template.render(self._context(variables)).strip()
