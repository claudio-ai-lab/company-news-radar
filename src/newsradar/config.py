"""Configuration: a company profile plus pipeline settings, loaded from YAML.

Everything organisation-specific lives in the config file: who the briefing is for,
the topics, the relevance rubric, the sources, optional prompt overrides and the
recipients. Secrets never do. API keys and SMTP credentials are read from environment
variables whose *names* the config refers to.

Two conveniences:

* ``extends: other.yaml`` deep-merges this file on top of another one (dicts merge,
  lists are replaced), so a private profile can reuse a shared base.
* ``${VAR}`` / ``${VAR:-default}`` inside strings is expanded from the environment.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from newsradar import i18n

STAGES = ("classification", "dedup", "summarization", "digest")
StageName = Literal["classification", "dedup", "summarization", "digest"]


class ConfigError(Exception):
    """Raised when a configuration file cannot be loaded or is invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- profile


class WatchItem(_Strict):
    """An organisation the reader always wants to hear about (supplier, customer, rival)."""

    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    relation: str = ""

    @property
    def terms(self) -> list[str]:
        return [self.name, *self.aliases]


class Profile(_Strict):
    """Who the briefing is for. Rendered into every prompt."""

    company: str = Field(min_length=1)
    description: str = Field(min_length=1)
    audience: str = "management team"
    focus: str = ""
    regions: list[str] = Field(default_factory=list)
    watchlist: list[WatchItem] = Field(default_factory=list)


class Topic(_Strict):
    """A section of the briefing. ``keywords`` are hints for the LLM and the offline baseline."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,40}$")
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    icon: str = Field(default="•", max_length=2)


class RubricLevel(_Strict):
    """One anchor of the 1-10 relevance scale, e.g. ``score: "7-8"``."""

    score: str = Field(pattern=r"^\d{1,2}(-\d{1,2})?$")
    meaning: str = Field(min_length=1)

    @property
    def bounds(self) -> tuple[int, int]:
        low, _, high = self.score.partition("-")
        return int(low), int(high or low)

    @model_validator(mode="after")
    def _check_range(self) -> RubricLevel:
        low, high = self.bounds
        if not 1 <= low <= high <= 10:
            raise ValueError(f"rubric score {self.score!r} must lie within 1-10")
        return self


class RelevanceConfig(_Strict):
    """The gate between "collected" and "worth reading"."""

    threshold: int = Field(default=6, ge=1, le=10)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # None: use the built-in default rubric in the output language.
    rubric: list[RubricLevel] | None = Field(default=None, min_length=2)
    guidance: list[str] = Field(default_factory=list)


class ImpactConfig(_Strict):
    """How the summariser labels the direction of a story's effect on the reader."""

    # Unset values fall back to built-in texts in the output language.
    perspective: str = ""
    negative: str | None = None
    positive: str | None = None
    negative_label: str | None = None
    positive_label: str | None = None


# --------------------------------------------------------------------------- sources


class Source(_Strict):
    """An RSS/Atom feed or a news sitemap, given by URL or by a local file path."""

    name: str = Field(min_length=1)
    type: Literal["rss", "sitemap"] = "rss"
    url: str | None = None
    path: str | None = None
    priority: int = Field(default=2, ge=1, le=3)
    language: str = Field(default="en", pattern=r"^[a-z]{2,3}$")  # language the source writes in
    max_age_hours: int | None = Field(default=48, gt=0)
    max_items: int = Field(default=100, gt=0)
    enabled: bool = True

    @model_validator(mode="after")
    def _one_location(self) -> Source:
        if (self.url is None) == (self.path is None):
            raise ValueError(f"source {self.name!r}: set exactly one of 'url' or 'path'")
        if self.url is not None and not self.url.startswith(("http://", "https://")):
            raise ValueError(f"source {self.name!r}: url must start with http:// or https://")
        return self


# --------------------------------------------------------------------------- LLM


class StageModel(_Strict):
    """Which provider/model a pipeline stage uses."""

    provider: str = "openai"
    model: str = "gpt-6-luna"
    # None leaves the sampling temperature to the provider. Current OpenAI reasoning
    # models accept only their default; older chat models can be pinned to 0.
    temperature: float | None = None
    # Passed as ``reasoning_effort`` to OpenAI-compatible providers when set. Valid values
    # depend on the model (gpt-6-luna: none, low, ...; gpt-6.1-sol: low and up).
    reasoning_effort: str | None = None
    max_output_tokens: int = Field(default=1500, gt=0)


class ProviderSettings(_Strict):
    kind: Literal["openai", "openai_compatible", "gemini", "fake"] = "openai_compatible"
    api_key_env: str | None = None
    base_url: str | None = None


class Price(_Strict):
    """USD per one million tokens."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)


DEFAULT_PROVIDERS: dict[str, ProviderSettings] = {
    "openai": ProviderSettings(kind="openai", api_key_env="OPENAI_API_KEY"),
    "deepseek": ProviderSettings(
        kind="openai_compatible",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
    ),
    "gemini": ProviderSettings(kind="gemini", api_key_env="GOOGLE_API_KEY"),
    "ollama": ProviderSettings(kind="openai_compatible", base_url="http://localhost:11434/v1"),
    "fake": ProviderSettings(kind="fake"),
}


class LLMConfig(_Strict):
    default: StageModel = Field(default_factory=StageModel)
    classification: StageModel | None = None
    dedup: StageModel | None = None
    summarization: StageModel | None = None
    digest: StageModel | None = None
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    pricing: dict[str, Price] = Field(default_factory=dict)
    daily_budget_usd: float = Field(default=1.0, gt=0)
    max_concurrency: int = Field(default=8, ge=1, le=64)
    request_timeout_s: float = Field(default=60.0, gt=0)

    def for_stage(self, stage: str) -> StageModel:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        override: StageModel | None = getattr(self, stage)
        return override or self.default

    def provider(self, name: str) -> ProviderSettings:
        merged = {**DEFAULT_PROVIDERS, **self.providers}
        if name not in merged:
            raise ConfigError(f"unknown LLM provider {name!r}; define it under llm.providers")
        return merged[name]

    @model_validator(mode="after")
    def _providers_exist(self) -> LLMConfig:
        known = {**DEFAULT_PROVIDERS, **self.providers}
        for stage in STAGES:
            name = self.for_stage(stage).provider
            if name not in known:
                raise ValueError(f"llm.{stage}: unknown provider {name!r}")
        return self


# --------------------------------------------------------------------------- pipeline


class ScrapeConfig(_Strict):
    enabled: bool = True
    timeout_s: float = Field(default=20.0, gt=0)
    delay_s: float = Field(default=1.0, ge=0)
    respect_robots_txt: bool = True
    user_agent: str = (
        "company-news-radar/0.1 (+https://github.com/claudio-ai-lab/company-news-radar)"
    )
    min_chars: int = Field(default=300, ge=0)
    max_chars: int = Field(default=12000, gt=0)
    max_concurrency: int = Field(default=4, ge=1, le=32)


class DedupConfig(_Strict):
    enabled: bool = True
    window_hours: int = Field(default=48, gt=0)
    title_jaccard: float = Field(default=0.35, ge=0, le=1)
    url_slug_jaccard: float = Field(default=0.5, ge=0, le=1)
    min_shared_entities: int = Field(default=2, ge=1)
    simhash_max_distance: int = Field(default=12, ge=0, le=64)
    use_embeddings: bool = False
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_similarity: float = Field(default=0.7, ge=0, le=1)
    min_confidence: float = Field(default=0.75, ge=0, le=1)
    max_llm_pairs: int = Field(default=200, ge=0)


class SummaryConfig(_Strict):
    max_key_points: int = Field(default=4, ge=0, le=8)
    max_summary_chars: int = Field(default=320, ge=80)


class DigestConfig(_Strict):
    title: str = "{company} — News Radar"
    max_articles: int = Field(default=40, gt=0)
    max_groups: int = Field(default=8, ge=1)
    executive_summary_max: int = Field(default=5, ge=0, le=10)
    other_group_label: str | None = None  # default: "Other news" in the output language


class PipelineConfig(_Strict):
    max_classifications_per_run: int = Field(default=300, gt=0)
    collect_timeout_s: float = Field(default=15.0, gt=0)
    scrape: ScrapeConfig = Field(default_factory=ScrapeConfig)
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    digest: DigestConfig = Field(default_factory=DigestConfig)


# --------------------------------------------------------------------------- outputs


class StorageConfig(_Strict):
    url: str = Field(default="sqlite:///data/newsradar.db", pattern=r"^sqlite:///")


class EmailConfig(_Strict):
    enabled: bool = False
    sender: str | None = None
    to: list[str] = Field(default_factory=list)
    subject: str = "{title} — {date}"
    smtp_host_env: str = "SMTP_HOST"
    smtp_port_env: str = "SMTP_PORT"
    smtp_user_env: str = "SMTP_USER"
    smtp_password_env: str = "SMTP_PASSWORD"
    starttls: bool = True

    @model_validator(mode="after")
    def _recipients(self) -> EmailConfig:
        if self.enabled and (not self.to or not self.sender):
            raise ValueError("output.email: 'sender' and at least one 'to' address are required")
        return self


OutputFormat = Literal["html", "markdown", "json"]
ALL_FORMATS: tuple[OutputFormat, ...] = ("html", "markdown", "json")


class OutputConfig(_Strict):
    language: str = Field(default="en", pattern=r"^[a-z]{2,3}$")  # briefing language
    locale: str | None = None  # date/number format, e.g. en_GB, de_CH; default: language
    timezone: str = "UTC"  # IANA name used for times in the digest, e.g. Europe/Berlin
    dir: str = "out"
    formats: list[OutputFormat] = Field(default_factory=lambda: list(ALL_FORMATS))
    email: EmailConfig = Field(default_factory=EmailConfig)

    @field_validator("locale")
    @classmethod
    def _known_locale(cls, value: str | None) -> str | None:
        return i18n.validate_locale(value) if value else value

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        return i18n.validate_timezone(value)


class PromptsConfig(_Strict):
    """Optional directory with private prompt templates that replace the built-in ones."""

    dir: str | None = None


class AppConfig(_Strict):
    profile: Profile
    topics: list[Topic] = Field(min_length=1)
    relevance: RelevanceConfig = Field(default_factory=RelevanceConfig)
    impact: ImpactConfig = Field(default_factory=ImpactConfig)
    sources: list[Source] = Field(default_factory=list)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @model_validator(mode="after")
    def _unique_names(self) -> AppConfig:
        for label, names in (
            ("topic id", [t.id for t in self.topics]),
            ("source name", [s.name for s in self.sources]),
        ):
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise ValueError(f"duplicate {label}(s): {', '.join(dupes)}")
        if "none" in {t.id for t in self.topics}:
            raise ValueError("topic id 'none' is reserved for rejected articles")
        return self

    @property
    def base_dir(self) -> Path:
        """Directory of the config file; relative source/prompt paths resolve against it."""
        return self._base_dir

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self._base_dir / candidate

    def topic(self, topic_id: str) -> Topic | None:
        return next((t for t in self.topics if t.id == topic_id), None)

    @property
    def digest_title(self) -> str:
        return self.pipeline.digest.title.format(company=self.profile.company)

    # Language-dependent settings, resolved against the output language ------------

    @property
    def language(self) -> str:
        return self.output.language

    @property
    def locale(self) -> str:
        return self.output.locale or self.output.language

    @property
    def strings(self) -> dict[str, str]:
        return i18n.strings(self.language)

    @property
    def rubric(self) -> list[RubricLevel]:
        if self.relevance.rubric:
            return self.relevance.rubric
        defaults = i18n.DEFAULT_RUBRIC[i18n.supported(self.language)]
        return [RubricLevel(**level) for level in defaults]

    @property
    def impact_negative(self) -> str:
        default = i18n.DEFAULT_IMPACT[i18n.supported(self.language)]["negative"]
        return self.impact.negative or default

    @property
    def impact_positive(self) -> str:
        default = i18n.DEFAULT_IMPACT[i18n.supported(self.language)]["positive"]
        return self.impact.positive or default

    @property
    def negative_label(self) -> str:
        return self.impact.negative_label or self.strings["negative_label"]

    @property
    def positive_label(self) -> str:
        return self.impact.positive_label or self.strings["positive_label"]

    @property
    def other_group_label(self) -> str:
        return self.pipeline.digest.other_group_label or self.strings["other_news"]


# --------------------------------------------------------------------------- loading

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is referenced but not set")

        return _ENV_REF.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``: nested dicts merge, everything else is replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    if path in seen:
        raise ConfigError(f"circular 'extends' involving {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = data.pop("extends", None)
    if parent:
        parent_path = (path.parent / str(parent)).resolve()
        data = deep_merge(_read_yaml(parent_path, seen | {path}), data)
    return data


def load_config(path: str | Path) -> AppConfig:
    """Load, merge, expand and validate a configuration file."""
    config_path = Path(path).resolve()
    raw = _expand_env(_read_yaml(config_path))
    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration in {config_path}:\n{exc}") from exc
    config._base_dir = config_path.parent
    return config


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal ``.env`` loader: KEY=VALUE lines, never overriding existing variables."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
