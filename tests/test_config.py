from pathlib import Path

import pytest
import yaml

from newsradar.config import ConfigError, load_config, load_dotenv
from tests.conftest import EXAMPLE_DIR

MINIMAL = {
    "profile": {"company": "Acme", "description": "Makes things."},
    "topics": [{"id": "prices", "label": "Prices", "description": "Input prices."}],
}


def write(tmp_path: Path, data: dict, name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_example_configs_load() -> None:
    online = load_config(EXAMPLE_DIR / "config.yaml")
    offline = load_config(EXAMPLE_DIR / "offline.yaml")
    assert online.profile.company == offline.profile.company == "Example AG"
    assert all(s.url for s in online.sources)
    # extends: lists are replaced, mappings merged
    assert [s.path for s in offline.sources] == [
        "fixtures/example-wire.xml",
        "fixtures/industry-ledger.xml",
    ]
    assert offline.llm.for_stage("summarization").provider == "fake"
    assert offline.pipeline.dedup.window_hours == online.pipeline.dedup.window_hours
    assert offline.resolve("fixtures/example-wire.xml").is_file()


def test_minimal_config_uses_defaults(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.relevance.threshold == 6
    assert cfg.relevance.rubric is None and len(cfg.rubric) >= 2  # language default
    assert cfg.llm.for_stage("digest").model == cfg.llm.default.model
    assert cfg.digest_title == "Acme — News Radar"


def test_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADAR_TO", "ops@example.com")
    data = {
        **MINIMAL,
        "output": {
            "dir": "${RADAR_OUT:-out/acme}",
            "email": {"enabled": True, "sender": "r@example.com", "to": ["${RADAR_TO}"]},
        },
    }
    cfg = load_config(write(tmp_path, data))
    assert cfg.output.dir == "out/acme"
    assert cfg.output.email.to == ["ops@example.com"]


def test_missing_env_variable_is_an_error(tmp_path: Path) -> None:
    data = {**MINIMAL, "output": {"dir": "${RADAR_SURELY_UNSET_VAR}"}}
    with pytest.raises(ConfigError, match="RADAR_SURELY_UNSET_VAR"):
        load_config(write(tmp_path, data))


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"topics": MINIMAL["topics"] * 2}, "duplicate topic"),
        ({"topics": [{"id": "none", "label": "x", "description": "y"}]}, "reserved"),
        ({"sources": [{"name": "s", "url": "https://a.example", "path": "f.xml"}]}, "exactly one"),
        ({"sources": [{"name": "s", "url": "ftp://a.example"}]}, "http"),
        ({"relevance": {"rubric": [{"score": "0-3", "meaning": "a"}] * 2}}, "1-10"),
        ({"llm": {"default": {"provider": "nope"}}}, "unknown provider"),
        ({"surprise": True}, "Extra inputs"),
        ({"output": {"email": {"enabled": True}}}, "sender"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, patch: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, {**MINIMAL, **patch}))


def test_circular_extends(tmp_path: Path) -> None:
    write(tmp_path, {"extends": "b.yaml", **MINIMAL}, "a.yaml")
    write(tmp_path, {"extends": "a.yaml"}, "b.yaml")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_load_dotenv_does_not_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADAR_EXISTING", "keep")
    monkeypatch.delenv("RADAR_NEW", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nRADAR_EXISTING=override\nRADAR_NEW='value'\n", encoding="utf-8")
    load_dotenv(env)
    import os

    assert os.environ["RADAR_EXISTING"] == "keep"
    assert os.environ["RADAR_NEW"] == "value"
    monkeypatch.delenv("RADAR_NEW")
