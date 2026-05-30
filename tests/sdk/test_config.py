"""Tests for the SDK config loader/saver and resolver."""

from pathlib import Path

import pytest
from lumilake import LumilakeConfig
from lumilake._base_client import DEFAULT_BASE_URL, resolve_config
from lumilake.errors import ConfigInvalidError, ConfigNotFoundError


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUMILAKE_BASE_URL", raising=False)
    monkeypatch.delenv("LUMILAKE_API_KEY", raising=False)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    cfg = LumilakeConfig(base_url="http://localhost:19000")
    cfg.save(target)
    loaded = LumilakeConfig.from_file(target)
    assert loaded.base_url == cfg.base_url
    assert loaded.api_key is None


def test_save_round_trips_api_key(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    cfg = LumilakeConfig(base_url="http://localhost:19000", api_key="secret")
    cfg.save(target)
    loaded = LumilakeConfig.from_file(target)
    assert loaded.api_key == "secret"


def test_to_mapping_omits_none_api_key() -> None:
    cfg = LumilakeConfig(base_url="http://x")
    assert "api_key" not in cfg.to_mapping()


def test_save_escapes_toml_string(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    cfg = LumilakeConfig(base_url='http://localhost:19000/path/"quoted"')
    cfg.save(target)
    loaded = LumilakeConfig.from_file(target)
    assert loaded.base_url == cfg.base_url


def test_from_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigNotFoundError):
        LumilakeConfig.from_file(tmp_path / "absent.toml")


def test_from_file_invalid_toml_raises(tmp_path: Path) -> None:
    target = tmp_path / "bad.toml"
    target.write_text("not = valid = toml\n")
    with pytest.raises(ConfigInvalidError):
        LumilakeConfig.from_file(target)


def test_from_file_missing_base_url_raises(tmp_path: Path) -> None:
    target = tmp_path / "no_url.toml"
    target.write_text('api_key = "x"\n')
    with pytest.raises(ConfigInvalidError, match="Missing 'base_url'"):
        LumilakeConfig.from_file(target)


def test_from_env_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigInvalidError, match="LUMILAKE_BASE_URL"):
        LumilakeConfig.from_env()


def test_from_env_reads_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMILAKE_BASE_URL", "http://env:9000")
    monkeypatch.setenv("LUMILAKE_API_KEY", "env-key")
    cfg = LumilakeConfig.from_env()
    assert cfg.base_url == "http://env:9000"
    assert cfg.api_key == "env-key"


def test_save_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "config.toml"
    LumilakeConfig(base_url="x").save(target)
    assert target.exists()


def test_resolve_config_precedence_params(tmp_path: Path) -> None:
    """Explicit params win over env and file."""
    target = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://file", api_key="file-key").save(target)
    cfg = resolve_config(base_url="http://arg", api_key="arg-key", config_path=target)
    assert cfg.base_url == "http://arg"
    assert cfg.api_key == "arg-key"


def test_resolve_config_env_over_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://file", api_key="file-key").save(target)
    monkeypatch.setenv("LUMILAKE_BASE_URL", "http://env")
    monkeypatch.setenv("LUMILAKE_API_KEY", "env-key")
    cfg = resolve_config(config_path=target)
    assert cfg.base_url == "http://env"
    assert cfg.api_key == "env-key"


def test_resolve_config_falls_through_to_file(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://file", api_key="file-key").save(target)
    cfg = resolve_config(config_path=target)
    assert cfg.base_url == "http://file"
    assert cfg.api_key == "file-key"


def test_resolve_config_default_when_nothing_set(tmp_path: Path) -> None:
    cfg = resolve_config(config_path=tmp_path / "absent.toml")
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.api_key is None


def test_resolve_config_partial_kwargs_fill_from_file(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://file", api_key="file-key").save(target)
    cfg = resolve_config(api_key="arg-key", config_path=target)
    assert cfg.base_url == "http://file"
    assert cfg.api_key == "arg-key"
