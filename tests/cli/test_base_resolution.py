"""Base URL resolution for CLI commands."""

from pathlib import Path

import pytest
from lumilake.config import LumilakeConfig
from lumilake_cli.core import http


def _clear_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUMILAKE_BASE_URL", raising=False)
    monkeypatch.delenv("LUMILAKE_API_KEY", raising=False)


def test_resolve_falls_back_to_local_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env_vars(monkeypatch)
    missing = tmp_path / "missing.toml"
    base, source = http.resolve_base_url(missing)
    assert base == http.DEFAULT_LOCAL_BASE_URL
    assert source == "default"


def test_resolve_reads_saved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env_vars(monkeypatch)
    path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://stored:9000").save(path)
    base, source = http.resolve_base_url(path)
    assert base == "http://stored:9000"
    assert source == "config"


def test_resolve_prefers_env_over_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LUMILAKE_BASE_URL", "http://env:9000")
    path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://stored:9000").save(path)
    base, source = http.resolve_base_url(path)
    assert base == "http://env:9000"
    assert source == "env"


def test_resolve_recovers_from_corrupt_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env_vars(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("not = valid toml = at all\n")
    base, source = http.resolve_base_url(path)
    assert base == http.DEFAULT_LOCAL_BASE_URL
    assert source == "default"


def test_resolve_empty_env_falls_through_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty ``LUMILAKE_BASE_URL`` must not mask the saved config value."""
    monkeypatch.setenv("LUMILAKE_BASE_URL", "")
    path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://stored:9000").save(path)
    base, source = http.resolve_base_url(path)
    assert base == "http://stored:9000"
    assert source == "config"


def test_save_config_escapes_toml_string_values(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    LumilakeConfig(base_url='http://host/"quoted"').save(path)
    assert LumilakeConfig.from_file(path).base_url == 'http://host/"quoted"'
