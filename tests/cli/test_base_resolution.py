"""Base URL resolution for CLI commands (after the login removal)."""

import os
from pathlib import Path

import pytest
from lumilake_cli.core import http
from lumilake_cli.core.config import LumilakeConfig, save_config


def _clear_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUMILAKE_BASE_URL", raising=False)


def test_resolve_falls_back_to_local_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env_var(monkeypatch)
    missing = tmp_path / "missing.toml"
    base, source = http._resolve_base_url(missing)
    assert base == http.DEFAULT_LOCAL_BASE_URL
    assert source == "default"


def test_resolve_reads_saved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env_var(monkeypatch)
    path = tmp_path / "config.toml"
    save_config(LumilakeConfig(base_url="http://stored:9000"), path=path)
    base, source = http._resolve_base_url(path)
    assert base == "http://stored:9000"
    assert source == "config"


def test_resolve_prefers_env_over_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LUMILAKE_BASE_URL", "http://env:9000")
    path = tmp_path / "config.toml"
    save_config(LumilakeConfig(base_url="http://stored:9000"), path=path)
    base, source = http._resolve_base_url(path)
    assert base == "http://env:9000"
    assert source == "env"


def test_resolve_recovers_from_corrupt_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env_var(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("not = valid toml = at all\n")
    base, source = http._resolve_base_url(path)
    assert base == http.DEFAULT_LOCAL_BASE_URL
    assert source == "default"


def test_no_login_command_present() -> None:
    """The CLI should no longer expose login / logout / refresh commands."""
    from lumilake_cli.commands import base as base_commands

    names: set[str] = set()
    for cmd in base_commands.app.registered_commands:
        if cmd.name is not None:
            names.add(cmd.name)
            continue
        callback = cmd.callback
        if callback is not None:
            names.add(callback.__name__)
    assert "login" not in names
    assert "logout" not in names
    os.environ.pop("LUMILAKE_NEVER_LOGIN", None)
