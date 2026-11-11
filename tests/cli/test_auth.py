"""CLI init / deinit / config + HttpClient auth header."""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from lumilake.config import LumilakeConfig
from lumilake_cli.commands.base import _redact_api_key, app
from lumilake_cli.core import http

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BOX_RE = re.compile(r"[│─╭╮╰╯]")


def _plain_output(output: str) -> str:
    """Rich's colour codes and error box split an option name when it renders.

    The bad parameter is reported inside a width-wrapped panel, so strip the
    colour codes, the box drawing and all whitespace before asserting on it.
    """
    plain = BOX_RE.sub("", ANSI_RE.sub("", output))
    return "".join(plain.split())


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUMILAKE_API_KEY", raising=False)
    monkeypatch.delenv("LUMILAKE_BASE_URL", raising=False)


@pytest.fixture
def captured_request() -> Iterator[MagicMock]:
    response = MagicMock(status_code=200, text="{}", json=MagicMock(return_value={}))
    with patch.object(http.requests, "request", return_value=response) as mock:
        yield mock


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_http_client_attaches_bearer_when_key_set(
    captured_request: MagicMock,
) -> None:
    client = http.HttpClient(base_url="http://lumilake.test", api_key="my-token")
    client.get("/healthz", version_prefix=True)
    assert (
        captured_request.call_args.kwargs["headers"]["Authorization"]
        == "Bearer my-token"
    )


def test_http_client_omits_header_when_key_unset(
    captured_request: MagicMock,
) -> None:
    client = http.HttpClient(base_url="http://lumilake.test")
    client.get("/healthz", version_prefix=True)
    assert "Authorization" not in captured_request.call_args.kwargs["headers"]


def test_client_from_config_reads_env_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LUMILAKE_API_KEY", "env-token")
    client = http.client_from_config(config_path=tmp_path / "absent.toml")
    assert client.api_key == "env-token"


def test_client_from_config_reads_file_key(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://stored:9000", api_key="file-token").save(path)
    client = http.client_from_config(config_path=path)
    assert client.api_key == "file-token"
    assert client.base_url == "http://stored:9000"


def test_client_from_config_omits_key_when_unset(tmp_path: Path) -> None:
    client = http.client_from_config(config_path=tmp_path / "absent.toml")
    assert client.api_key is None


def test_init_writes_config_with_api_key(runner: CliRunner, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    result = runner.invoke(
        app,
        [
            "init",
            "http://lumilake.test",
            "--api-key",
            "lm-init",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    loaded = LumilakeConfig.from_file(config_path)
    assert loaded.base_url == "http://lumilake.test"
    assert loaded.api_key == "lm-init"


def test_init_overwrites_with_force(runner: CliRunner, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://old", api_key="old-key").save(config_path)
    result = runner.invoke(
        app,
        [
            "init",
            "http://new",
            "--api-key",
            "new-key",
            "--config",
            str(config_path),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    loaded = LumilakeConfig.from_file(config_path)
    assert loaded.base_url == "http://new"
    assert loaded.api_key == "new-key"


def test_init_aborts_without_force_on_existing(
    runner: CliRunner, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://old", api_key="old-key").save(config_path)
    # Reply 'n' to the overwrite prompt; init should exit without changing the file.
    result = runner.invoke(
        app,
        ["init", "http://new", "--config", str(config_path)],
        input="n\n",
    )
    assert result.exit_code == 0
    loaded = LumilakeConfig.from_file(config_path)
    assert loaded.api_key == "old-key"


def test_init_deploy_dir_takes_the_url_from_the_deployment_env(
    runner: CliRunner, tmp_path: Path
) -> None:
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()
    (deploy_dir / ".env").write_text('LUMILAKE_SERVER_PORT="9100"\n')
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        app,
        [
            "init",
            "--deploy-dir",
            deploy_dir.as_posix(),
            "--config",
            config_path.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert LumilakeConfig.from_file(config_path).base_url == "http://127.0.0.1:9100"


def test_init_deploy_dir_ignores_an_inline_comment(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``deploy init`` writes the port with a trailing explanation."""
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()
    (deploy_dir / ".env").write_text(
        'LUMILAKE_SERVER_PORT="9100"          # Bind port for the server\n'
    )
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        app,
        [
            "init",
            "--deploy-dir",
            deploy_dir.as_posix(),
            "--config",
            config_path.as_posix(),
        ],
    )

    assert result.exit_code == 0, result.output
    assert LumilakeConfig.from_file(config_path).base_url == "http://127.0.0.1:9100"


def test_init_deploy_dir_keeps_a_saved_api_key(
    runner: CliRunner, tmp_path: Path
) -> None:
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()
    (deploy_dir / ".env").write_text('LUMILAKE_SERVER_PORT="9100"\n')
    config_path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://old", api_key="saved-key").save(config_path)

    result = runner.invoke(
        app,
        [
            "init",
            "--deploy-dir",
            deploy_dir.as_posix(),
            "--config",
            config_path.as_posix(),
        ],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    loaded = LumilakeConfig.from_file(config_path)
    assert loaded.base_url == "http://127.0.0.1:9100"
    assert loaded.api_key == "saved-key"


def test_init_deploy_dir_honors_an_explicit_api_key(
    runner: CliRunner, tmp_path: Path
) -> None:
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()
    (deploy_dir / ".env").write_text('LUMILAKE_SERVER_PORT="9100"\n')
    config_path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://old", api_key="saved-key").save(config_path)

    result = runner.invoke(
        app,
        [
            "init",
            "--deploy-dir",
            deploy_dir.as_posix(),
            "--api-key",
            "new-key",
            "--config",
            config_path.as_posix(),
            "--force",
        ],
    )

    assert result.exit_code == 0, result.output
    assert LumilakeConfig.from_file(config_path).api_key == "new-key"


def test_init_deploy_dir_requires_an_env_file(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A directory that is not a deployment is rejected instead of silently
    saving the default port."""
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        app,
        [
            "init",
            "--deploy-dir",
            deploy_dir.as_posix(),
            "--config",
            config_path.as_posix(),
        ],
    )

    assert result.exit_code == 2, result.output
    assert "--deploy-dir" in _plain_output(result.output)
    assert not config_path.exists()


def test_init_rejects_a_url_with_deploy_dir(runner: CliRunner, tmp_path: Path) -> None:
    """The URL argument and --deploy-dir name the same thing, so passing both is
    a usage error rather than a silently ignored argument."""
    deploy_dir = tmp_path / "deploy"
    deploy_dir.mkdir()
    (deploy_dir / ".env").write_text('LUMILAKE_SERVER_PORT="9100"\n')
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        app,
        [
            "init",
            "http://elsewhere:9000",
            "--deploy-dir",
            deploy_dir.as_posix(),
            "--config",
            config_path.as_posix(),
        ],
    )

    assert result.exit_code == 2, result.output
    plain = _plain_output(result.output)
    assert "--deploy-dir" in plain
    assert "Passeither" in plain
    assert "notboth." in plain
    assert not config_path.exists()


def test_init_without_a_url_saves_the_local_default(
    runner: CliRunner, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    result = runner.invoke(app, ["init", "--config", config_path.as_posix()])
    assert result.exit_code == 0, result.output
    assert LumilakeConfig.from_file(config_path).base_url == "http://127.0.0.1:9000"


def test_deinit_deletes_config_file(runner: CliRunner, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://x", api_key="y").save(config_path)
    result = runner.invoke(app, ["deinit", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert not config_path.exists()


def test_deinit_on_missing_warns_but_succeeds(
    runner: CliRunner, tmp_path: Path
) -> None:
    config_path = tmp_path / "absent.toml"
    result = runner.invoke(app, ["deinit", "--config", str(config_path)])
    assert result.exit_code == 0


def test_config_redacts_api_key_by_default(runner: CliRunner, tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    LumilakeConfig(
        base_url="http://lumilake.test", api_key="lm_pat_supersecretkeyxyz"
    ).save(config_path)
    result = runner.invoke(
        app,
        ["config", "--source", "file", "--config", str(config_path)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["api_key"].startswith("lm_p")
    assert payload["api_key"].endswith("yxyz")
    assert "*" in payload["api_key"]


def test_config_shows_plain_with_show_api_key(
    runner: CliRunner, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.toml"
    LumilakeConfig(base_url="http://lumilake.test", api_key="plain-key").save(
        config_path
    )
    result = runner.invoke(
        app,
        [
            "config",
            "--source",
            "file",
            "--show-api-key",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["api_key"] == "plain-key"


def test_config_invalid_source_exits_nonzero(runner: CliRunner) -> None:
    result = runner.invoke(app, ["config", "--source", "garbage"])
    assert result.exit_code == 2


def test_redact_api_key_short_keys_fully_masked() -> None:
    # Keys shorter than 8 chars (4 prefix + 4 suffix) are fully masked.
    assert _redact_api_key("short") == "*****"


def test_redact_api_key_none_passthrough() -> None:
    assert _redact_api_key(None) is None
