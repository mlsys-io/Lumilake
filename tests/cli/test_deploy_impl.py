"""Smoke-tests for the Python deploy orchestration.

The full ``lumilake deploy up`` flow touches Docker, FlowMesh, and the
network; those paths are exercised in integration. These tests lock down the
pure-Python bits: ``.env`` helpers, setup layout, and FlowMesh teardown.
"""

import importlib
import os
from pathlib import Path

import pytest

from lumilake import envs
from lumilake.cli.commands import deploy as deploy_cmd
from lumilake.deploy import doctor as doctor_mod
from lumilake.deploy import flowmesh as fm
from lumilake.deploy import setup as setup_mod
from lumilake.deploy.env import read_env_value


def test_read_env_value_handles_quoted_and_unquoted() -> None:
    path = Path(__file__).parent / "_env_fixture"
    path.write_text('KEY_Q="value-q"\nKEY_U=value-u\n# COMMENT\n')
    try:
        assert read_env_value(path, "KEY_Q") == "value-q"
        assert read_env_value(path, "KEY_U") == "value-u"
        assert read_env_value(path, "MISSING") == ""
    finally:
        path.unlink()


def test_load_project_env_refreshes_env_registry(tmp_path: Path) -> None:
    keys = ("LUMILAKE_IMAGE_TAG", "DATABASE_URL")
    old_values = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["LUMILAKE_IMAGE_TAG"] = "stale"
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    'LUMILAKE_IMAGE_TAG="fresh"',
                    'DATABASE_URL="postgresql://postgres:pw@db.example.com/postgres"',
                    "",
                ]
            )
        )

        setup_mod.load_project_env(tmp_path)

        assert envs.LUMILAKE_IMAGE_TAG == "fresh"
        assert envs.DATABASE_URL == "postgresql://postgres:pw@db.example.com/postgres"
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(envs)


def test_resolve_infra_layout_uses_flowmesh_env_file(tmp_path: Path) -> None:
    assert setup_mod._resolve_infra_layout(tmp_path).deploy_fm is False

    (tmp_path / ".env.flowmesh").write_text("SERVER_HTTP_PORT=18000\n")

    assert setup_mod._resolve_infra_layout(tmp_path).deploy_fm is True


def test_doctor_rejects_malformed_s3_url(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'LUMILAKE_SERVER_HOST="0.0.0.0"',
                'LUMILAKE_SERVER_PORT="9000"',
                'LUMILAKE_RUNTIME_ORCHESTRATOR_URL="http://127.0.0.1:18000"',
                'S3_ARCHIVE_PREFIX="lumilake-archive/artifacts"',
                'LUMILAKE_IMAGE_TAG="latest"',
                'DATABASE_URL="postgresql://postgres:pw@db.example.com/postgres"',
                'S3_URL="http://s3.example.com:9000"',
                'S3_USER_DATA_PREFIX="lumilake-user-data/users"',
                "",
            ]
        )
    )

    report = doctor_mod.run_env_checks(env_file)

    assert "S3_URL must use the s3:// scheme" in report.errors
    assert "S3_URL must include access key and secret" in report.errors
    assert "S3_URL must include a bucket or bucket/prefix path" in report.errors


def test_cli_init_declined_overwrite_does_not_patch_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env.example").write_text('LUMILAKE_IMAGE_TAG="latest"\n')
    target = tmp_path / ".env"
    original = 'LUMILAKE_IMAGE_TAG="old"\n'
    target.write_text(original)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(deploy_cmd.typer, "confirm", lambda *_a, **_kw: False)

    deploy_cmd.init(flowmesh=False, force=False)

    assert target.read_text() == original


def test_reset_preserves_flowmesh_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env.flowmesh"
    env_file.write_text("SERVER_HTTP_PORT=18000\nSERVER_TOKEN=t\n")

    stack_clean_calls: list[Path] = []
    run_calls: list[list[str]] = []

    def _stack_clean(path: Path) -> None:
        stack_clean_calls.append(path)

    def _run(cmd: list[str], **_kwargs: object) -> None:
        run_calls.append(cmd)

    monkeypatch.setattr(setup_mod.fm_mod, "stack_clean", _stack_clean)
    monkeypatch.setattr(setup_mod, "run", _run)

    setup_mod._reset_stack(tmp_path)

    assert stack_clean_calls == [env_file]
    assert env_file.is_file()
    assert run_calls == [["docker", "compose", "--profile", "server", "down", "-v"]]


def test_stack_down_proceeds_when_flowmesh_server_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``stack_down`` uses ``destroy_all_workers(ignore_unreachable=True)``
    so a failed bring-up can still tear down containers that did start.
    """
    monkeypatch.setattr(fm, "_WORKDIR", tmp_path)

    destroy_kwargs: dict[str, object] = {}

    class _StubClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "_StubClient":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def destroy_all_workers(self, *, ignore_unreachable: bool = False) -> bool:
            destroy_kwargs["ignore_unreachable"] = ignore_unreachable
            return False

    monkeypatch.setattr(fm, "NodeClient", _StubClient)

    compose_calls: list[list[str]] = []

    class _StubStack:
        def run(
            self,
            args: list[str],
            *,
            env_file: Path,
            env: dict[str, str] | None = None,
        ) -> None:
            compose_calls.append(args)

    monkeypatch.setattr(fm, "_stack", _StubStack())

    env_file = tmp_path / ".env.flowmesh"
    env_file.write_text("SERVER_HTTP_PORT=18000\nSERVER_TOKEN=t\n")

    fm.stack_down(env_file)

    assert destroy_kwargs == {"ignore_unreachable": True}
    assert compose_calls == [["down"]]
