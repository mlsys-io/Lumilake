"""Smoke-tests for the Python deploy orchestration.

The full ``lumilake deploy up`` flow touches Docker, FlowMesh, and the
network; those paths are exercised in integration. These tests lock down the
pure-Python bits: ``.env`` helpers, setup layout, and FlowMesh teardown.
"""

import importlib
import os
from pathlib import Path
from typing import Any

import pytest
from lumilake import envs
from lumilake_cli.commands import deploy as deploy_cmd
from lumilake_deploy import doctor as doctor_mod
from lumilake_deploy import flowmesh as fm
from lumilake_deploy import purge as purge_mod
from lumilake_deploy import setup as setup_mod
from lumilake_deploy.env import read_env_value
from lumilake_deploy.errors import DeployError


def test_read_env_value_handles_quoted_and_unquoted(tmp_path: Path) -> None:
    path = tmp_path / "env_fixture"
    path.write_text('KEY_Q="value-q"\nKEY_U=value-u\n# COMMENT\n')
    assert read_env_value(path, "KEY_Q") == "value-q"
    assert read_env_value(path, "KEY_U") == "value-u"
    assert read_env_value(path, "MISSING") == ""


def test_load_project_env_refreshes_env_registry(tmp_path: Path) -> None:
    keys = ("LUMILAKE_IMAGE_TAG",)
    old_values = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["LUMILAKE_IMAGE_TAG"] = "stale"
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    'LUMILAKE_IMAGE_TAG="fresh"',
                    "",
                ]
            )
        )

        setup_mod.load_project_env(tmp_path)

        assert envs.LUMILAKE_IMAGE_TAG == "fresh"
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(envs)


def test_server_image_ref_uses_default_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_mod.envs, "LUMILAKE_REGISTRY", "ghcr.io/mlsys-io")
    assert (
        setup_mod.server_image_ref("0.2.0") == "ghcr.io/mlsys-io/lumilake_server:0.2.0"
    )


def test_server_image_ref_honors_registry_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(setup_mod.envs, "LUMILAKE_REGISTRY", "my.private.io/foo/")
    assert setup_mod.server_image_ref("dev") == "my.private.io/foo/lumilake_server:dev"


def test_pull_server_image_invokes_image_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(setup_mod.envs, "LUMILAKE_REGISTRY", "ghcr.io/mlsys-io")
    monkeypatch.setattr(setup_mod, "image_pull", lambda tag: calls.append(tag))

    setup_mod.pull_server_image("latest")

    assert calls == ["ghcr.io/mlsys-io/lumilake_server:latest"]


def test_purge_plan_targets_requested_image_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_mod.envs, "LUMILAKE_REGISTRY", "ghcr.io/mlsys-io")
    monkeypatch.setattr(purge_mod.docker_client, "image_exists", lambda _image: True)

    plan = purge_mod.build_server_image_purge_plan(tmp_path, "old")

    assert plan.image_ref == "ghcr.io/mlsys-io/lumilake_server:old"
    assert plan.exists is True


def test_purge_plan_reports_missing_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(setup_mod.envs, "LUMILAKE_REGISTRY", "ghcr.io/mlsys-io")
    monkeypatch.setattr(purge_mod.docker_client, "image_exists", lambda _image: False)

    plan = purge_mod.build_server_image_purge_plan(tmp_path, "missing")

    assert plan.image_ref == "ghcr.io/mlsys-io/lumilake_server:missing"
    assert plan.exists is False


def test_cli_purge_dry_run_does_not_remove_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = purge_mod.PurgePlan(
        image_ref="ghcr.io/mlsys-io/lumilake_server:old",
        exists=True,
    )
    monkeypatch.setattr(
        deploy_cmd.purge_mod,
        "build_server_image_purge_plan",
        lambda *_args, **_kwargs: plan,
    )

    def _run_purge(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not remove images")

    monkeypatch.setattr(deploy_cmd.purge_mod, "run_server_image_purge", _run_purge)

    deploy_cmd.purge(
        _fake_ctx(tmp_path),
        image_tag="old",
        dry_run=True,
        force=False,
    )


def test_cli_purge_removes_requested_image_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = purge_mod.PurgePlan(
        image_ref="ghcr.io/mlsys-io/lumilake_server:old",
        exists=True,
    )
    calls: list[tuple[purge_mod.PurgePlan, bool]] = []

    def _run_purge(
        purge_plan: purge_mod.PurgePlan, *, force: bool = False
    ) -> purge_mod.PurgeResult:
        calls.append((purge_plan, force))
        return purge_mod.PurgeResult(
            image_ref="ghcr.io/mlsys-io/lumilake_server:old",
            removed=True,
        )

    monkeypatch.setattr(
        deploy_cmd.purge_mod,
        "build_server_image_purge_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(deploy_cmd.purge_mod, "run_server_image_purge", _run_purge)

    deploy_cmd.purge(
        _fake_ctx(tmp_path),
        image_tag="old",
        dry_run=False,
        force=True,
    )

    assert calls == [(plan, True)]


def test_cli_purge_treats_disappeared_image_as_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = purge_mod.PurgePlan(
        image_ref="ghcr.io/mlsys-io/lumilake_server:old",
        exists=True,
    )
    monkeypatch.setattr(
        deploy_cmd.purge_mod,
        "build_server_image_purge_plan",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        deploy_cmd.purge_mod,
        "run_server_image_purge",
        lambda *_args, **_kwargs: purge_mod.PurgeResult(
            image_ref="ghcr.io/mlsys-io/lumilake_server:old",
            removed=False,
        ),
    )

    deploy_cmd.purge(
        _fake_ctx(tmp_path),
        image_tag="old",
        dry_run=False,
        force=False,
    )


def test_cli_purge_reports_docker_remove_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = purge_mod.PurgePlan(
        image_ref="ghcr.io/mlsys-io/lumilake_server:old",
        exists=True,
    )
    monkeypatch.setattr(
        deploy_cmd.purge_mod,
        "build_server_image_purge_plan",
        lambda *_args, **_kwargs: plan,
    )

    def _run_purge(*_args: object, **_kwargs: object) -> purge_mod.PurgeResult:
        raise DeployError("docker remove failed")

    monkeypatch.setattr(
        deploy_cmd.purge_mod,
        "run_server_image_purge",
        _run_purge,
    )

    with pytest.raises(deploy_cmd.typer.Exit) as exc_info:
        deploy_cmd.purge(
            _fake_ctx(tmp_path),
            image_tag="old",
            dry_run=False,
            force=False,
        )

    assert exc_info.value.exit_code == 1


def test_resolve_infra_layout_uses_flowmesh_env_file(tmp_path: Path) -> None:
    assert setup_mod._resolve_infra_layout(tmp_path).deploy_fm is False

    (tmp_path / ".env.flowmesh").write_text("SERVER_HTTP_PORT=18000\n")

    assert setup_mod._resolve_infra_layout(tmp_path).deploy_fm is True


def test_doctor_reports_unknown_s3_url_as_warning(tmp_path: Path) -> None:
    """S3_URL is no longer a known key — doctor warns instead of validates it."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'LUMILAKE_SERVER_HOST="0.0.0.0"',
                'LUMILAKE_SERVER_PORT="9000"',
                'LUMILAKE_RUNTIME_ORCHESTRATOR_URL="http://127.0.0.1:18000"',
                'S3_ARCHIVE_PREFIX="lumilake-archive/artifacts"',
                'LUMILAKE_IMAGE_TAG="latest"',
                'LUMID_DATA_URL="http://127.0.0.1:9102"',
                'LUMID_DATA_TOKEN="tok"',
                'S3_URL="http://s3.example.com:9000"',
                "",
            ]
        )
    )

    report = doctor_mod.run_env_checks(env_file)

    assert any(
        "S3_URL" in msg for msg in report.warnings
    ), "S3_URL should be flagged as unknown"
    assert not any("S3_URL must use the s3:// scheme" in msg for msg in report.errors)


def test_doctor_required_env_failures_name_the_env_var(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("")  # empty file: every required var is missing

    report = doctor_mod.run_env_checks(env_file)

    for var in (
        "LUMILAKE_SERVER_HOST",
        "LUMILAKE_SERVER_PORT",
        "LUMILAKE_RUNTIME_ORCHESTRATOR_URL",
        "S3_ARCHIVE_PREFIX",
        "LUMILAKE_IMAGE_TAG",
    ):
        matched = [msg for msg in report.errors if var in msg and "fix:" in msg]
        assert matched, f"{var} should produce an actionable error: {report.errors}"


def test_doctor_database_url_not_required(tmp_path: Path) -> None:
    """DATABASE_URL is no longer required; a complete env without it passes."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'LUMILAKE_SERVER_HOST="0.0.0.0"',
                'LUMILAKE_SERVER_PORT="9000"',
                'LUMILAKE_RUNTIME_ORCHESTRATOR_URL="http://127.0.0.1:18000"',
                'S3_ARCHIVE_PREFIX="lumilake-archive/artifacts"',
                'LUMILAKE_IMAGE_TAG="latest"',
                'LUMID_DATA_URL="http://127.0.0.1:9102"',
                'LUMID_DATA_TOKEN="tok"',
                "",
            ]
        )
    )

    report = doctor_mod.run_env_checks(env_file)

    assert not any(
        "DATABASE_URL" in msg for msg in report.errors
    ), f"DATABASE_URL must not appear in errors: {report.errors}"


def test_doctor_s3_url_not_required(tmp_path: Path) -> None:
    """S3_URL is no longer required; a complete env without it passes."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'LUMILAKE_SERVER_HOST="0.0.0.0"',
                'LUMILAKE_SERVER_PORT="9000"',
                'LUMILAKE_RUNTIME_ORCHESTRATOR_URL="http://127.0.0.1:18000"',
                'S3_ARCHIVE_PREFIX="lumilake-archive/artifacts"',
                'LUMILAKE_IMAGE_TAG="latest"',
                'LUMID_DATA_URL="http://127.0.0.1:9102"',
                'LUMID_DATA_TOKEN="tok"',
                "",
            ]
        )
    )

    report = doctor_mod.run_env_checks(env_file)

    assert not any(
        "S3_URL" in msg for msg in report.errors
    ), f"S3_URL must not appear in errors: {report.errors}"


def test_doctor_missing_env_file_message_is_actionable(tmp_path: Path) -> None:
    report = doctor_mod.run_env_checks(tmp_path / ".env")
    assert any("lumilake deploy init" in msg for msg in report.errors)


def test_doctor_requires_lumid_data_url_for_retrieval(tmp_path: Path) -> None:
    """doctor requires LUMID_DATA_URL — all DataRetrievalOp modes need it."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'LUMILAKE_SERVER_HOST="0.0.0.0"',
                'LUMILAKE_SERVER_PORT="9000"',
                'LUMILAKE_RUNTIME_ORCHESTRATOR_URL="http://127.0.0.1:18000"',
                'S3_ARCHIVE_PREFIX="lumilake-archive/artifacts"',
                'LUMILAKE_IMAGE_TAG="latest"',
                'S3_DATA_PREFIX="lumilake-demo"',
                'LUMID_DATA_TOKEN="tok"',
                "",
            ]
        )
    )

    report = doctor_mod.run_env_checks(env_file)

    assert any(
        "LUMID_DATA_URL" in msg for msg in report.errors
    ), f"LUMID_DATA_URL must be required; errors={report.errors}"
    matched = [msg for msg in report.errors if "LUMID_DATA_URL" in msg]
    assert all("fix:" in msg for msg in matched)


def test_doctor_lumid_data_token_optional(tmp_path: Path) -> None:
    """doctor leaves LUMID_DATA_TOKEN optional.

    It falls back to LUMILAKE_RUNTIME_TOKEN at SDK load time.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                'LUMILAKE_SERVER_HOST="0.0.0.0"',
                'LUMILAKE_SERVER_PORT="9000"',
                'LUMILAKE_RUNTIME_ORCHESTRATOR_URL="http://127.0.0.1:18000"',
                'S3_ARCHIVE_PREFIX="lumilake-archive/artifacts"',
                'LUMILAKE_IMAGE_TAG="latest"',
                'S3_DATA_PREFIX="lumilake-demo"',
                'LUMID_DATA_URL="http://127.0.0.1:9102"',
                "",
            ]
        )
    )

    report = doctor_mod.run_env_checks(env_file)

    assert not any(
        "LUMID_DATA_TOKEN" in msg for msg in report.errors
    ), f"LUMID_DATA_TOKEN must not be required; errors={report.errors}"


def test_doctor_clean_on_valid_comprehensive_env(tmp_path: Path) -> None:
    """A fully-populated .env with all known optional vars produces zero warnings."""
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
                'S3_URL="s3://access:secret@s3.example.com:9000"',
                'S3_DATA_PREFIX="lumilake-demo"',
                'LUMILAKE_QUEUE_QUANTUM_HIGH="400"',
                'LUMILAKE_QUEUE_QUANTUM_MEDIUM="200"',
                'LUMILAKE_QUEUE_QUANTUM_LOW="100"',
                "",
            ]
        )
    )

    report = doctor_mod.run_env_checks(env_file)

    assert not report.errors, f"unexpected errors: {report.errors}"
    assert not report.warnings, f"unexpected warnings: {report.warnings}"


def _fake_ctx(project_dir: Path) -> Any:
    """Stand-in for the ``typer.Context`` that the callback populates."""

    class _Ctx:
        obj = project_dir

    return _Ctx()


def test_cli_init_declined_overwrite_does_not_patch_existing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / ".env"
    original = 'LUMILAKE_IMAGE_TAG="old"\n'
    target.write_text(original)

    monkeypatch.setattr(deploy_cmd.typer, "confirm", lambda *_a, **_kw: False)

    deploy_cmd.init(_fake_ctx(tmp_path), flowmesh=False, force=False)

    assert target.read_text() == original


def test_cli_init_uses_packaged_template(tmp_path: Path) -> None:
    deploy_cmd.init(_fake_ctx(tmp_path), flowmesh=False, force=True)
    written = (tmp_path / ".env").read_text()
    assert "LUMILAKE_IMAGE_TAG" in written


def test_cli_init_previews_new_env_without_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def _preview(t: Path, content: str) -> None:
        events.append(f"preview:{t.name}")

    def _confirm(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("new env files should not prompt")

    monkeypatch.setattr(deploy_cmd, "_preview_write", _preview)
    monkeypatch.setattr(deploy_cmd.typer, "confirm", _confirm)

    deploy_cmd.init(_fake_ctx(tmp_path), flowmesh=False, force=False)

    assert events == ["preview:.env"]
    assert (tmp_path / ".env").is_file()


def test_cli_init_previews_before_overwrite_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``init`` shows the preview before asking to overwrite an existing file."""
    target = tmp_path / ".env"
    target.write_text('LUMILAKE_IMAGE_TAG="existing"\n')

    events: list[str] = []

    def _preview(t: Path, content: str) -> None:
        events.append(f"preview:{t.name}")

    def _confirm(prompt: str, default: bool = False) -> bool:
        events.append(f"confirm:{prompt}")
        return True

    monkeypatch.setattr(deploy_cmd, "_preview_write", _preview)
    monkeypatch.setattr(deploy_cmd.typer, "confirm", _confirm)

    deploy_cmd.init(_fake_ctx(tmp_path), flowmesh=False, force=False)

    preview_idx = next(i for i, e in enumerate(events) if e.startswith("preview:"))
    confirm_idx = next(i for i, e in enumerate(events) if e.startswith("confirm:"))
    assert preview_idx < confirm_idx


def test_cli_init_existing_file_shows_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / ".env"
    target.write_text('LUMILAKE_IMAGE_TAG="local-only"\n')

    captured: list[str] = []

    def _info(message: str) -> None:
        captured.append(message)

    monkeypatch.setattr(deploy_cmd.logging, "info", _info)
    monkeypatch.setattr(deploy_cmd.typer, "confirm", lambda *_a, **_kw: False)

    deploy_cmd.init(_fake_ctx(tmp_path), flowmesh=False, force=False)

    assert any("Diff vs existing" in line for line in captured)


def test_cli_up_writes_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup_calls: list[Path] = []

    def _run_setup(
        root: Path,
        *,
        background: bool = True,
        reset: bool = False,
        no_server: bool = False,
    ) -> None:
        setup_calls.append(root)

    monkeypatch.setattr(deploy_cmd, "_run_setup", _run_setup)
    monkeypatch.setattr(deploy_cmd.setup_mod, "load_project_env", lambda _r: None)
    monkeypatch.setattr(deploy_cmd.envs, "LUMILAKE_SERVER_PORT", 12345)

    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(deploy_cmd, "DEFAULT_CONFIG_PATH", config_path)

    deploy_cmd.up(_fake_ctx(tmp_path))

    assert setup_calls == [tmp_path]
    assert config_path.is_file()
    assert 'base_url = "http://127.0.0.1:12345"' in config_path.read_text()


def test_cli_up_warns_on_config_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(deploy_cmd, "_run_setup", lambda *a, **kw: None)
    monkeypatch.setattr(deploy_cmd.setup_mod, "load_project_env", lambda _r: None)
    monkeypatch.setattr(deploy_cmd.envs, "LUMILAKE_SERVER_PORT", 9001)

    config_path = tmp_path / "config.toml"
    config_path.write_text('base_url = "http://127.0.0.1:9000"\n')
    monkeypatch.setattr(deploy_cmd, "DEFAULT_CONFIG_PATH", config_path)

    messages: list[str] = []
    monkeypatch.setattr(deploy_cmd.logging, "info", lambda m: messages.append(m))

    deploy_cmd.up(_fake_ctx(tmp_path))

    assert any("http://127.0.0.1:9000 -> http://127.0.0.1:9001" in m for m in messages)


def test_cli_up_leaves_matching_config_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(deploy_cmd, "_run_setup", lambda *a, **kw: None)
    monkeypatch.setattr(deploy_cmd.setup_mod, "load_project_env", lambda _r: None)
    monkeypatch.setattr(deploy_cmd.envs, "LUMILAKE_SERVER_PORT", 9000)

    config_path = tmp_path / "config.toml"
    original = 'base_url = "http://127.0.0.1:9000"\n'
    config_path.write_text(original)
    monkeypatch.setattr(deploy_cmd, "DEFAULT_CONFIG_PATH", config_path)

    messages: list[str] = []
    monkeypatch.setattr(deploy_cmd.logging, "info", lambda m: messages.append(m))

    deploy_cmd.up(_fake_ctx(tmp_path))

    assert config_path.read_text() == original
    assert messages == ["CLI config already points at http://127.0.0.1:9000."]


def test_cli_init_honors_project_dir_argument(tmp_path: Path) -> None:
    """``--project-dir`` (passed via ctx.obj) routes init into the chosen dir."""
    deploy_cmd.init(_fake_ctx(tmp_path), flowmesh=False, force=False)
    assert (tmp_path / ".env").is_file()


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
    assert len(run_calls) == 1
    cmd = run_calls[0]
    assert cmd[:2] == ["docker", "compose"]
    assert cmd[-4:] == ["--profile", "server", "down", "-v"]
    assert "-f" in cmd and "--project-directory" in cmd


def test_cli_reset_aborts_when_confirmation_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop_calls: list[Path] = []
    setup_calls: list[Path] = []

    monkeypatch.setattr(deploy_cmd.typer, "confirm", lambda *_a, **_kw: False)
    monkeypatch.setattr(
        deploy_cmd.stop_mod,
        "run_stop",
        lambda root, **_kw: stop_calls.append(root),
    )
    monkeypatch.setattr(
        deploy_cmd,
        "_run_setup",
        lambda root, **_kw: setup_calls.append(root),
    )

    with pytest.raises(deploy_cmd.typer.Exit) as exc_info:
        deploy_cmd.reset(_fake_ctx(tmp_path), yes=False)

    assert exc_info.value.exit_code == 0
    assert stop_calls == []
    assert setup_calls == []


def test_cli_reset_yes_bypasses_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop_calls: list[tuple[Path, bool]] = []
    setup_calls: list[tuple[Path, bool]] = []

    def _confirm(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("confirm should not be called with --yes")

    def _run_stop(root: Path, *, purge: bool = False, **_kwargs: object) -> None:
        stop_calls.append((root, purge))

    def _run_setup(root: Path, *, reset: bool = False, **_kwargs: object) -> None:
        setup_calls.append((root, reset))

    monkeypatch.setattr(deploy_cmd.typer, "confirm", _confirm)
    monkeypatch.setattr(deploy_cmd.stop_mod, "run_stop", _run_stop)
    monkeypatch.setattr(deploy_cmd, "_run_setup", _run_setup)

    deploy_cmd.reset(_fake_ctx(tmp_path), yes=True)

    assert stop_calls == [(tmp_path, True)]
    assert setup_calls == [(tmp_path, True)]


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
    assert compose_calls == [["--profile", "root", "down"]]
