"""Smoke-tests for the Python deploy orchestration.

The full ``lumilake deploy up`` flow touches Docker, FlowMesh, and the
network; those paths are exercised in integration. These tests lock down the
pure-Python bits: ``.env`` helpers, setup layout, and FlowMesh teardown.
"""

import importlib
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from flowmesh_cli_stack.utils import STACK_PATH_KEYS
from flowmesh_stack.env import load_env

from lumilake import config as sdk_config_mod
from lumilake import envs
from lumilake.config import LumilakeConfig
from lumilake_cli.commands import deploy as deploy_cmd
from lumilake_cli.core import config as cli_config_mod
from lumilake_deploy import containers as containers_mod
from lumilake_deploy import doctor as doctor_mod
from lumilake_deploy import flowmesh as fm
from lumilake_deploy import purge as purge_mod
from lumilake_deploy import setup as setup_mod
from lumilake_deploy import stop as stop_mod
from lumilake_deploy.env import read_env_value
from lumilake_deploy.errors import DeployError


def test_read_env_value_handles_quoted_and_unquoted(tmp_path: Path) -> None:
    path = tmp_path / "env_fixture"
    path.write_text('KEY_Q="value-q"\nKEY_U=value-u\n# COMMENT\n')
    assert read_env_value(path, "KEY_Q") == "value-q"
    assert read_env_value(path, "KEY_U") == "value-u"
    assert read_env_value(path, "MISSING") == ""


def test_read_env_value_drops_an_inline_comment(tmp_path: Path) -> None:
    """The packaged template annotates values on the same line."""
    path = tmp_path / "env_fixture"
    path.write_text('LUMILAKE_STACK_NAME=""     # Name of this deploy stack\n')
    assert read_env_value(path, "LUMILAKE_STACK_NAME") == ""
    path.write_text('LUMILAKE_STACK_NAME="demo"   # Name of this deploy stack\n')
    assert read_env_value(path, "LUMILAKE_STACK_NAME") == "demo"


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


def test_doctor_missing_env_file_message_is_actionable(tmp_path: Path) -> None:
    report = doctor_mod.run_env_checks(tmp_path / ".env")
    assert any("lumilake deploy init" in msg for msg in report.errors)


def test_doctor_rejects_a_stack_name_that_is_not_a_compose_project(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('LUMILAKE_STACK_NAME="Bad Name"\n')

    report = doctor_mod.run_env_checks(env_file)

    assert any("LUMILAKE_STACK_NAME" in msg for msg in report.errors)


def test_doctor_rejects_the_reserved_stack_name(tmp_path: Path) -> None:
    """``lumilake`` is the FlowMesh name an unnamed bundled deployment uses."""
    env_file = tmp_path / ".env"
    env_file.write_text('LUMILAKE_STACK_NAME="lumilake"\n')

    report = doctor_mod.run_env_checks(env_file)

    matched = [msg for msg in report.errors if "LUMILAKE_STACK_NAME" in msg]
    assert matched
    assert "unnamed bundled deployments use" in matched[0]


def test_doctor_accepts_a_valid_stack_name(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('LUMILAKE_STACK_NAME="demo-1"\n')

    report = doctor_mod.run_env_checks(env_file)

    assert not any("LUMILAKE_STACK_NAME" in msg for msg in report.errors)


def test_load_project_env_drops_an_inherited_stack_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment with no name never keeps the caller's stack name."""
    monkeypatch.setitem(os.environ, "LUMILAKE_STACK_NAME", "inherited")
    monkeypatch.delenv("LUMILAKE_IMAGE_TAG", raising=False)
    (tmp_path / ".env").write_text('LUMILAKE_IMAGE_TAG="dev"\n')
    try:
        setup_mod.load_project_env(tmp_path)

        assert "LUMILAKE_STACK_NAME" not in os.environ
        assert envs.LUMILAKE_STACK_NAME == ""
        assert containers_mod.read_stack_name(tmp_path) == ""
    finally:
        os.environ.pop("LUMILAKE_STACK_NAME", None)
        importlib.reload(envs)


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
                'LUMID_DATA_URL="http://127.0.0.1:9102"',
                'LUMID_DATA_TOKEN="lumid-token"',
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


def test_cli_up_never_writes_the_cli_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deploy up`` reports how to point the CLI here; it never writes it."""
    monkeypatch.setattr(deploy_cmd, "_run_setup", lambda *a, **kw: None)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", home.as_posix())
    config_path = home / ".lumilake" / "config.toml"
    monkeypatch.setattr(sdk_config_mod, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(cli_config_mod, "DEFAULT_CONFIG_PATH", config_path)
    saved: list[Path | None] = []

    def _save(_self: LumilakeConfig, path: Path | None = None) -> None:
        saved.append(path)

    monkeypatch.setattr(LumilakeConfig, "save", _save)
    (tmp_path / ".env").write_text('LUMILAKE_SERVER_PORT="9100"\n')

    messages: list[str] = []
    monkeypatch.setattr(deploy_cmd.logging, "info", lambda m: messages.append(m))

    deploy_cmd.up(_fake_ctx(tmp_path))

    assert saved == []
    assert not config_path.exists()
    assert any("--deploy-dir" in m for m in messages)
    assert any("http://127.0.0.1:9100" in m for m in messages)


def test_point_cli_at_deployment_quotes_a_path_with_a_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The logged command stays pasteable when the directory has a space."""
    root = tmp_path / "with space"
    root.mkdir()
    (root / ".env").write_text('LUMILAKE_SERVER_PORT="9100"\n')

    messages: list[str] = []
    monkeypatch.setattr(deploy_cmd.logging, "info", lambda m: messages.append(m))

    deploy_cmd._point_cli_at_deployment(root)

    assert "'" + root.as_posix() + "'" in messages[0]


def test_cli_init_honors_project_dir_argument(tmp_path: Path) -> None:
    """``--project-dir`` (passed via ctx.obj) routes init into the chosen dir."""
    deploy_cmd.init(_fake_ctx(tmp_path), flowmesh=False, force=False)
    assert (tmp_path / ".env").is_file()


def test_run_stop_names_the_stack_in_the_compose_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``down`` tears down this deployment's Compose project only."""
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')
    run_calls: list[list[str]] = []

    def _run(cmd: list[str], **_kwargs: object) -> None:
        run_calls.append(cmd)

    monkeypatch.setattr(stop_mod, "load_project_env", lambda _root: None)
    monkeypatch.setattr(stop_mod.docker_client, "engine_is_up", lambda: False)
    monkeypatch.setattr(stop_mod, "run", _run)

    stop_mod.run_stop(tmp_path)

    assert len(run_calls) == 1
    cmd = run_calls[0]
    assert cmd[:2] == ["docker", "compose"]
    assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "demo"
    assert cmd[-1] == "down"


def test_reset_preserves_flowmesh_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')
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
    assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "demo"
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
    (tmp_path / fm.WORKDIR_NAME).mkdir()

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


class _WorkerCreation:
    """Records every worker-creation call and answers with the created aliases."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self, _client: object, **kwargs: object
    ) -> list[tuple[str, dict[str, str]]]:
        self.calls.append(kwargs)
        kind = kwargs["kind"]
        return [(f"{kind}-0", {"id": "wkr-1", "alias": f"demo_worker_{kind}_0"})]


@pytest.fixture
def flowmesh_env(tmp_path: Path) -> Path:
    """An env file complete enough for ``create_workers`` to read."""
    env_file = tmp_path / ".env.flowmesh"
    env_file.write_text(
        "SERVER_HTTP_PORT=18000\nSERVER_TOKEN=t\n"
        "FLOWMESH_REGISTRY=ghcr.io/mlsys-io\nFLOWMESH_VERSION=v1\n"
    )
    return env_file


@pytest.fixture
def worker_creation(monkeypatch: pytest.MonkeyPatch) -> _WorkerCreation:
    """Stubs the FlowMesh client and the image pull that ``create_workers`` uses."""
    creation = _WorkerCreation()

    class _StubContext:
        def __enter__(self) -> "_StubContext":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(fm, "sdk_create_workers", creation)
    monkeypatch.setattr(fm, "NodeClient", lambda **_kw: _StubContext())
    monkeypatch.setattr(fm, "_pull_worker_image", lambda *_a, **_kw: None)
    return creation


def test_flowmesh_workers_carry_the_stack_name_slug(
    tmp_path: Path,
    flowmesh_env: Path,
    worker_creation: _WorkerCreation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both worker calls name the stack, so two stacks on a host don't collide,
    and each created worker is logged by its alias."""
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')
    monkeypatch.setattr(fm, "_resolve_gpu_device_ids", lambda _raw: (["0", "1"], ""))
    logged: list[str] = []
    monkeypatch.setattr(fm, "_info", logged.append)

    fm.create_workers(flowmesh_env, cpu_count=2, gpu_devices="all")

    assert [call["kind"] for call in worker_creation.calls] == ["cpu", "gpu"]
    assert all(call["slug"] == "demo" for call in worker_creation.calls)
    assert worker_creation.calls[0]["count"] == 2
    assert worker_creation.calls[1]["count"] == 2
    assert worker_creation.calls[1]["targets"] == "0,1"
    assert "Created cpu-0: demo_worker_cpu_0" in logged
    assert "Created gpu-0: demo_worker_gpu_0" in logged


def test_stack_up_builds_workdir_in_deploy_dir_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workdir, its mounts and the resolved path keys follow the deployment."""
    deploy_dir = tmp_path / "deployment"
    deploy_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    env_keys = {
        "SERVER_HTTP_PORT": "18000",
        "SERVER_GRPC_PORT": "18001",
        "REDIS_CONTROL_PORT": "18002",
        "REDIS_TELEMETRY_PORT": "18003",
        "REDIS_TLS_DIR": "./secrets/tls/redis",
        "SERVER_TLS_DIR": "./secrets/tls/server",
        "SERVER_WORKER_CONFIG": "./configs/worker_config.yaml",
        "FLOWMESH_PLUGIN_DATA_DIR": "./plugin-data",
    }
    env_file = deploy_dir / ".env.flowmesh"
    env_file.write_text("".join(f"{k}={v}\n" for k, v in env_keys.items()))
    for key in STACK_PATH_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("FLOWMESH_PLUGIN_DATA_DIR", raising=False)
    monkeypatch.setattr(load_env, "_loaded", None, raising=False)

    compose_calls: list[list[str]] = []

    class _StubStack:
        def run(
            self,
            args: list[str],
            *,
            env_file: Path,
            env: dict[str, str] | None = None,
        ) -> object:
            compose_calls.append(args)
            return type("_Result", (), {"returncode": 0})()

    monkeypatch.setattr(fm, "_stack", _StubStack())
    monkeypatch.setattr(fm, "check_ports", lambda _ports: [])
    monkeypatch.chdir(elsewhere)

    fm.stack_up(env_file)

    workdir = deploy_dir / fm.WORKDIR_NAME
    assert (workdir / "secrets" / "tls" / "redis").is_dir()
    assert (workdir / "secrets" / "tls" / "server").is_dir()
    assert (workdir / "configs" / "worker_config.yaml").is_file()
    assert (workdir / "plugin-data").is_dir()
    assert not (elsewhere / fm.WORKDIR_NAME).exists()
    for key in (
        "REDIS_TLS_DIR",
        "SERVER_TLS_DIR",
        "SERVER_WORKER_CONFIG",
        "FLOWMESH_PLUGIN_DATA_DIR",
    ):
        assert Path(os.environ[key]).is_relative_to(workdir)
    assert compose_calls == [["--profile", "root", "up", "-d", "--wait"]]


def test_stack_up_creates_the_env_files_worker_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker config composes mounts is the one the env file names."""
    deploy_dir = tmp_path / "deployment"
    deploy_dir.mkdir()
    env_file = deploy_dir / ".env.flowmesh"
    env_file.write_text(
        "SERVER_HTTP_PORT=18000\nSERVER_GRPC_PORT=18001\n"
        "REDIS_CONTROL_PORT=18002\nREDIS_TELEMETRY_PORT=18003\n"
        "SERVER_WORKER_CONFIG=./configs/custom.yaml\n"
    )
    for key in STACK_PATH_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(load_env, "_loaded", None, raising=False)

    class _StubStack:
        def run(
            self,
            args: list[str],
            *,
            env_file: Path,
            env: dict[str, str] | None = None,
        ) -> object:
            return type("_Result", (), {"returncode": 0})()

    monkeypatch.setattr(fm, "_stack", _StubStack())
    monkeypatch.setattr(fm, "check_ports", lambda _ports: [])

    fm.stack_up(env_file)

    assert (deploy_dir / fm.WORKDIR_NAME / "configs" / "custom.yaml").is_file()


def test_stack_down_runs_without_a_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stack started before the upgrade is still torn down from its deployment."""
    deploy_dir = tmp_path / "deployment"
    deploy_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / fm.WORKDIR_NAME).mkdir(parents=True)

    env_file = deploy_dir / ".env.flowmesh"
    env_file.write_text(
        "SERVER_HTTP_PORT=18000\nSERVER_TOKEN=t\nFLOWMESH_VERSION=v1\n"
        "FLOWMESH_REGISTRY=ghcr.io/mlsys-io\n"
    )

    class _StubContext:
        def __enter__(self) -> "_StubContext":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def destroy_all_workers(self, *, ignore_unreachable: bool = False) -> bool:
            return False

    compose_calls: list[list[str]] = []

    class _StubStack:
        def run(
            self,
            args: list[str],
            *,
            env_file: Path,
            env: dict[str, str] | None = None,
        ) -> object:
            compose_calls.append(args)
            return type("_Result", (), {"returncode": 0})()

    monkeypatch.setattr(fm, "NodeClient", lambda **_kw: _StubContext())
    monkeypatch.setattr(fm, "_stack", _StubStack())
    monkeypatch.chdir(elsewhere)

    fm.stack_down(env_file)
    fm.stack_clean(env_file)

    assert compose_calls == [
        ["--profile", "root", "down"],
        ["--profile", "root", "down", "-v"],
    ]
    assert not (deploy_dir / fm.WORKDIR_NAME).exists()


def test_server_container_defaults_to_bare_name(tmp_path: Path) -> None:
    """No stack name yields the bare server container name."""
    (tmp_path / ".env").write_text("")
    assert containers_mod.server_container(
        containers_mod.read_stack_name(tmp_path)
    ) == ("lumilake-server")


def test_server_container_carries_stack_name(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')
    name = containers_mod.read_stack_name(tmp_path)
    assert containers_mod.server_container(name) == "lumilake-server-demo"


def test_state_volumes_default_to_bare_names(tmp_path: Path) -> None:
    """An empty stack name keeps the historical postgres volume name."""
    (tmp_path / ".env").write_text("")
    volumes = stop_mod._state_volumes(tmp_path)
    assert volumes[0] == "lumilake-postgres-data"


def test_state_volumes_carry_stack_name(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')
    volumes = stop_mod._state_volumes(tmp_path)
    assert volumes[0] == "lumilake-postgres-data-demo"


def test_remove_state_volumes_targets_the_stack_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--wipe-archive`` removes the named stack's volume, not another's."""
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')
    removed: list[str] = []
    monkeypatch.setattr(
        stop_mod.docker_client,
        "volume_exists",
        lambda name: name == "lumilake-postgres-data-demo",
    )

    def _remove(name: str) -> bool:
        removed.append(name)
        return True

    monkeypatch.setattr(stop_mod.docker_client, "volume_remove", _remove)
    stop_mod._remove_state_volumes(tmp_path)

    assert "lumilake-postgres-data-demo" in removed
    assert "lumilake-postgres-data" not in removed


def test_container_names_server_carries_stack_name(tmp_path: Path) -> None:
    """The server container name carries the name from ``.env``."""
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')
    names = containers_mod.container_names(tmp_path)
    assert names["server"] == "lumilake-server-demo"


def test_container_names_server_defaults_to_bare(tmp_path: Path) -> None:
    """No stack name in ``.env`` keeps the bare server container name."""
    (tmp_path / ".env").write_text("")
    names = containers_mod.container_names(tmp_path)
    assert names["server"] == "lumilake-server"


def test_compose_project_args_name_the_stack(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')
    assert containers_mod.compose_project_args(tmp_path) == ["-p", "demo"]


def test_compose_project_args_are_empty_without_a_stack_name(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("")
    assert containers_mod.compose_project_args(tmp_path) == []


def test_flowmesh_slug_is_derived_from_the_stack_name(tmp_path: Path) -> None:
    """One setting names the bundled FlowMesh stack as well."""
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')
    (tmp_path / ".env.flowmesh").write_text("FLOWMESH_STACK_SUFFIX=other\n")
    names = containers_mod.container_names(tmp_path)
    assert names["flowmesh"] == "flowmesh_node_demo_server"
    assert containers_mod.flowmesh_state_volumes(tmp_path)[0] == (
        "flowmesh_node_demo_postgres_data"
    )


def test_flowmesh_slug_keeps_env_flowmesh_suffix_without_a_stack_name(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("")
    (tmp_path / ".env.flowmesh").write_text("FLOWMESH_STACK_SUFFIX=other\n")
    assert containers_mod.container_names(tmp_path)["flowmesh"] == (
        "flowmesh_node_other_server"
    )


def test_load_project_env_reads_the_stack_name(tmp_path: Path) -> None:
    keys = ("LUMILAKE_STACK_NAME",)
    old_values = {key: os.environ.get(key) for key in keys}
    try:
        (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="demo"\n')

        setup_mod.load_project_env(tmp_path)

        assert envs.LUMILAKE_STACK_NAME == "demo"
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(envs)


def test_stack_name_accepts_only_compose_project_names() -> None:
    assert envs.valid_stack_name("stack-1_a")
    assert not envs.valid_stack_name("Demo")
    assert not envs.valid_stack_name("Bad Name")
    assert not envs.valid_stack_name("-leading")


def test_stack_name_matches_the_flowmesh_suffix_the_name_produces() -> None:
    """A name FlowMesh would rewrite would share another stack's FlowMesh."""
    assert envs.valid_stack_name("x-9")
    assert not envs.valid_stack_name("x-")
    assert not envs.valid_stack_name("x_")
    assert not envs.valid_stack_name("x" * 49)
    assert envs.valid_stack_name("x" * 48)
    assert not envs.valid_stack_name("lumilake")


def _patch_valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give ``envs.validate()`` a valid environment apart from the stack name."""
    values: dict[str, object] = {
        "LUMILAKE_SERVER_HOST": "0.0.0.0",
        "RUNTIME_ORCHESTRATOR_URL": "http://127.0.0.1:18000",
        "LUMILAKE_REQUIRE_IDENTITY_PROVIDER": False,
        "LUMILAKE_SERVER_PORT": 9000,
        "LUMILAKE_API_DEFAULT_URL": "https://lum.id/llm/v1/chat/completions",
        "LUMILAKE_JOB_MANAGER_TYPE": "priority",
        "LUMILAKE_COST_DB_SEC_PER_QUERY": 0.05,
        "LUMILAKE_COST_CPU_SEC_PER_NODE": 0.1,
        "LUMILAKE_COST_DEFAULT_MODEL_SIZE_B": 20.0,
        "LUMILAKE_COST_INPUT_QUERY_COUNT": 1,
        "LUMILAKE_SCHEDULER_CUSTOM_ARGS": "{}",
        "LUMILAKE_RUNTIME_MANAGER_TYPE": "default",
        "LUMILAKE_STARVATION_LIMIT": 3,
        "LUMILAKE_BATCH_ACCUMULATION_SECONDS": 0.0,
        "LUMILAKE_CPU_WORKER_GROUP_SIZE": 1,
        "LUMILAKE_GPU_WORKER_GROUP_SIZE": 0,
        "LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS": 60.0,
        "LUMID_DATA_TIMEOUT_SECONDS": 30.0,
        "LUMILAKE_HTTP_TIMEOUT_SECONDS": 60.0,
    }
    for name, value in values.items():
        monkeypatch.setattr(envs, name, value)


def test_validate_rejects_an_invalid_stack_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server refuses to start on a name Compose would refuse."""
    _patch_valid_env(monkeypatch)
    monkeypatch.setattr(envs, "LUMILAKE_STACK_NAME", "Demo")
    with pytest.raises(ValueError, match="LUMILAKE_STACK_NAME"):
        envs.validate()

    monkeypatch.setattr(envs, "LUMILAKE_STACK_NAME", "demo-1")
    envs.validate()


def test_run_setup_rejects_an_invalid_stack_name_before_flowmesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``up`` fails before it starts FlowMesh, so no stack is left half-built."""
    (tmp_path / ".env").write_text('LUMILAKE_STACK_NAME="_x"\n')
    (tmp_path / ".env.flowmesh").write_text("FLOWMESH_STACK_SUFFIX=lumilake\n")

    def _unused(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("FlowMesh must not start for an invalid name")

    monkeypatch.setattr(setup_mod, "check_docker", lambda: None)
    monkeypatch.setattr(setup_mod, "require_commands", lambda _cmds: None)
    monkeypatch.setattr(setup_mod.fm_mod, "stack_pull", _unused)
    monkeypatch.setattr(setup_mod.fm_mod, "stack_up", _unused)
    monkeypatch.setattr(setup_mod.fm_mod, "create_workers", _unused)

    try:
        with pytest.raises(DeployError, match="LUMILAKE_STACK_NAME"):
            setup_mod.run_setup(tmp_path, setup_mod.SetupOptions())
    finally:
        os.environ.pop("LUMILAKE_STACK_NAME", None)
        importlib.reload(envs)


def test_blank_gpu_devices_create_only_cpu_workers(
    flowmesh_env: Path,
    worker_creation: _WorkerCreation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blank ``LUMILAKE_GPU_DEVICES`` creates only CPU workers and never probes
    the host, where ``main`` claims a worker on every GPU."""
    probed: list[list[str]] = []
    warnings: list[str] = []

    def _smi(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        probed.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="0\n1\n", stderr="")

    monkeypatch.setattr(fm.subprocess, "run", _smi)
    monkeypatch.setattr(fm, "_warn", warnings.append)

    fm.create_workers(flowmesh_env, cpu_count=1, gpu_devices="")

    assert [call["kind"] for call in worker_creation.calls] == ["cpu"]
    assert probed == []
    assert len(warnings) == 1 and "is blank" in warnings[0]
