"""Tests for Deploy (sync) + AsyncDeploy.

Mocks the ``lumilake.deploy`` modules the resource calls into,
so unit tests don't need Docker or FlowMesh running.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from lumilake import CONTAINER_NAMES, AsyncDeploy, Deploy, DeployError
from lumilake_deploy.errors import DeployError as _CLIDeployError


@pytest.fixture
def deploy(tmp_path: Path) -> Deploy:
    return Deploy(repo_root=tmp_path)


@pytest.fixture
def async_deploy(tmp_path: Path) -> AsyncDeploy:
    return AsyncDeploy(repo_root=tmp_path)


def test_sync_up_invokes_run_setup(deploy: Deploy, tmp_path: Path) -> None:
    with patch("lumilake.resources.deploy.setup_mod.run_setup") as run_setup:
        deploy.up()
    run_setup.assert_called_once()
    root, opts = run_setup.call_args.args
    assert root == tmp_path
    assert opts.background is True
    assert opts.reset is False


def test_sync_down_calls_run_stop(deploy: Deploy, tmp_path: Path) -> None:
    with patch("lumilake.resources.deploy.stop_mod.run_stop") as run_stop:
        deploy.down()
    run_stop.assert_called_once_with(tmp_path, purge=False, wipe_archive=False)


def test_sync_down_with_wipe_archive(deploy: Deploy, tmp_path: Path) -> None:
    with patch("lumilake.resources.deploy.stop_mod.run_stop") as run_stop:
        deploy.down(wipe_archive=True)
    run_stop.assert_called_once_with(tmp_path, purge=False, wipe_archive=True)


def test_sync_clean_calls_run_stop_purge(deploy: Deploy, tmp_path: Path) -> None:
    with patch("lumilake.resources.deploy.stop_mod.run_stop") as run_stop:
        deploy.clean()
    run_stop.assert_called_once_with(tmp_path, purge=True)


def test_sync_restart_single_service(deploy: Deploy) -> None:
    with patch("lumilake.resources.deploy.docker_client.container_restart") as restart:
        deploy.restart(service="server")
    restart.assert_called_once_with("lumilake-server")


def test_sync_restart_unknown_service_raises(deploy: Deploy) -> None:
    with pytest.raises(DeployError, match="unknown service"):
        deploy.restart(service="not-a-thing")


def test_sync_restart_full_stack(deploy: Deploy, tmp_path: Path) -> None:
    with (
        patch("lumilake.resources.deploy.stop_mod.run_stop") as run_stop,
        patch("lumilake.resources.deploy.setup_mod.run_setup") as run_setup,
    ):
        deploy.restart()
    run_stop.assert_called_once_with(tmp_path, purge=False)
    run_setup.assert_called_once()
    _, opts = run_setup.call_args.args
    assert opts.reset is False
    assert opts.background is True


def test_sync_reset(deploy: Deploy, tmp_path: Path) -> None:
    with (
        patch("lumilake.resources.deploy.stop_mod.run_stop") as run_stop,
        patch("lumilake.resources.deploy.setup_mod.run_setup") as run_setup,
    ):
        deploy.reset()
    run_stop.assert_called_once_with(tmp_path, purge=True)
    _, opts = run_setup.call_args.args
    assert opts.reset is True


def test_sync_logs(deploy: Deploy) -> None:
    with patch(
        "lumilake.resources.deploy.docker_client.container_logs_tail",
        return_value="line1\nline2\n",
    ) as tail:
        out = deploy.logs(service="server", tail=10)
    assert out == "line1\nline2\n"
    tail.assert_called_once_with(
        "lumilake-server", tail=10, since=None, timestamps=False
    )


def test_sync_init_writes_env(deploy: Deploy, tmp_path: Path) -> None:
    template = tmp_path / ".env.example"
    template.write_text("FOO=1\n")
    deploy.init()
    assert (tmp_path / ".env").read_text() == "FOO=1\n"


def test_sync_init_refuses_overwrite_without_force(
    deploy: Deploy, tmp_path: Path
) -> None:
    (tmp_path / ".env.example").write_text("FOO=1\n")
    (tmp_path / ".env").write_text("OLD=1\n")
    with pytest.raises(DeployError, match="already exists"):
        deploy.init()


def test_sync_init_force_overwrites(deploy: Deploy, tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("FRESH=1\n")
    (tmp_path / ".env").write_text("OLD=1\n")
    deploy.init(force=True)
    assert (tmp_path / ".env").read_text() == "FRESH=1\n"


def test_cli_deploy_error_translates_to_sdk_error(deploy: Deploy) -> None:
    """CLI-side DeployError surfaces to the caller as the SDK's DeployError."""
    with patch(
        "lumilake.resources.deploy.setup_mod.run_setup",
        side_effect=_CLIDeployError("port in use"),
    ):
        with pytest.raises(DeployError, match="port in use"):
            deploy.up()


def test_update_flowmesh(deploy: Deploy, tmp_path: Path) -> None:
    with patch("lumilake.resources.deploy.update_fm_mod.run_update") as run_update:
        deploy.update_flowmesh()
    run_update.assert_called_once_with(tmp_path)


def test_container_names_public() -> None:
    assert "server" in CONTAINER_NAMES
    assert CONTAINER_NAMES["server"] == "lumilake-server"


def test_methods_raise_clear_error_when_backend_missing(deploy: Deploy) -> None:
    """Without the ``deploy`` extra installed, lifecycle methods raise
    DeployError with an install hint. ``init`` is exempt — it only
    touches the local filesystem."""
    with patch("lumilake.resources.deploy._BACKEND_AVAILABLE", False):
        with pytest.raises(DeployError, match=r"lumilake\[deploy\]"):
            deploy.up()
        with pytest.raises(DeployError, match=r"lumilake\[deploy\]"):
            deploy.down()
        with pytest.raises(DeployError, match=r"lumilake\[deploy\]"):
            deploy.clean()


def test_init_works_without_backend_extra(deploy: Deploy, tmp_path: Path) -> None:
    """``init`` is filesystem-only and must work without the deploy extra."""
    (tmp_path / ".env.example").write_text("ok\n")
    with patch("lumilake.resources.deploy._BACKEND_AVAILABLE", False):
        deploy.init()
    assert (tmp_path / ".env").read_text() == "ok\n"


@pytest.mark.asyncio
async def test_async_up(async_deploy: AsyncDeploy, tmp_path: Path) -> None:
    with patch("lumilake.resources.deploy.setup_mod.run_setup") as run_setup:
        await async_deploy.up()
    run_setup.assert_called_once()


@pytest.mark.asyncio
async def test_async_down(async_deploy: AsyncDeploy, tmp_path: Path) -> None:
    with patch("lumilake.resources.deploy.stop_mod.run_stop") as run_stop:
        await async_deploy.down(wipe_archive=True)
    run_stop.assert_called_once_with(tmp_path, purge=False, wipe_archive=True)


@pytest.mark.asyncio
async def test_async_logs(async_deploy: AsyncDeploy) -> None:
    with patch(
        "lumilake.resources.deploy.docker_client.container_logs_tail",
        return_value="line1\n",
    ):
        out = await async_deploy.logs(service="server", tail=5)
    assert out == "line1\n"


@pytest.mark.asyncio
async def test_async_failure_translates_error(async_deploy: AsyncDeploy) -> None:
    with patch(
        "lumilake.resources.deploy.setup_mod.run_setup",
        side_effect=_CLIDeployError("port in use"),
    ):
        with pytest.raises(DeployError, match="port in use"):
            await async_deploy.up()
