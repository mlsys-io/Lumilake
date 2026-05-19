"""Local lumilake-stack lifecycle.

Backed by ``lumilake_deploy``, gated on the ``deploy`` extra. Without
``pip install 'lumilake[sdk,deploy]'`` (or
``uv sync --extra sdk --extra deploy``) every method raises
``DeployError`` with an install hint. Server-API resources work without
the extra.

``AsyncDeploy`` dispatches each call through ``asyncio.to_thread`` so
the event loop stays responsive across Docker / FlowMesh work; the
backend ``DeployError`` is translated at the boundary.
"""

import asyncio
import logging
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from lumilake.errors import DeployError

logger = logging.getLogger(__name__)

# Gate on the optional backend's presence so real ImportError from
# inside lumilake_deploy still bubbles up loudly instead of being
# misreported as "install the deploy extra".
_BACKEND_AVAILABLE = find_spec("lumilake_deploy") is not None

if _BACKEND_AVAILABLE:
    from lumilake_deploy import containers as _containers
    from lumilake_deploy import docker_client
    from lumilake_deploy import purge as purge_mod
    from lumilake_deploy import setup as setup_mod
    from lumilake_deploy import stop as stop_mod
    from lumilake_deploy import update_flowmesh as update_fm_mod
    from lumilake_deploy.errors import DeployError as _BackendDeployError
    from lumilake_deploy.setup import SetupOptions

    SERVICE_NAMES: tuple[str, ...] = _containers.SERVICE_NAMES
else:
    _containers = None  # type: ignore[assignment]
    docker_client = None  # type: ignore[assignment]
    purge_mod = None  # type: ignore[assignment]
    setup_mod = None  # type: ignore[assignment]
    stop_mod = None  # type: ignore[assignment]
    update_fm_mod = None  # type: ignore[assignment]
    _BackendDeployError = Exception  # type: ignore[misc, assignment]
    SetupOptions = None  # type: ignore[assignment, misc]
    SERVICE_NAMES = (
        "server",
        "flowmesh",
        "flowmesh-redis",
        "flowmesh-redis-telemetry",
    )

ENV_FILE_NAME = ".env"


def _require_backend(action: str) -> None:
    if not _BACKEND_AVAILABLE:
        raise DeployError(
            action,
            exit_code=2,
            stderr=(
                "lumilake[deploy] is not installed. Install with "
                "`pip install 'lumilake[sdk,deploy]'` (or "
                "`uv sync --extra sdk --extra deploy`) to enable deploy "
                "lifecycle methods."
            ),
        )


def _container_for(deploy_dir: Path, service: str) -> str:
    _require_backend("restart")
    names = _containers.container_names(deploy_dir)
    container = names.get(service)
    if container is None:
        raise DeployError(
            "restart",
            exit_code=2,
            stderr=(
                f"unknown service {service!r}. " f"Choose from: {', '.join(names)}"
            ),
        )
    return container


def _wrap(action: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a deploy backend call, translating its DeployError to ours."""
    try:
        return fn(*args, **kwargs)
    except _BackendDeployError as exc:
        raise DeployError(action, exit_code=1, stderr=str(exc)) from exc


class _DeployBase:
    """State + small file helpers shared by Deploy and AsyncDeploy.
    Not a public API — instantiate ``Deploy`` or ``AsyncDeploy`` instead.
    """

    def __init__(self, repo_root: Path | str) -> None:
        self._repo_root = Path(repo_root)

    def _do_init(self, force: bool) -> None:
        _require_backend("init")
        from lumilake_deploy.assets import env_example_path

        template = env_example_path()
        target = self._repo_root / ENV_FILE_NAME
        if target.exists() and not force:
            raise DeployError(
                "init",
                exit_code=2,
                stderr=(f"{target} already exists. Pass force=True to overwrite."),
            )
        target.write_text(template.read_text())
        logger.info("init: wrote %s from packaged %s", target, template.name)


class Deploy(_DeployBase):
    """Sync local lumilake-stack lifecycle. Methods block until the
    corresponding ``lumilake_deploy`` function returns. Requires the
    ``deploy`` extra (``pip install 'lumilake[sdk,deploy]'``); otherwise
    every method except ``init`` raises ``DeployError`` with an install
    hint on first call.
    """

    def up(
        self,
        *,
        background: bool = True,
        no_server: bool = False,
        reset: bool = False,
    ) -> None:
        _require_backend("up")
        opts = SetupOptions(
            reset=reset,
            no_server=no_server,
            background=background,
        )
        _wrap("up", setup_mod.run_setup, self._repo_root, opts)
        logger.info("deploy up: ok")

    def down(self, *, wipe_archive: bool = False) -> None:
        _require_backend("down")
        _wrap(
            "down",
            stop_mod.run_stop,
            self._repo_root,
            purge=False,
            wipe_archive=wipe_archive,
        )
        logger.info("deploy down: ok")

    def clean(self) -> None:
        _require_backend("clean")
        _wrap("clean", stop_mod.run_stop, self._repo_root, purge=True)
        logger.info("deploy clean: ok")

    def purge(self, image_tag: str, *, force: bool = False) -> None:
        """Remove one local Lumilake server image tag."""
        _require_backend("purge")
        plan = _wrap(
            "purge",
            purge_mod.build_server_image_purge_plan,
            self._repo_root,
            image_tag,
        )
        result = _wrap(
            "purge",
            purge_mod.run_server_image_purge,
            plan,
            force=force,
        )
        logger.info("deploy purge %s: removed=%s", image_tag, result.removed)

    def restart(self, service: str | None = None) -> None:
        _require_backend("restart")
        if service is not None:
            container = _container_for(self._repo_root, service)
            _wrap("restart", docker_client.container_restart, container)
            logger.info("deploy restart %s: ok", service)
            return
        _wrap("restart", stop_mod.run_stop, self._repo_root, purge=False)
        opts = SetupOptions(
            reset=False,
            no_server=False,
            background=True,
        )
        _wrap("restart", setup_mod.run_setup, self._repo_root, opts)
        logger.info("deploy restart (full): ok")

    def reset(self) -> None:
        _require_backend("reset")
        _wrap("reset", stop_mod.run_stop, self._repo_root, purge=True)
        opts = SetupOptions(
            reset=True,
            no_server=False,
            background=True,
        )
        _wrap("reset", setup_mod.run_setup, self._repo_root, opts)
        logger.info("deploy reset: ok")

    def logs(
        self,
        service: str = "server",
        *,
        tail: int = 200,
        since: Any = None,
        timestamps: bool = False,
    ) -> str:
        """Return the last ``tail`` lines from ``service``'s container.

        ``since`` is a ``datetime`` (or ``None``); the CLI command
        ``lumilake_cli.commands.deploy._parse_since`` converts ``"10m"``
        style strings to a datetime when callers want CLI-style values.
        """
        _require_backend("logs")
        return _wrap(
            "logs",
            docker_client.container_logs_tail,
            _container_for(self._repo_root, service),
            tail=tail,
            since=since,
            timestamps=timestamps,
        )

    def init(self, *, force: bool = False) -> None:
        """Create ``.env`` from the packaged ``lumilake_deploy`` template."""
        self._do_init(force)

    def update_flowmesh(self) -> None:
        _require_backend("update-flowmesh")
        _wrap("update-flowmesh", update_fm_mod.run_update, self._repo_root)
        logger.info("deploy update-flowmesh: ok")


class AsyncDeploy(_DeployBase):
    """Async local lumilake-stack lifecycle. Each method awaits the same
    ``lumilake_deploy`` work the sync class performs, dispatched through
    ``asyncio.to_thread`` so the event loop stays responsive. Same
    ``deploy``-extra requirement as ``Deploy``.
    """

    async def up(
        self,
        *,
        background: bool = True,
        no_server: bool = False,
        reset: bool = False,
    ) -> None:
        await asyncio.to_thread(
            Deploy(self._repo_root).up,
            background=background,
            no_server=no_server,
            reset=reset,
        )

    async def down(self, *, wipe_archive: bool = False) -> None:
        await asyncio.to_thread(Deploy(self._repo_root).down, wipe_archive=wipe_archive)

    async def clean(self) -> None:
        await asyncio.to_thread(Deploy(self._repo_root).clean)

    async def purge(self, image_tag: str, *, force: bool = False) -> None:
        await asyncio.to_thread(
            Deploy(self._repo_root).purge,
            image_tag,
            force=force,
        )

    async def restart(self, service: str | None = None) -> None:
        await asyncio.to_thread(Deploy(self._repo_root).restart, service)

    async def reset(self) -> None:
        await asyncio.to_thread(Deploy(self._repo_root).reset)

    async def logs(
        self,
        service: str = "server",
        *,
        tail: int = 200,
        since: Any = None,
        timestamps: bool = False,
    ) -> str:
        return await asyncio.to_thread(
            Deploy(self._repo_root).logs,
            service,
            tail=tail,
            since=since,
            timestamps=timestamps,
        )

    async def init(self, *, force: bool = False) -> None:
        await asyncio.to_thread(Deploy(self._repo_root).init, force=force)

    async def update_flowmesh(self) -> None:
        await asyncio.to_thread(Deploy(self._repo_root).update_flowmesh)
