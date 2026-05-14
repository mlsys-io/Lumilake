"""Local lumilake-stack lifecycle.

Optional surface — backed by ``lumilake.deploy``, gated on the ``deploy``
extra. ``pip install 'lumilake[sdk,deploy]'`` (or
``uv sync --extra sdk --extra deploy``) pulls in the deploy machinery
(docker SDK, psycopg, flowmesh-*); without that, the resource
still attaches to the client but every method raises ``DeployError``
with an install hint on first call. Server-API resources work without
the extra.

Sync (``Deploy``) calls into ``lumilake.deploy`` directly. Async
(``AsyncDeploy``) dispatches the same calls through ``asyncio.to_thread``
so the event loop stays responsive across the Docker / FlowMesh work.
The deploy library's ``DeployError`` is translated to the SDK's
``DeployError`` at the boundary.

Always invokes the lumilake deploy machinery; never falls back to raw
``docker compose``.
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from lumilake.errors import DeployError

logger = logging.getLogger(__name__)

# Service name → container the operation actually targets. Public so callers
# can introspect what `restart()` / `logs()` accept.
CONTAINER_NAMES: dict[str, str] = {
    "server": "lumilake-server",
    "flowmesh": "flowmesh_node_server",
    "flowmesh-redis": "flowmesh_node_redis_control",
    "flowmesh-redis-telemetry": "flowmesh_node_redis_telemetry",
}


# Lazy import of the deploy backend. Importing ``lumilake.deploy`` pulls in
# docker / psycopg / flowmesh-*, which is too heavy for callers that
# only want the HTTP resources. The import is gated; failure is captured and
# re-raised at first deploy-method invocation.
try:
    from lumilake_deploy import docker_client
    from lumilake_deploy import setup as setup_mod
    from lumilake_deploy import stop as stop_mod
    from lumilake_deploy import update_flowmesh as update_fm_mod
    from lumilake_deploy.env import ENV_FILE_NAME, ENV_TEMPLATE_NAME
    from lumilake_deploy.errors import DeployError as _BackendDeployError
    from lumilake_deploy.setup import SetupOptions

    _BACKEND_AVAILABLE = True
    _BACKEND_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover — exercised in the missing-extra test
    _BACKEND_AVAILABLE = False
    _BACKEND_IMPORT_ERROR = _exc
    docker_client = None  # type: ignore[assignment]
    setup_mod = None  # type: ignore[assignment]
    stop_mod = None  # type: ignore[assignment]
    update_fm_mod = None  # type: ignore[assignment]
    _BackendDeployError = Exception  # type: ignore[misc, assignment]
    SetupOptions = None  # type: ignore[assignment, misc]
    # Bundled file-system constants for backend-free ``init``. These match
    # lumilake_deploy.env when the deploy extra is installed.
    ENV_FILE_NAME = ".env"
    ENV_TEMPLATE_NAME = ".env.example"


def _require_backend(action: str) -> None:
    if not _BACKEND_AVAILABLE:
        raise DeployError(
            action,
            exit_code=2,
            stderr=(
                "lumilake[deploy] is not installed. Install with "
                "`pip install 'lumilake[sdk,deploy]'` (or "
                "`uv sync --extra sdk --extra deploy`) to enable deploy "
                f"lifecycle methods. Original ImportError: "
                f"{_BACKEND_IMPORT_ERROR}"
            ),
        )


def _container_for(service: str) -> str:
    container = CONTAINER_NAMES.get(service)
    if container is None:
        raise DeployError(
            "restart",
            exit_code=2,
            stderr=(
                f"unknown service {service!r}. "
                f"Choose from: {', '.join(CONTAINER_NAMES)}"
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
        # ``init`` only touches the local filesystem and works without the
        # backend extra installed.
        template = self._repo_root / ENV_TEMPLATE_NAME
        target = self._repo_root / ENV_FILE_NAME
        if not template.is_file():
            raise DeployError(
                "init",
                exit_code=2,
                stderr=f"template not found: {template}",
            )
        if target.exists() and not force:
            raise DeployError(
                "init",
                exit_code=2,
                stderr=(f"{target} already exists. Pass force=True to overwrite."),
            )
        shutil.copy2(template, target)
        logger.info("init: wrote %s from %s", target, template.name)


class Deploy(_DeployBase):
    """Sync local lumilake-stack lifecycle. Methods block until the
    corresponding ``lumilake.deploy`` function returns. Requires the
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

    def restart(self, service: str | None = None) -> None:
        _require_backend("restart")
        if service is not None:
            _wrap("restart", docker_client.container_restart, _container_for(service))
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
        ``lumilake.cli.commands.deploy._parse_since`` converts ``"10m"``
        style strings to a datetime when callers want CLI-style values.
        """
        _require_backend("logs")
        return _wrap(
            "logs",
            docker_client.container_logs_tail,
            _container_for(service),
            tail=tail,
            since=since,
            timestamps=timestamps,
        )

    def init(self, *, force: bool = False) -> None:
        """Create ``.env`` from the bundled template. Backend-free."""
        self._do_init(force)

    def update_flowmesh(self) -> None:
        _require_backend("update-flowmesh")
        _wrap("update-flowmesh", update_fm_mod.run_update, self._repo_root)
        logger.info("deploy update-flowmesh: ok")


class AsyncDeploy(_DeployBase):
    """Async local lumilake-stack lifecycle. Each method awaits the same
    ``lumilake.deploy`` work the sync class performs, dispatched through
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
