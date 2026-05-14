"""Deploy commands: init, doctor, up, down, clean, restart, reset, logs."""

import datetime as dt
import re
import shutil
import sys
from pathlib import Path

import typer
from flowmesh_cli_stack.stack import stack_env_example
from lumilake import envs
from lumilake_deploy import docker_client
from lumilake_deploy import flowmesh as fm_mod
from lumilake_deploy import setup as setup_mod
from lumilake_deploy import stop as stop_mod
from lumilake_deploy import update_flowmesh as update_fm_mod
from lumilake_deploy.doctor import DoctorFinding, run_doctor
from lumilake_deploy.env import (
    ENV_FILE_NAME,
    ENV_TEMPLATE_NAME,
    FLOWMESH_ENV_FILE_NAME,
    patch_env_value,
)
from lumilake_deploy.errors import DeployError
from lumilake_deploy.setup import (
    SetupOptions,
    build_server_image,
    pull_server_image,
)

from ..core import logging
from ..core.typer import get_typer

app = get_typer(help="Deploy and manage the Lumilake stack.")


def _find_project_root() -> Path:
    """Walk up from cwd looking for the env template."""
    candidate = Path.cwd()
    for _ in range(10):
        if (candidate / ENV_TEMPLATE_NAME).is_file():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    logging.error(
        f"Cannot find project root (no {ENV_TEMPLATE_NAME}). "
        "Run from the project directory."
    )
    raise typer.Exit(code=1)


def _run_setup(
    root: Path,
    *,
    background: bool = True,
    reset: bool = False,
    no_server: bool = False,
) -> None:
    try:
        setup_mod.run_setup(
            root,
            SetupOptions(
                reset=reset,
                no_server=no_server,
                background=background,
            ),
        )
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc


def _confirm_overwrite(target: Path, *, force: bool) -> bool:
    """Return True when the target may be written, False to skip."""
    if not target.exists() or force:
        return True
    if typer.confirm(f"{target} already exists. Overwrite?", default=False):
        return True
    logging.info(f"Keeping existing {target}.")
    return False


def _flowmesh_env_value(env_path: Path, key: str) -> str:
    """Read a single line from ``.env.flowmesh`` (no quoting strip pattern)."""
    if not env_path.is_file():
        return ""
    prefix = f"{key}="
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :]
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value
    return ""


_FLOWMESH_LOCAL_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("FLOWMESH_STACK_SUFFIX", "lumilake"),
    ("SERVER_GRPC_TLS_CA_FILE", ""),
    ("SERVER_GRPC_TLS_CERT_FILE", ""),
    ("SERVER_GRPC_TLS_KEY_FILE", ""),
    ("REDIS_TLS_CA_FILE", ""),
    ("REDIS_TLS_CERT_FILE", ""),
    ("REDIS_TLS_KEY_FILE", ""),
)


def _patch_flowmesh_env(env_path: Path) -> None:
    """Disable TLS bind mounts on the bundled FlowMesh template."""
    for key, value in _FLOWMESH_LOCAL_OVERRIDES:
        patch_env_value(env_path, key, value)


def _cross_populate_runtime(env_path: Path, fm_env_path: Path) -> None:
    """After ``init --flowmesh`` writes both files, point ``.env``'s
    ``LUMILAKE_RUNTIME_ORCHESTRATOR_URL`` at the FlowMesh server's port."""
    server_port = _flowmesh_env_value(fm_env_path, "SERVER_HTTP_PORT") or "18000"
    patch_env_value(
        env_path,
        "LUMILAKE_RUNTIME_ORCHESTRATOR_URL",
        f"http://127.0.0.1:{server_port}",
    )


@app.command("init")
def init(
    flowmesh: bool = typer.Option(
        False,
        "--flowmesh",
        help="Also generate ``.env.flowmesh`` from FlowMesh's stack template.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing env files without prompting.",
    ),
) -> None:
    """Initialize ``.env`` from ``.env.example``."""
    root = _find_project_root()
    template = root / ENV_TEMPLATE_NAME
    target = root / ENV_FILE_NAME
    if not template.is_file():
        logging.error(f"Template not found: {template}")
        raise typer.Exit(code=1)
    wrote_env = False
    if _confirm_overwrite(target, force=force):
        shutil.copy2(template, target)
        logging.success(f"Wrote {target} from {template.name}.")
        wrote_env = True

    if flowmesh:
        fm_target = root / FLOWMESH_ENV_FILE_NAME
        wrote_flowmesh_env = False
        if _confirm_overwrite(fm_target, force=force):
            example = stack_env_example()
            if not example.exists():
                logging.error(f"FlowMesh env example not found: {example}")
                raise typer.Exit(code=1)
            fm_target.write_text(example.read_text())
            logging.success(f"Wrote {fm_target} from {example.name}.")
            _patch_flowmesh_env(fm_target)
            wrote_flowmesh_env = True
        if wrote_env:
            _cross_populate_runtime(target, fm_target)
        elif wrote_flowmesh_env:
            logging.info(
                f"Keeping existing {target}; not patching runtime URL from "
                f"{fm_target}."
            )


@app.command()
def doctor(
    flowmesh: bool = typer.Option(
        False,
        "--flowmesh",
        help="Also validate that ``.env.flowmesh`` is present.",
    ),
) -> None:
    """Validate ``.env`` (and optionally ``.env.flowmesh``)."""
    root = _find_project_root()

    def _emit(finding: DoctorFinding) -> None:
        if finding.level == "error":
            logging.error(finding.message)
        elif finding.level == "warning":
            logging.warning(finding.message)
        else:
            logging.info(finding.message)

    report = run_doctor(
        root / ENV_FILE_NAME,
        flowmesh_env_path=(root / FLOWMESH_ENV_FILE_NAME) if flowmesh else None,
        callback=_emit,
    )
    if report.errors:
        logging.error(f"Doctor found {len(report.errors)} issue(s).")
        raise typer.Exit(code=1)
    logging.success("Doctor checks passed.")


@app.command()
def build() -> None:
    """Build the lumilake server Docker image from source."""
    root = _find_project_root()
    setup_mod.load_project_env(root)
    image_tag = envs.LUMILAKE_IMAGE_TAG or "latest"
    try:
        build_server_image(root, image_tag)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def pull() -> None:
    """Pull the published lumilake server image from the registry."""
    root = _find_project_root()
    setup_mod.load_project_env(root)
    image_tag = envs.LUMILAKE_IMAGE_TAG or "latest"
    try:
        pull_server_image(image_tag)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def up() -> None:
    """Start the full Lumilake stack (Docker-backed).

    The server image must already be present locally — run
    ``lumilake deploy pull`` or ``lumilake deploy build`` first.
    """
    _run_setup(_find_project_root(), background=True)


@app.command()
def down(
    wipe_archive: bool = typer.Option(
        False,
        "--wipe-archive",
        help=(
            "Also wipe runtime state that accumulates across "
            "deploy cycles (FlowMesh postgres + redis). "
            "Fixes the 'Duplicate key' / silent-retry-loop failure mode "
            "after a killed prior run. Use this between experiment runs "
            "if you don't want a full "
            "``deploy reset`` + re-stage."
        ),
    ),
) -> None:
    """Stop all services (keep data)."""
    try:
        stop_mod.run_stop(
            _find_project_root(),
            purge=False,
            wipe_archive=wipe_archive,
        )
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def clean() -> None:
    """Stop all services and delete volumes."""
    try:
        stop_mod.run_stop(_find_project_root(), purge=True)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc


SERVICE_NAMES = (
    "server",
    "flowmesh",
    "flowmesh-redis",
    "flowmesh-redis-telemetry",
)


def _container_names() -> dict[str, str]:
    """Resolve service-name → container-name, slugging FlowMesh from the env."""
    env_fm = _find_project_root() / FLOWMESH_ENV_FILE_NAME
    slug = fm_mod.stack_slug(env_fm) if env_fm.is_file() else "flowmesh_node"
    return {
        "server": "lumilake-server",
        "flowmesh": f"{slug}_server",
        "flowmesh-redis": f"{slug}_redis_control",
        "flowmesh-redis-telemetry": f"{slug}_redis_telemetry",
    }


@app.command()
def status() -> None:
    """Show the running state of every Lumilake stack container."""
    rows: list[tuple[str, str, str, str]] = []
    for service, container in _container_names().items():
        state = docker_client.container_status(container)
        health = (
            docker_client.container_health_status(container)
            if state == "running"
            else ""
        )
        rows.append((service, container, state, health))

    svc_w = max(len(r[0]) for r in rows)
    cnt_w = max(len(r[1]) for r in rows)
    state_w = max(len(r[2]) for r in rows)
    for service, container, state, health in rows:
        line = f"{service:<{svc_w}}  {container:<{cnt_w}}  {state:<{state_w}}"
        if health:
            line += f"  ({health})"
        logging.info(line)


@app.command()
def restart(
    service: str | None = typer.Argument(
        None,
        help=(
            "Single service to restart. Choose from: "
            f"{', '.join(SERVICE_NAMES)}. When omitted, restarts the "
            "entire stack (re-runs setup, rebuilds images)."
        ),
    ),
) -> None:
    """Restart deployment services.

    Without ``service``: stops the entire stack and re-runs setup
    (rebuilds images if code changed). With ``service``: restarts just
    that container in place.
    """
    if service is not None:
        container = _container_names().get(service)
        if container is None:
            logging.error(
                f"Unknown service '{service}'. "
                f"Choose from: {', '.join(SERVICE_NAMES)}"
            )
            raise typer.Exit(code=1)
        try:
            docker_client.container_restart(container)
        except DeployError as exc:
            logging.error(str(exc))
            raise typer.Exit(code=1) from exc
        return

    root = _find_project_root()
    try:
        stop_mod.run_stop(root, purge=False)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc
    _run_setup(root, background=True)


@app.command()
def reset() -> None:
    """Clean reset (stop + purge + up; deletes all data)."""
    root = _find_project_root()
    try:
        stop_mod.run_stop(root, purge=True)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc
    _run_setup(root, background=True, reset=True)


_SINCE_RE = re.compile(r"^(\d+)([smhd])$")


def _parse_since(value: str | None) -> dt.datetime | None:
    """Parse ``10m`` / ``2h`` / ``1d`` / ``30s`` into a past ``datetime``.

    Returns ``None`` when ``value`` is ``None``. Exits the CLI on an
    unparsable value.
    """
    if value is None:
        return None
    match = _SINCE_RE.match(value.strip())
    if not match:
        logging.error(
            f"--since must be one of <N>s/m/h/d (e.g. 30s, 10m, 2h, 1d); got {value!r}"
        )
        raise typer.Exit(code=1)
    amount = int(match.group(1))
    unit = match.group(2)
    delta = {
        "s": dt.timedelta(seconds=amount),
        "m": dt.timedelta(minutes=amount),
        "h": dt.timedelta(hours=amount),
        "d": dt.timedelta(days=amount),
    }[unit]
    return dt.datetime.now(dt.UTC) - delta


@app.command()
def logs(
    service: str = typer.Argument(
        "server",
        help=f"Service name: {', '.join(SERVICE_NAMES)}",
    ),
    tail: int | None = typer.Option(
        None, "--tail", "-n", help="Show last N lines and exit (default: follow live)."
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help=(
            "Only show logs newer than this duration: "
            "<N>s, <N>m, <N>h, <N>d (e.g. 30s, 10m, 2h, 1d)."
        ),
    ),
    timestamps: bool = typer.Option(
        False,
        "--timestamps",
        "-t",
        help="Prepend an RFC3339 timestamp to each log line.",
    ),
) -> None:
    """Show logs for a Lumilake service.

    Follows by default; use ``-n`` to show the last N lines and exit.
    ``--since`` filters to entries newer than the given relative duration.
    """
    container = _container_names().get(service)
    if container is None:
        logging.error(
            f"Unknown service '{service}'. Choose from: {', '.join(SERVICE_NAMES)}"
        )
        raise typer.Exit(code=1)
    since_dt = _parse_since(since)
    try:
        if tail is not None:
            output = docker_client.container_logs_tail(
                container, tail=tail, since=since_dt, timestamps=timestamps
            )
            sys.stdout.write(output)
            if output and not output.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            for chunk in docker_client.container_logs_stream(
                container, since=since_dt, timestamps=timestamps
            ):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        return


@app.command("update-flowmesh")
def update_flowmesh_cmd() -> None:
    """Re-lock and install the latest FlowMesh packages."""
    try:
        update_fm_mod.run_update(_find_project_root())
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc
