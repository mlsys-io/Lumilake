"""Deploy commands: init, doctor, up, down, clean, purge, restart, reset, logs."""

import datetime as dt
import difflib
import re
import sys
from pathlib import Path

import typer
from flowmesh_cli_stack.stack import stack_env_example
from lumilake import envs
from lumilake_deploy import docker_client
from lumilake_deploy import purge as purge_mod
from lumilake_deploy import setup as setup_mod
from lumilake_deploy import stop as stop_mod
from lumilake_deploy import update_flowmesh as update_fm_mod
from lumilake_deploy.assets import env_example_path
from lumilake_deploy.containers import SERVICE_NAMES, container_names
from lumilake_deploy.doctor import DoctorFinding, run_doctor
from lumilake_deploy.env import (
    ENV_FILE_NAME,
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
from ..core.config import DEFAULT_CONFIG_PATH, LumilakeConfig, load_config, save_config
from ..core.typer import get_typer

app = get_typer(
    help=(
        "Deploy and manage the Lumilake stack. Reads .env (and optionally "
        ".env.flowmesh) from --project-dir (or the current working "
        "directory). The packaged compose file and server image are "
        "resolved from the installed lumilake-deploy package."
    )
)


@app.callback()
def _deploy_callback(
    ctx: typer.Context,
    project_dir: Path = typer.Option(
        None,
        "--project-dir",
        "-C",
        envvar="LUMILAKE_DEPLOY_DIR",
        help=(
            "Directory holding .env / .env.flowmesh and where compose "
            "stores runtime state. Defaults to the current working directory."
        ),
    ),
) -> None:
    ctx.obj = project_dir.resolve() if project_dir else Path.cwd()


def _project_dir(ctx: typer.Context) -> Path:
    if isinstance(ctx.obj, Path):
        return ctx.obj
    return Path.cwd()


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


def _preview_write(target: Path, new_content: str) -> None:
    if target.exists():
        old = target.read_text().splitlines(keepends=True)
        new = new_content.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                old,
                new,
                fromfile=f"a/{target.name}",
                tofile=f"b/{target.name}",
                n=3,
            )
        )
        if diff:
            logging.info(f"Diff vs existing {target}:")
            for line in diff.splitlines():
                logging.info(line)
        else:
            logging.info(f"{target} already matches the bundled template.")
        return
    logging.info(f"Preview of {target} (first 20 lines):")
    for line in new_content.splitlines()[:20]:
        logging.info(line)
    remaining = max(0, len(new_content.splitlines()) - 20)
    if remaining:
        logging.info(f"... and {remaining} more line(s).")


def _confirm_overwrite(target: Path, *, force: bool, new_content: str) -> bool:
    """Return True when the target may be written, False to skip.

    Shows a preview (or diff vs an existing file) before prompting on
    overwrite. Brand-new writes pass through without prompting.
    """
    if force:
        return True
    _preview_write(target, new_content)
    if not target.exists():
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
    ctx: typer.Context,
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
    """Initialize ``.env`` from the bundled ``.env.example`` template."""
    root = _project_dir(ctx)
    template = env_example_path()
    target = root / ENV_FILE_NAME
    template_text = template.read_text()
    wrote_env = False
    if _confirm_overwrite(target, force=force, new_content=template_text):
        target.write_text(template_text)
        logging.success(f"Wrote {target} from packaged {template.name}.")
        wrote_env = True

    if flowmesh:
        fm_target = root / FLOWMESH_ENV_FILE_NAME
        example = stack_env_example()
        if not example.exists():
            logging.error(f"FlowMesh env example not found: {example}")
            raise typer.Exit(code=1)
        fm_template_text = example.read_text()
        wrote_flowmesh_env = False
        if _confirm_overwrite(fm_target, force=force, new_content=fm_template_text):
            fm_target.write_text(fm_template_text)
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
    ctx: typer.Context,
    flowmesh: bool = typer.Option(
        False,
        "--flowmesh",
        help="Also validate that ``.env.flowmesh`` is present.",
    ),
) -> None:
    """Validate ``.env`` (and optionally ``.env.flowmesh``)."""
    root = _project_dir(ctx)

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
def build(ctx: typer.Context) -> None:
    """Build the lumilake server Docker image from source."""
    root = _project_dir(ctx)
    setup_mod.load_project_env(root)
    image_tag = setup_mod.resolve_image_tag()
    try:
        build_server_image(root, image_tag)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def pull(ctx: typer.Context) -> None:
    """Pull the published lumilake server image from the registry."""
    root = _project_dir(ctx)
    setup_mod.load_project_env(root)
    image_tag = setup_mod.resolve_image_tag()
    try:
        pull_server_image(image_tag)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc


def _write_cli_config(root: Path, config_path: Path | None = None) -> None:
    """Persist the local server URL so subsequent CLI / SDK calls find it.

    Reads the deployment's ``.env`` to compute the host port the server
    is listening on, then writes ``base_url`` to ``config_path``. If a
    config already exists with a different ``base_url``, we log a hint
    and overwrite. Errors are non-fatal — the stack is already running.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    setup_mod.load_project_env(root)
    port = envs.LUMILAKE_SERVER_PORT or 9000
    base_url = f"http://127.0.0.1:{port}"
    try:
        existing = load_config(config_path)
    except (FileNotFoundError, ValueError):
        existing = None
    if existing is not None and existing.base_url == base_url:
        logging.info(f"CLI config already points at {base_url}.")
        return
    if existing is not None and existing.base_url != base_url:
        logging.info(f"Updating {config_path}: {existing.base_url} -> {base_url}")
    try:
        save_config(LumilakeConfig(base_url=base_url), path=config_path)
    except OSError as exc:
        logging.warning(f"Could not save CLI config to {config_path}: {exc}")
        return
    logging.info(f"Saved CLI config to {config_path} (base_url={base_url}).")


@app.command()
def up(ctx: typer.Context) -> None:
    """Start the full Lumilake stack (Docker-backed).

    The server image must already be present locally — run
    ``lumilake deploy pull`` or ``lumilake deploy build`` first.
    """
    root = _project_dir(ctx)
    _run_setup(root, background=True)
    _write_cli_config(root)


@app.command()
def down(
    ctx: typer.Context,
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
    """Stop the stack but keep data volumes.

    Safe to run between sessions: the archive bucket (job records, run
    artifacts) and the compute postgres/minio volumes survive, so
    ``deploy up`` resumes against the same state. ``--wipe-archive`` also
    removes the compute Postgres and FlowMesh runtime-state volumes, while
    preserving MinIO corpus data. Use ``deploy reset`` (destructive) to
    wipe every volume instead.
    """
    try:
        stop_mod.run_stop(
            _project_dir(ctx),
            purge=False,
            wipe_archive=wipe_archive,
        )
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def clean(ctx: typer.Context) -> None:
    """Stop all services and delete volumes."""
    try:
        stop_mod.run_stop(_project_dir(ctx), purge=True)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc


@app.command()
def purge(
    ctx: typer.Context,
    image_tag: str = typer.Argument(
        ...,
        help="Lumilake server image tag to purge from local Docker.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="List the image to be purged without deleting it.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force image removal even if Docker reports dependent containers.",
    ),
) -> None:
    """Purge one Lumilake server image tag from local Docker."""
    try:
        plan = purge_mod.build_server_image_purge_plan(
            _project_dir(ctx),
            image_tag=image_tag,
        )
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc

    logging.info(f"Purging Lumilake server image with tag '{image_tag}'...")
    if not plan.exists:
        logging.warning(f"Image not found: {plan.image_ref}")
        logging.info("No images to purge.")
        return

    if dry_run:
        logging.info("Images to be purged:")
        logging.info(f"  {plan.image_ref}")
        return

    try:
        result = purge_mod.run_server_image_purge(plan, force=force)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc
    if result.removed:
        logging.success(f"Removed image: {result.image_ref}")
        return
    logging.warning(f"Image already absent: {result.image_ref}")


@app.command()
def status(ctx: typer.Context) -> None:
    """Show the running state of every Lumilake stack container."""
    rows: list[tuple[str, str, str, str]] = []
    for service, container in container_names(_project_dir(ctx)).items():
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
    ctx: typer.Context,
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
    root = _project_dir(ctx)
    if service is not None:
        container = container_names(root).get(service)
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

    try:
        stop_mod.run_stop(root, purge=False)
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc
    _run_setup(root, background=True)


@app.command()
def reset(
    ctx: typer.Context,
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the destructive-action confirmation prompt.",
    ),
) -> None:
    """Wipe the archive and every volume, then start the stack fresh.

    Destructive: removes the compute postgres / minio volumes (job
    records, run artifacts, demo data) and the FlowMesh runtime state.
    Use ``deploy down`` if you want to stop the stack but keep its data.
    """
    if not yes and not typer.confirm(
        "deploy reset deletes every Lumilake volume (archive + compute "
        "data). Continue?",
        default=False,
    ):
        logging.info("Aborted.")
        raise typer.Exit(code=0)
    root = _project_dir(ctx)
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
    ctx: typer.Context,
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
    container = container_names(_project_dir(ctx)).get(service)
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
def update_flowmesh_cmd(ctx: typer.Context) -> None:
    """Re-lock and install the latest FlowMesh packages."""
    try:
        update_fm_mod.run_update(_project_dir(ctx))
    except DeployError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1) from exc
