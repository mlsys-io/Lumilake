"""Stop the Lumilake stack."""

from pathlib import Path

from . import docker_client
from . import flowmesh as fm_impl
from .assets import compose_path
from .containers import flowmesh_state_volumes
from .env import FLOWMESH_ENV_FILE_NAME
from .errors import DeployError
from .shell import info, run

SERVER_CONTAINER = "lumilake-server"
COMPOSE_PROFILES = (
    "postgres",
    "minio",
    "server",
)


def _state_volumes(project_root: Path) -> tuple[str, ...]:
    """Volumes that ``--wipe-archive`` removes — local compute postgres
    plus the FlowMesh runtime state derived from the operator's
    ``.env.flowmesh`` slug. Touches no corpus / research-records data
    (those live under ``lumilake-minio-data``)."""
    return ("lumilake-postgres-data", *flowmesh_state_volumes(project_root))


def _stop_flowmesh_stack(project_root: Path, *, purge: bool) -> None:
    env_file = project_root / FLOWMESH_ENV_FILE_NAME
    if not env_file.is_file():
        return
    info("Stopping FlowMesh stack...")
    fm_impl.stop_stack(env_file=env_file)
    if purge:
        # Drop containers + volumes, but keep .env.flowmesh so a subsequent
        # ``deploy up`` doesn't need a fresh ``deploy init --flowmesh``.
        fm_impl.clean_stack(env_file=env_file)


def _remove_state_volumes(project_root: Path) -> None:
    """Remove the volumes holding state postgres and FlowMesh runtime
    state. Skip volumes that don't exist."""
    for vol in _state_volumes(project_root):
        # Compose prefixes volumes with the project name; try both.
        for candidate in (vol, f"{project_root.name}_{vol}"):
            if not docker_client.volume_exists(candidate):
                continue
            if docker_client.volume_remove(candidate):
                info(f"Removed volume {candidate}")
            else:
                info(
                    f"WARNING: failed to remove volume {candidate} "
                    "(still in use by a running container?); "
                    "state state may persist into next deploy up."
                )
            break


def run_stop(
    project_root: Path, *, purge: bool = False, wipe_archive: bool = False
) -> None:
    """Stop all services.

    Pass ``purge=True`` to remove *all* docker volumes (loses corpus +
    research records; equivalent of ``deploy reset``).

    Pass ``wipe_archive=True`` (without ``purge``) to wipe just the
    state / runtime state that accumulates across deploy cycles.
    """
    engine_up = docker_client.engine_is_up()

    if engine_up:
        try:
            if docker_client.container_stop(SERVER_CONTAINER):
                info(f"Stopped {SERVER_CONTAINER}")
        except DeployError as exc:
            info(f"WARNING: {exc}")
    else:
        info("WARNING: Cannot connect to Docker. Containers may still be running.")

    _stop_flowmesh_stack(project_root, purge=purge)

    cmd: list[str] = [
        "docker",
        "compose",
        "-f",
        str(compose_path()),
        "--project-directory",
        str(project_root),
    ]
    for prof in COMPOSE_PROFILES:
        cmd.extend(["--profile", prof])
    cmd.append("down")
    if purge:
        cmd.append("-v")
        info("Purging volumes...")

    run(cmd, cwd=project_root, check=False)

    if wipe_archive and not purge:
        info("Wiping state volumes (corpus + records preserved)...")
        _remove_state_volumes(project_root)

    info("All services stopped.")
