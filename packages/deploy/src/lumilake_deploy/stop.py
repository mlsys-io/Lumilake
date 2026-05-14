"""Stop the Lumilake stack."""

from pathlib import Path

from . import docker_client
from . import flowmesh as fm_impl
from .env import FLOWMESH_ENV_FILE_NAME
from .errors import DeployError
from .shell import info, run

SERVER_CONTAINER = "lumilake-server"
COMPOSE_PROFILES = (
    "postgres",
    "minio",
    "server",
)

# Volumes that accumulate state / runtime state across deploy
# cycles and cause duplicate-key / stale-retry errors on ``deploy up``
# when a prior run was killed mid-flight. ``--wipe-archive`` removes
# these without touching the staged corpus or research-records tables,
# which live in ``lumilake-minio-data``.
STATE_VOLUMES = (
    # Compute postgres state.
    "lumilake-postgres-data",
    # FlowMesh runtime state (jobs + task queue).
    "flowmesh-node_postgres_data",
    "flowmesh-node_redis_control_data",
    "flowmesh-node_redis_telemetry_data",
)


def _stop_flowmesh_stack(project_root: Path, *, purge: bool) -> None:
    env_file = project_root / FLOWMESH_ENV_FILE_NAME
    if not env_file.is_file():
        return
    info("Stopping FlowMesh stack...")
    fm_impl.stop_stack(env_file=env_file)
    if purge:
        fm_impl.clean_stack(env_file=env_file)
        env_file.unlink(missing_ok=True)


def _remove_state_volumes(project_root: Path) -> None:
    """Remove the volumes holding state postgres and FlowMesh runtime
    state. Skip volumes that don't exist."""
    for vol in STATE_VOLUMES:
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

    cmd: list[str] = ["docker", "compose"]
    for prof in COMPOSE_PROFILES:
        cmd.extend(["--profile", prof])
    cmd.append("down")
    if purge:
        cmd.append("-v")
        info("Purging volumes...")

    run(cmd, cwd=project_root, check=False)

    if wipe_archive and not purge:
        info("Wiping state state volumes (corpus + records preserved)...")
        _remove_state_volumes(project_root)

    info("All services stopped.")
