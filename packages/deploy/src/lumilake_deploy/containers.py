"""Service-to-container mapping for the Lumilake stack.

``LUMILAKE_STACK_NAME`` in ``deploy_dir/.env`` names the Compose project
and is carried in the server container and postgres volume names. The
FlowMesh side derives its ``FLOWMESH_STACK_SLUG`` from the same setting, so
the SDK looks the names up from the deployment directory instead of
hardcoding ``flowmesh_node_*``.
"""

from pathlib import Path

from . import flowmesh as fm_mod
from .env import ENV_FILE_NAME, FLOWMESH_ENV_FILE_NAME, read_env_value

SERVICE_NAMES: tuple[str, ...] = (
    "server",
    "flowmesh",
    "flowmesh-redis",
    "flowmesh-redis-telemetry",
)


def read_stack_name(deploy_dir: Path) -> str:
    """Return ``LUMILAKE_STACK_NAME`` from ``deploy_dir/.env``."""
    return read_env_value(deploy_dir / ENV_FILE_NAME, "LUMILAKE_STACK_NAME")


def server_container(name: str) -> str:
    """Server container name for a stack name (bare when unset)."""
    return f"lumilake-server-{name}" if name else "lumilake-server"


def server_data_volume(name: str) -> str:
    """Server postgres volume name for a stack name (bare when unset)."""
    return f"lumilake-postgres-data-{name}" if name else "lumilake-postgres-data"


def compose_project_args(deploy_dir: Path) -> list[str]:
    """``docker compose -p`` arguments selecting the stack's project."""
    name = read_stack_name(deploy_dir)
    return ["-p", name] if name else []


def _flowmesh_slug(deploy_dir: Path, name: str) -> str:
    """FlowMesh slug for ``deploy_dir``, or the bare base without a stack."""
    env_fm = deploy_dir / FLOWMESH_ENV_FILE_NAME
    if not env_fm.is_file():
        env_fm = deploy_dir / ENV_FILE_NAME
    return fm_mod.stack_slug(env_fm, stack_name=name)


def container_names(deploy_dir: Path) -> dict[str, str]:
    """Map each service name to the docker container name."""
    name = read_stack_name(deploy_dir)
    slug = _flowmesh_slug(deploy_dir, name)
    return {
        "server": server_container(name),
        "flowmesh": f"{slug}_server",
        "flowmesh-redis": f"{slug}_redis_control",
        "flowmesh-redis-telemetry": f"{slug}_redis_telemetry",
    }


def flowmesh_state_volumes(deploy_dir: Path) -> tuple[str, ...]:
    """Volume names FlowMesh creates for its postgres + redis state."""
    slug = _flowmesh_slug(deploy_dir, read_stack_name(deploy_dir))
    return (
        f"{slug}_postgres_data",
        f"{slug}_redis_control_data",
        f"{slug}_redis_telemetry_data",
    )
