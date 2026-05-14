"""Service-to-container mapping for the Lumilake stack.

FlowMesh names containers and volumes from ``FLOWMESH_STACK_SLUG`` (the
slugged form of ``FLOWMESH_STACK_SUFFIX``). ``lumilake deploy init
--flowmesh`` sets the suffix to ``lumilake``, so the SDK can't hardcode
``flowmesh_node_*`` — it has to look the slug up from ``.env.flowmesh``
in the operator's deployment directory.
"""

from pathlib import Path

from . import flowmesh as fm_mod
from .env import FLOWMESH_ENV_FILE_NAME

SERVICE_NAMES: tuple[str, ...] = (
    "server",
    "flowmesh",
    "flowmesh-redis",
    "flowmesh-redis-telemetry",
)


def container_names(deploy_dir: Path) -> dict[str, str]:
    """Map each service name to the docker container name.

    Looks up ``FLOWMESH_STACK_SLUG`` from ``deploy_dir/.env.flowmesh``
    when present; falls back to ``flowmesh_node`` so the SDK keeps
    pre-init behavior intact.
    """
    env_fm = deploy_dir / FLOWMESH_ENV_FILE_NAME
    slug = fm_mod.stack_slug(env_fm) if env_fm.is_file() else "flowmesh_node"
    return {
        "server": "lumilake-server",
        "flowmesh": f"{slug}_server",
        "flowmesh-redis": f"{slug}_redis_control",
        "flowmesh-redis-telemetry": f"{slug}_redis_telemetry",
    }


def flowmesh_state_volumes(deploy_dir: Path) -> tuple[str, ...]:
    """Volume names FlowMesh creates for its postgres + redis state."""
    env_fm = deploy_dir / FLOWMESH_ENV_FILE_NAME
    slug = fm_mod.stack_slug(env_fm) if env_fm.is_file() else "flowmesh_node"
    return (
        f"{slug}_postgres_data",
        f"{slug}_redis_control_data",
        f"{slug}_redis_telemetry_data",
    )
