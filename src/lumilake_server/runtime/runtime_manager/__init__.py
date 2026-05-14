"""Runtime manager implementations."""

from lumilake import envs

from .base import BaseRuntimeManager
from .flowmesh import FlowmeshRuntimeManager

RUNTIME_MANAGER_TYPES: dict[str, type[FlowmeshRuntimeManager]] = {
    "default": FlowmeshRuntimeManager,
    "flowmesh": FlowmeshRuntimeManager,
}


def create_runtime_manager(
    runtime_manager_type: str | None = None, **kwargs
) -> FlowmeshRuntimeManager:
    """Create a runtime manager instance based on type."""
    if runtime_manager_type is None:
        runtime_manager_type = envs.LUMILAKE_RUNTIME_MANAGER_TYPE
    runtime_manager_type = runtime_manager_type.lower()
    if runtime_manager_type not in RUNTIME_MANAGER_TYPES:
        valid_types = ", ".join(RUNTIME_MANAGER_TYPES.keys())
        raise ValueError(
            f"Unknown runtime manager type '{runtime_manager_type}'. Valid types:"
            f" {valid_types}"
        )
    return RUNTIME_MANAGER_TYPES[runtime_manager_type](**kwargs)


__all__ = [
    "BaseRuntimeManager",
    "FlowmeshRuntimeManager",
    "create_runtime_manager",
    "RUNTIME_MANAGER_TYPES",
]
