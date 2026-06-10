"""Built-in optimizer registry: ``halo`` and ``topological-sort``.

Select via ``LUMILAKE_OPTIMIZER_TYPE``; plugins extend via ``OPTIMIZER_PROVIDERS``.
"""

from lumilake import envs
from lumilake_hook import OptimizerHandle, OptimizerProvider

from lumilake_server.hooks.optimizer_providers import OPTIMIZER_PROVIDERS

from .base import BaseOptimizer
from .halo import HaloOptimizer
from .remote import RemoteOptimizer
from .topological_sort import TopologicalSortOptimizer

OPTIMIZER_TYPES: dict[str, type[BaseOptimizer]] = {
    "halo": HaloOptimizer,
    "topological-sort": TopologicalSortOptimizer,
}


def create_optimizer(
    optimizer_type: str | None = None, **kwargs
) -> BaseOptimizer | OptimizerHandle:
    """Return an optimizer instance, falling through to registered providers."""
    if optimizer_type is None:
        optimizer_type = envs.LUMILAKE_OPTIMIZER_TYPE
    optimizer_type = optimizer_type.lower()

    if optimizer_type in OPTIMIZER_TYPES:
        return OPTIMIZER_TYPES[optimizer_type](**kwargs)

    for provider in OPTIMIZER_PROVIDERS:
        provider_names = provider.list_optimizers()
        lower_to_original = {name.lower(): name for name in provider_names}
        if optimizer_type in lower_to_original:
            return provider.create_optimizer(
                lower_to_original[optimizer_type], **kwargs
            )

    remote_types: list[str] = []
    for p in OPTIMIZER_PROVIDERS:
        remote_types.extend(p.list_optimizers())
    raise ValueError(
        f"Unknown optimizer type '{optimizer_type}'. "
        f"Local: {sorted(OPTIMIZER_TYPES)}. "
        f"Provider-advertised: {sorted(set(remote_types))}."
    )


__all__ = [
    "BaseOptimizer",
    "HaloOptimizer",
    "OPTIMIZER_PROVIDERS",
    "OPTIMIZER_TYPES",
    "OptimizerProvider",
    "RemoteOptimizer",
    "TopologicalSortOptimizer",
    "create_optimizer",
]
