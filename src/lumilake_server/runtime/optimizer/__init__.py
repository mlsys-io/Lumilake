"""Optimizer implementations for Lumilake.

Ships two optimizers:

- ``halo`` — cost-aware DP scheduler over the runtime graph with a
  multimodal cost model.
- ``topological-sort`` — naive baseline that pins every GPU node to
  the first GPU worker and every CPU node to the first CPU worker, in
  topological order. Used for measurement comparisons against HALO.

Select at runtime via ``LUMILAKE_OPTIMIZER_TYPE``.
"""

from lumilake import envs

from .base import BaseOptimizer
from .halo import HaloOptimizer
from .topological_sort import TopologicalSortOptimizer

OPTIMIZER_TYPES: dict[str, type[BaseOptimizer]] = {
    "halo": HaloOptimizer,
    "topological-sort": TopologicalSortOptimizer,
}


def create_optimizer(optimizer_type: str | None = None, **kwargs) -> BaseOptimizer:
    """Create an optimizer instance.

    Parameters
    ----------
    optimizer_type : str | None
        Optimizer name. If ``None``, uses ``LUMILAKE_OPTIMIZER_TYPE``.
    **kwargs
        Passed to the optimizer constructor.
    """
    if optimizer_type is None:
        optimizer_type = envs.LUMILAKE_OPTIMIZER_TYPE
    optimizer_type = optimizer_type.lower()

    if optimizer_type not in OPTIMIZER_TYPES:
        raise ValueError(
            f"Unknown optimizer type '{optimizer_type}'. "
            f"Available: {', '.join(OPTIMIZER_TYPES)}."
        )
    return OPTIMIZER_TYPES[optimizer_type](**kwargs)


__all__ = [
    "BaseOptimizer",
    "HaloOptimizer",
    "TopologicalSortOptimizer",
    "OPTIMIZER_TYPES",
    "create_optimizer",
]
