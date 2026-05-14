"""Optimizer implementations for Lumilake.

Ships ``halo`` — the HALO cost-aware scheduler that uses a DP solver
over the runtime graph plus a multimodal cost model.

Select at runtime via ``LUMILAKE_OPTIMIZER_TYPE``.
"""

from lumilake import envs

from .base import BaseOptimizer
from .halo import HaloOptimizer

OPTIMIZER_TYPES: dict[str, type[BaseOptimizer]] = {
    "halo": HaloOptimizer,
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
    "OPTIMIZER_TYPES",
    "create_optimizer",
]
