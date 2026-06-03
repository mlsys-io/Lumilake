from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lumilake_server.runtime.optimizer.base import BaseOptimizer


@runtime_checkable
class OptimizerProvider(Protocol):
    """Plugin hook: contributes optimizer types beyond the local registry.

    ``create_optimizer(type)`` consults providers when ``type`` isn't in
    ``OPTIMIZER_TYPES``.

    Lumilake may call ``list_optimizers()`` multiple times per request — during
    submission validation, at the ``/optimizer/list`` endpoint, and inside the
    factory lookup. If your underlying source is slow or unreliable, cache the
    result inside your provider implementation (e.g. populate a private
    attribute on first call and reuse it). See ``docs/PLUGINS.md`` for a worked
    example.

    Names returned by ``list_optimizers()`` are compared case-insensitively.
    The platform lowercases display names; providers should return names in
    whatever casing fits their backend.
    """

    def list_optimizers(self) -> list[str]: ...

    def create_optimizer(
        self, optimizer_type: str, **kwargs: Any
    ) -> "BaseOptimizer": ...
