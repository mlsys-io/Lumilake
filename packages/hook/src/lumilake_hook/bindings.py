from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from lumid_hooks import BaseBindings as _LumidBaseBindings
from lumid_hooks import HookBindings as _LumidHookBindings

from .optimizer import OptimizerProvider


@runtime_checkable
class HookBindings(_LumidHookBindings, Protocol):
    @property
    def optimizer_providers(self) -> Sequence[OptimizerProvider]: ...


@dataclass(frozen=True)
class BaseBindings(_LumidBaseBindings):
    optimizer_providers: Sequence[OptimizerProvider] = field(default_factory=tuple)


__all__: list[str] = ["BaseBindings", "HookBindings"]
