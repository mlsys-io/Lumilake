import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Self, TypeVar

from lumilake_server.common import Message
from lumilake_server.utils.utils import unique_id

T = TypeVar("T")
SingleDtype = str | list[Message]


class OpRegistry:
    def __init__(self, name: str = "ops"):
        self._name = name
        self._registry: dict[str, type[Op]] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        def _register(cls):
            # Use provided name or default to class name
            key = name if name is not None else cls.__name__
            if key in self._registry:
                raise ValueError(f"'{key}' is already registered in {self._name}")
            self._registry[key] = cls
            return cls

        return _register

    def get_class(self, key: Any) -> type["Op"]:
        if key not in self._registry:
            raise KeyError(f"'{key}' is not registered in {self._name}")
        return self._registry[key]


class Op(ABC):
    id: str
    inputs: list["Op"]
    registry: OpRegistry = OpRegistry()

    def __init__(self, inputs: list["Op"] | None = None) -> None:
        self.id = unique_id()
        self.inputs = [] if inputs is None else inputs
        self._max_iter: int | None = None

    @property
    def looping(self) -> bool:
        return self._max_iter is not None

    @property
    def max_iter(self) -> int | None:
        """
        Maximum number of iterations. There are three cases:
        - None: No looping.
        - -1: Infinite looping.
        - n: Looping for n times.
        """
        return self._max_iter

    def copy(self, new_id: bool = True) -> Self:
        new_op = copy.copy(self)
        # Create new unique attributes
        if new_id:
            new_op.id = unique_id()
        new_op.inputs = new_op.inputs.copy()
        return new_op

    def set_looping(self, looping: bool = True, max_iter: int | None = None) -> None:
        """
        Set the looping parameters.

        Parameters
        ----------
        looping : bool
            Whether to enable looping.
        max_iter : int | None
            Maximum number of iterations. There are three cases:
            - None: No looping.
            - -1: Infinite looping.
            - n: Looping for n times.
        """
        if looping:
            self._max_iter = -1 if max_iter is None else max_iter
        else:
            self._max_iter = None

    def _traverse_graph(self, graph: dict[str, "Op"]):
        assert self.id not in graph

        graph[self.id] = self
        for op in self.inputs:
            if op.id not in graph:
                op._traverse_graph(graph)

    def traverse_graph(self, graph: dict[str, "Op"] | None = None) -> dict[str, "Op"]:
        graph = {} if graph is None else graph
        self._traverse_graph(graph)
        return graph

    @abstractmethod
    def _serialize(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def _from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "Op":
        pass

    def serialize(self) -> dict[str, Any]:
        serialized = self._serialize()
        serialized["_id"] = self.id
        serialized["_op"] = self.__class__.__name__
        serialized["_max_iter"] = self._max_iter
        serialized["_inputs"] = [op.id for op in self.inputs]
        return serialized

    @classmethod
    def from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "Op":
        op = cls._from_json(data, other_ops)
        op.id = data["_id"]
        max_iter = data["_max_iter"]
        op.set_looping(max_iter is not None, max_iter)
        return op

    def __hash__(self) -> int:
        return hash(self.id)


class FunctionalOp(Op):
    pass
