import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from lumilake_server.common import Message
from lumilake_server.utils.utils import unique_id

T = TypeVar("T")
SingleDtype = str | list[Message]


class RecordKey(BaseModel):
    """A record key an op declares: the node that carries its rows' keys, and
    the path into it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: StrictStr = Field(min_length=1)
    path: StrictStr = ""


def record_key(key: Any, op_id: str) -> RecordKey:
    """An op's declared record key; an invalid one raises naming the field."""
    try:
        return RecordKey.model_validate(key)
    except ValidationError as exc:
        error = exc.errors()[0]
        field = ".".join(str(part) for part in ("key", *error["loc"]))
        raise ValueError(f"op '{op_id}' field '{field}' {error['msg']}") from exc


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


class RowShape(Enum):
    """How many values one item of an op's output holds: a per-row item holds
    one value, and a grouped item holds a table's values."""

    PER_ROW = "per_row"
    GROUPED = "grouped"


class Op(ABC):
    id: str
    inputs: list["Op"]
    registry: OpRegistry = OpRegistry()
    declared_id: str | None = None
    """The id the workflow declared for this op, surviving scope and slicing."""

    @property
    def row_shape(self) -> RowShape:
        """The shape of one output item: a consumer of a row-wise op reads one
        value per input row."""
        return RowShape.PER_ROW

    @property
    def profilable(self) -> bool:
        """Whether a preflight may profile the op with the data profiler."""
        return False

    @property
    def samples(self) -> bool:
        """Whether the op's output is a random draw."""
        return False

    result_path: str | None = None
    """The result-envelope path an output archives this op's payload at, or
    ``None`` when the op's own item shape is the payload."""

    @property
    def row_source(self) -> "Op | None":
        """The input whose rows this op emits one output row per, or ``None``
        when the row count is not the input's."""
        return None

    @property
    def keys_own_records(self) -> bool:
        """Whether the op gives each of its own records a key: an expand child,
        a reduce group or a join row."""
        return False

    def __init__(self, inputs: list["Op"] | None = None) -> None:
        self.id = unique_id()
        self.inputs = [] if inputs is None else inputs
        self._max_iter: int | None = None
        self.key: RecordKey | None = None

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
        if new_op.key is not None:
            new_op.key = new_op.key.copy()
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
        if self.declared_id is not None:
            serialized["_declared_id"] = self.declared_id
        if self.key is not None:
            serialized["key"] = self.key.model_dump()
        return serialized

    @classmethod
    def from_json(cls, data: dict[str, Any], other_ops: dict[str, "Op"]) -> "Op":
        op = cls._from_json(data, other_ops)
        op.id = data["_id"]
        op.declared_id = data.get("_declared_id")
        max_iter = data["_max_iter"]
        op.set_looping(max_iter is not None, max_iter)
        key = data.get("key")
        op.key = None if key is None else record_key(key, op.declared_id or op.id)
        return op

    def __hash__(self) -> int:
        return hash(self.id)


class FunctionalOp(Op):
    pass
