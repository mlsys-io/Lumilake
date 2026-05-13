from collections import defaultdict
from collections.abc import Iterable
from typing import Any, TypeVar, overload

from lumilake.ops import InputOp, Op, OutputOp
from lumilake.utils.graph import topological_sort

OpType = TypeVar("OpType", bound=Op)


class CompiledGraph:
    def __init__(self, graph: "Graph", inputs: dict[str, list[str]]) -> None:
        self.graph = graph
        self.inputs = inputs

        self._coalesce_rewrite_hits: dict[str, int] | None = None
        self._coalesce_rewrite_skipped: bool | None = None

    def copy(self, new_ids: bool = True) -> "CompiledGraph":
        """Creates a deep copy of the compiled graph"""
        copied = CompiledGraph(self.graph.copy(new_ids), self.inputs.copy())
        copied._coalesce_rewrite_hits = (
            dict(self._coalesce_rewrite_hits)
            if isinstance(self._coalesce_rewrite_hits, dict)
            else None
        )
        copied._coalesce_rewrite_skipped = self._coalesce_rewrite_skipped
        return copied

    def serialize(self) -> dict[str, Any]:
        return {
            "graph": self.graph.serialize(),
            "inputs": self.inputs,
        }

    @overload
    def iter_ops(self, op_type: None = None) -> Iterable[Op]: ...

    @overload
    def iter_ops(self, op_type: type[OpType]) -> Iterable[OpType]: ...

    def iter_ops(self, op_type: type[Op] | None = None) -> Iterable[Op]:
        return self.graph.iter_ops(op_type)

    def dependencies(self) -> dict[Op, set[Op]]:
        return self.graph.dependencies()


class Graph:
    def __init__(self, graph: dict[str, Op]) -> None:
        self._graph = graph

        # Build input and output op mappings.
        self._input_ops: dict[str, InputOp] = {}
        self._output_ops: dict[str, OutputOp] = {}
        for op in self._graph.values():
            if isinstance(op, InputOp):
                if op.name in self._input_ops:
                    raise ValueError(f"Duplicate input name: {op.name}")
                self._input_ops[op.name] = op
            elif isinstance(op, OutputOp):
                if op.name in self._output_ops:
                    raise ValueError(f"Duplicate output name: {op.name}")
                self._output_ops[op.name] = op

        # Sort the graph in topological order.
        self.topological_sort()

    def as_dict(self) -> dict[str, Op]:
        """Mapping from op ID to op"""
        return self._graph

    @property
    def input_ops(self) -> dict[str, InputOp]:
        """Mapping from input name to input op"""
        return self._input_ops

    @property
    def output_ops(self) -> dict[str, OutputOp]:
        """Mapping from output name to output op"""
        return self._output_ops

    @overload
    def iter_ops(self, op_type: None = None) -> Iterable[Op]: ...

    @overload
    def iter_ops(self, op_type: type[OpType]) -> Iterable[OpType]: ...

    def iter_ops(self, op_type: type[Op] | None = None) -> Iterable[Op]:
        """Iterates over all ops in the graph in topological order"""
        if op_type is None:
            return self._graph.values()
        return (op for op in self._graph.values() if isinstance(op, op_type))

    @property
    def node_count(self) -> int:
        return len(self._graph)

    def compile(self, *_, **inputs: list[str]) -> CompiledGraph:
        # Check input length
        if len(inputs) != len(self._input_ops):
            raise ValueError(
                f"Expected {len(self._input_ops)} inputs but got {len(inputs)}"
            )

        # Check all inputs are present
        for name in self._input_ops:
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")

        # Check all inputs are lists of strings
        for name, value in inputs.items():
            if not isinstance(value, list):
                raise ValueError(f"Input {name} is not a list")
            for v in value:
                if not isinstance(v, str):
                    raise ValueError(f"Input {name} contains a non-string value")

        return CompiledGraph(self, inputs)

    def serialize(self) -> dict[str, Any]:
        serialized_graph = {op_id: op.serialize() for op_id, op in self._graph.items()}
        return serialized_graph

    @classmethod
    def from_ops(cls, ops: list[OutputOp]) -> "Graph":
        graph: dict[str, Op] = {}
        for op in ops:
            op.traverse_graph(graph)
        return cls(graph)

    @classmethod
    def from_json(cls, json_graph: dict[str, Any]) -> "Graph":
        graph: dict[str, Op] = {}
        for op_id, op_data in json_graph.items():
            op_cls = Op.registry.get_class(op_data["_op"])
            op = op_cls.from_json(op_data, graph)
            graph[op_id] = op
        return cls(graph)

    def copy(self, new_ids: bool = True) -> "Graph":
        node_map = {op: op.copy(new_ids) for op in self.iter_ops()}

        # Update op dependencies
        output_ops: list[OutputOp] = []
        for new_op in node_map.values():
            new_op.inputs = [node_map[inp] for inp in new_op.inputs]
            if isinstance(new_op, OutputOp):
                output_ops.append(new_op)

        return Graph.from_ops(output_ops)

    def dependencies(self) -> dict[Op, set[Op]]:
        """Returns a dependency mapping from op to a set of dependent ops"""
        dependencies: defaultdict[Op, set[Op]] = defaultdict(set)
        for op in self._graph.values():
            for dep in op.inputs:
                dependencies[dep].add(op)
        return dict(dependencies)

    def topological_sort(self) -> None:
        dependencies: defaultdict[Op, set[Op]] = defaultdict(set)
        for op in self._graph.values():
            for dep in op.inputs:
                dependencies[dep].add(op)
        sorted_ops = topological_sort(dict(dependencies))
        self._graph = {op.id: op for op in sorted_ops}
