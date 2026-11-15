from typing import Any, cast

import pytest
from fastapi import HTTPException

from lumilake_server.graphs import CompiledGraph
from lumilake_server.routes.jobs import (
    _validate_list_lambda_slice_span,
    _validate_runtime_graphs,
)
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.runtime.server import LumilakeServer

_BUILD_ERROR = (
    "OutputOp 'greeting' input must be an LLMOp or DataRetrievalOp (got FormatOp)"
)


class _RejectingBuilder:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def build(self, compiled: Any, node_prefix: str | None = None) -> Any:
        self.calls.append(node_prefix)
        raise ValueError(_BUILD_ERROR)


class _AcceptingBuilder:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def build(self, compiled: Any, node_prefix: str | None = None) -> Any:
        self.calls.append(node_prefix)
        return object()


class _Server:
    def __init__(self, builder: Any) -> None:
        self._runtime_builder = builder


_GRAPH = cast(CompiledGraph, object())


def _server(builder: Any) -> LumilakeServer:
    return cast(LumilakeServer, _Server(builder))


def test_unrunnable_graph_becomes_422_not_an_accepted_job() -> None:
    builder = _RejectingBuilder()
    with pytest.raises(HTTPException) as exc_info:
        _validate_runtime_graphs(_server(builder), {"g": _GRAPH})
    assert exc_info.value.status_code == 422
    # The caller must be told WHICH op is wrong, not just that something is.
    assert _BUILD_ERROR in exc_info.value.detail
    assert builder.calls == ["g"]


def test_runnable_graph_passes_through() -> None:
    builder = _AcceptingBuilder()
    _validate_runtime_graphs(_server(builder), {"a": _GRAPH, "b": _GRAPH})
    assert builder.calls == ["a", "b"]


def test_every_graph_is_checked_not_just_the_first() -> None:
    class _FailsOnSecond:
        def __init__(self) -> None:
            self.calls: list[str | None] = []

        def build(self, compiled: Any, node_prefix: str | None = None) -> Any:
            self.calls.append(node_prefix)
            if node_prefix == "second":
                raise ValueError(_BUILD_ERROR)
            return object()

    builder = _FailsOnSecond()
    with pytest.raises(HTTPException) as exc_info:
        _validate_runtime_graphs(_server(builder), {"first": _GRAPH, "second": _GRAPH})
    assert exc_info.value.status_code == 422
    assert builder.calls == ["first", "second"]


@pytest.mark.parametrize("raised", [ValueError, KeyError, AssertionError])
def test_the_builder_s_error_kinds_all_become_422(raised: type[Exception]) -> None:
    class _Raises:
        def build(self, compiled: Any, node_prefix: str | None = None) -> Any:
            raise raised("nope")

    with pytest.raises(HTTPException) as exc_info:
        _validate_runtime_graphs(_server(_Raises()), {"g": _GRAPH})
    assert exc_info.value.status_code == 422


def test_no_graphs_is_not_an_error() -> None:
    builder = _AcceptingBuilder()
    _validate_runtime_graphs(_server(builder), {})
    assert builder.calls == []


def _list_lambda_graph(output_name: str) -> RuntimeGraph:
    node_id = "g__lambda"
    op = RuntimeOp(
        node_id=node_id,
        task_type="python",
        backend="python",
        model="",
        data_spec={"mode": "list"},
        model_spec={},
        inference_spec={},
    )
    return RuntimeGraph(
        nodes={node_id: op},
        node_order=[node_id],
        output_node_map={node_id: output_name},
    )


def _row_graph(output_name: str) -> RuntimeGraph:
    node_id = "g__row"
    op = RuntimeOp(
        node_id=node_id,
        task_type="python",
        backend="python",
        model="",
        data_spec={"mode": "row"},
        model_spec={},
        inference_spec={},
    )
    return RuntimeGraph(
        nodes={node_id: op},
        node_order=[node_id],
        output_node_map={node_id: output_name},
    )


def _slice(graph_name: str, slice_index: int, slice_length: int) -> WorkflowSliceMeta:
    return WorkflowSliceMeta(
        public_graph_name="g",
        slice_index=slice_index,
        slice_start=slice_index * slice_length,
        slice_length=slice_length,
        total_length=slice_length * 2,
        template_hash="h",
    )


def test_list_lambda_output_spanning_multiple_slices_is_rejected() -> None:
    """A list-mode Lambda runs once over the whole input lists of one run, so
    its single whole-list result cannot be partitioned per slice. A submission
    that would split it across more than one slice must be rejected at submit
    time with a message naming the op and the fix."""
    with pytest.raises(HTTPException) as exc_info:
        _validate_list_lambda_slice_span(
            {"g": _list_lambda_graph("out")},
            {"g": [_slice("g__slice_1", 0, 2), _slice("g__slice_2", 1, 2)]},
        )
    assert exc_info.value.status_code == 422
    assert "out" in exc_info.value.detail
    assert "2" in exc_info.value.detail


def test_list_lambda_output_single_slice_is_accepted() -> None:
    _validate_list_lambda_slice_span(
        {"g": _list_lambda_graph("out")},
        {"g": [_slice("g", 0, 2)]},
    )


def test_row_mode_output_spanning_multiple_slices_is_accepted() -> None:
    """A row-mode op emits one output per input row, so it can be split across
    slices; only list-cardinality outputs are constrained."""
    _validate_list_lambda_slice_span(
        {"g": _row_graph("out")},
        {"g": [_slice("g__slice_1", 0, 2), _slice("g__slice_2", 1, 2)]},
    )


def test_list_lambda_output_not_an_output_is_accepted() -> None:
    """A list-mode Lambda feeding a row-mode python step is not a
    list-cardinality output node (the row-mode step is a barrier), so
    multi-slice is fine."""
    lambda_id = "g__lambda"
    downstream_id = "g__downstream"
    op = RuntimeOp(
        node_id=lambda_id,
        task_type="python",
        backend="python",
        model="",
        data_spec={"mode": "list"},
        model_spec={},
        inference_spec={},
    )
    downstream = RuntimeOp(
        node_id=downstream_id,
        task_type="python",
        backend="python",
        model="",
        data_spec={"mode": "row"},
        model_spec={},
        inference_spec={},
        dependencies=(lambda_id,),
    )
    graph = RuntimeGraph(
        nodes={lambda_id: op, downstream_id: downstream},
        node_order=[lambda_id, downstream_id],
        output_node_map={downstream_id: "out"},
    )
    _validate_list_lambda_slice_span(
        {"g": graph},
        {"g": [_slice("g__slice_1", 0, 2), _slice("g__slice_2", 1, 2)]},
    )
