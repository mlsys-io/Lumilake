from typing import Any, cast

import pytest
from fastapi import HTTPException

from lumilake_server.graphs import CompiledGraph
from lumilake_server.routes.jobs import _validate_runtime_graphs
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
