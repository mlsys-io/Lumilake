"""`/jobs` must refuse what `/jobs/preview` refuses.

Observed 2026-09-19 against production. The same workflow -- an OutputOp whose
`ref` pointed at a FormatOp -- got two different answers:

    POST /jobs/preview  -> 500 "data profile preflight failed: OutputOp
                            'greeting' input must be an LLMOp or
                            DataRetrievalOp (got FormatOp)"
    POST /jobs          -> 200 {"job_id": "req-NUQRwh7kDDMEnG8ejttJLb",
                                "status": "pending"}

The submitted job then failed asynchronously with that identical message. So
the validation existed and simply was not on the submit path: it lives inside
`RuntimeGraphBuilder.build`, which either route reached only through
`_any_graph_requires_gpu`, and that is gated on `hardware.gpu == 0`. A request
with no hardware override -- the common case, and the one that bit -- skipped
it entirely.

These cover the boundary conversion that `_validate_runtime_graphs` adds. The
structural rule itself belongs to the builder and is exercised by the runtime
tests.
"""

import logging
from typing import Any

import pytest
from fastapi import HTTPException

from lumilake_server.routes.jobs import _validate_runtime_graphs

_BUILD_ERROR = (
    "OutputOp 'greeting' input must be an LLMOp or DataRetrievalOp (got FormatOp)"
)


class _RejectingBuilder:
    """Stands in for RuntimeGraphBuilder rejecting an unrunnable graph."""

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
        if builder is not None:
            self._runtime_builder = builder


def test_unrunnable_graph_becomes_422_not_an_accepted_job() -> None:
    builder = _RejectingBuilder()
    with pytest.raises(HTTPException) as exc_info:
        _validate_runtime_graphs(_Server(builder), {"g": object()})
    assert exc_info.value.status_code == 422
    # The caller must be told WHICH op is wrong, not just that something is.
    assert _BUILD_ERROR in exc_info.value.detail
    assert builder.calls == ["g"]


def test_runnable_graph_passes_through() -> None:
    builder = _AcceptingBuilder()
    _validate_runtime_graphs(_Server(builder), {"a": object(), "b": object()})
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
        _validate_runtime_graphs(
            _Server(builder), {"first": object(), "second": object()}
        )
    assert exc_info.value.status_code == 422
    assert builder.calls == ["first", "second"]


@pytest.mark.parametrize("raised", [ValueError, KeyError, AssertionError])
def test_the_builder_s_error_kinds_all_become_422(raised: type[Exception]) -> None:
    class _Raises:
        def build(self, compiled: Any, node_prefix: str | None = None) -> Any:
            raise raised("nope")

    with pytest.raises(HTTPException) as exc_info:
        _validate_runtime_graphs(_Server(_Raises()), {"g": object()})
    assert exc_info.value.status_code == 422


def test_a_server_without_a_builder_is_left_alone() -> None:
    """Must not turn a missing internal into a failed submit.

    The existing route tests drive a fake server that has no
    `_runtime_builder`; before this guard the new call raised AttributeError
    and would have broken them -- i.e. it would have converted "this test
    double is minimal" into "your job is rejected".
    """
    _validate_runtime_graphs(_Server(None), {"g": object()})


def test_no_graphs_is_not_an_error() -> None:
    builder = _AcceptingBuilder()
    _validate_runtime_graphs(_Server(builder), {})
    assert builder.calls == []


def test_logger_is_not_required() -> None:
    # Guards against the helper growing a logging dependency that route
    # callers would then have to thread through.
    logging.getLogger("unused")
    _validate_runtime_graphs(_Server(_AcceptingBuilder()), {"g": object()})
