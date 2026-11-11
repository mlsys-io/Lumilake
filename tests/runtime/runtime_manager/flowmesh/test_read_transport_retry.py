"""FlowMesh reads retry a dropped connection; a submit never does."""

from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
from flowmesh.exceptions import APIError, FlowMeshConnectionError
from flowmesh.models.workers import WorkerInfo

import lumilake_server.runtime.runtime_manager.flowmesh as fm_mod
from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_manager.flowmesh import (
    _FLOWMESH_RETRY_DELAYS_S,
    FlowmeshRuntimeManager,
)

_TRANSPORT_ERRORS: list[Callable[[], Exception]] = [
    lambda: httpx.ReadError("connection dropped"),
    lambda: httpx.RemoteProtocolError("server disconnected"),
    lambda: httpx.ConnectError("refused"),
    lambda: httpx.WriteError("broken pipe"),
    lambda: httpx.CloseError("close failed"),
    lambda: FlowMeshConnectionError("Failed to connect to http://flowmesh"),
]

_NOT_RETRIED_ERRORS: list[Callable[[], Exception]] = [
    lambda: httpx.ReadTimeout("timed out"),
    lambda: httpx.WriteTimeout("timed out"),
    lambda: httpx.ConnectTimeout("connect timed out"),
    lambda: httpx.PoolTimeout("no free connection"),
    lambda: httpx.UnsupportedProtocol("no scheme"),
    lambda: httpx.ProxyError("bad proxy"),
]


def _record_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def _noop_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(fm_mod.asyncio, "sleep", _noop_sleep)
    return delays


class _FlakyCall:
    """Raises each queued error in turn, then returns ``value``."""

    def __init__(self, errors: list[Exception], value: Any = "ok") -> None:
        self._errors = list(errors)
        self._value = value
        self.calls = 0

    async def __call__(self, *_: Any, **__: Any) -> Any:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return self._value


@pytest.mark.asyncio
@pytest.mark.parametrize("make_error", _TRANSPORT_ERRORS)
async def test_a_read_retries_a_dropped_connection(
    make_error: Callable[[], Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    delays = _record_sleep(monkeypatch)
    call = _FlakyCall([make_error()])

    result = await fm_mod._retry_transient_read(lambda: call(), "task t-1")

    assert result == "ok"
    assert call.calls == 2
    assert delays == [_FLOWMESH_RETRY_DELAYS_S[0]]


@pytest.mark.asyncio
async def test_a_read_gives_up_after_the_same_attempts_as_a_gateway_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = _record_sleep(monkeypatch)
    call = _FlakyCall([httpx.ReadError("dropped")] * 20)

    with pytest.raises(httpx.ReadError):
        await fm_mod._retry_transient_read(lambda: call(), "task t-1")

    assert call.calls == len(_FLOWMESH_RETRY_DELAYS_S) + 1
    assert delays == list(_FLOWMESH_RETRY_DELAYS_S)


@pytest.mark.asyncio
@pytest.mark.parametrize("make_error", _NOT_RETRIED_ERRORS)
async def test_a_read_does_not_retry_a_timeout_or_a_configuration_error(
    make_error: Callable[[], Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every timeout would wait out the full HTTP timeout again on each attempt,
    since the shared client gives connect and pool the same value as a read, and
    a configuration error never clears."""
    delays = _record_sleep(monkeypatch)
    error = make_error()
    call = _FlakyCall([error])

    with pytest.raises(type(error)):
        await fm_mod._retry_transient_read(lambda: call(), "task t-1")

    assert call.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_a_read_does_not_retry_an_unrelated_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = _record_sleep(monkeypatch)
    call = _FlakyCall([ValueError("bad payload")])

    with pytest.raises(ValueError):
        await fm_mod._retry_transient_read(lambda: call(), "task t-1")

    assert call.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_a_result_fetch_survives_a_dropped_connection(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive fetch of a task's response: a ReadError on the first try is
    retried and the batch carries on."""
    delays = _record_sleep(monkeypatch)
    retrieve_call = _FlakyCall([httpx.ReadError("dropped")], value={"items": []})

    class _FakeResults:
        retrieve = staticmethod(retrieve_call)

    class _FakeFm:
        results = _FakeResults()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))
    saved: dict[str, Any] = {}
    monkeypatch.setattr(
        flowmesh_manager,
        "_save_json_artifact",
        lambda request_info, name, data: saved.setdefault("name", name),
    )

    uri = await flowmesh_manager._archive_task_response(
        request_info=RequestInfo(
            request_id="req-1", runtime_graphs={}, data_profile_graphs={}
        ),
        tid="t-1",
        node_id="n-1",
    )

    assert retrieve_call.calls == 2
    assert delays == [_FLOWMESH_RETRY_DELAYS_S[0]]
    assert uri == saved["name"]


@pytest.mark.asyncio
async def test_a_task_status_read_survives_a_dropped_connection(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_sleep(monkeypatch)
    status_call = _FlakyCall([httpx.ReadError("dropped")], value="DONE")

    class _FakeTasks:
        status = staticmethod(status_call)

    class _FakeFm:
        tasks = _FakeTasks()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    assert await flowmesh_manager.fetch_task_status("t-1") == "DONE"
    assert status_call.calls == 2


@pytest.mark.asyncio
async def test_a_submit_is_not_retried_after_a_dropped_connection(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether the server created the workflow is unknown after a transport
    error, so a submit never retries on one."""
    delays = _record_sleep(monkeypatch)
    submit_call = _FlakyCall([httpx.ReadError("dropped")])

    class _FakeWorkflows:
        submit = staticmethod(submit_call)

        @staticmethod
        async def list() -> list[Any]:
            raise AssertionError("a transport error must not trigger adoption")

    class _FakeFm:
        workflows = _FakeWorkflows()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    with pytest.raises(httpx.ReadError):
        await flowmesh_manager._submit_workflow("yaml", "sid-123")

    assert submit_call.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_a_submit_still_retries_a_gateway_error(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays = _record_sleep(monkeypatch)
    gateway = APIError(
        "boom", status_code=502, method="POST", url="https://flowmesh.internal/x"
    )

    class _Response:
        workflow_id = "wf-1"
        tasks = [type("T", (), {"task_id": "t1"})()]

    submit_call = _FlakyCall([gateway], value=_Response())

    class _FakeWorkflows:
        submit = staticmethod(submit_call)

        @staticmethod
        async def list() -> list[Any]:
            return []

    class _FakeFm:
        workflows = _FakeWorkflows()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    assert await flowmesh_manager._submit_workflow("yaml", "sid-123") == (
        "wf-1",
        ["t1"],
    )
    assert submit_call.calls == 2
    assert delays == [_FLOWMESH_RETRY_DELAYS_S[0]]


def _worker(worker_id: str) -> WorkerInfo:
    return WorkerInfo(
        id=worker_id,
        namespace="ns",
        cluster="cluster-0",
        node_id="node-0",
        node_alias="node-0",
        status="IDLE",
        stale=False,
        cordoned=False,
    )


class _FakeWorkers:
    def __init__(self, list_call: _FlakyCall, retrieve_call: _FlakyCall) -> None:
        self.list = list_call
        self.retrieve = retrieve_call


class _FakeWorkersFm:
    def __init__(self, workers: _FakeWorkers) -> None:
        self.workers = workers


def _install_workers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    list_call: _FlakyCall | None = None,
    retrieve_call: _FlakyCall | None = None,
) -> None:
    fake = _FakeWorkersFm(
        _FakeWorkers(
            list_call or _FlakyCall([], value=[]),
            retrieve_call or _FlakyCall([], value=None),
        )
    )
    monkeypatch.setattr(fm_mod, "flowmesh_for_server", lambda: fake)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read",
    [
        FlowmeshRuntimeManager.get_workers,
        FlowmeshRuntimeManager.get_all_workers,
        FlowmeshRuntimeManager.get_dispatchable_workers,
    ],
)
async def test_a_worker_listing_survives_a_dropped_connection(
    read: Callable[[FlowmeshRuntimeManager], Awaitable[list[str]]],
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler's worker reads sit on the dispatch path, so one dropped
    connection must not fail a dispatch."""
    delays = _record_sleep(monkeypatch)
    listing = _FlakyCall([httpx.ReadError("dropped")], value=[_worker("wkr-1")])
    _install_workers(monkeypatch, list_call=listing)

    assert await read(flowmesh_manager) == ["wkr-1"]

    assert listing.calls == 2
    assert delays == [_FLOWMESH_RETRY_DELAYS_S[0]]


@pytest.mark.asyncio
async def test_a_worker_profile_read_survives_a_dropped_connection(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_sleep(monkeypatch)
    retrieve = _FlakyCall([httpx.ReadError("dropped")], value=_worker("wkr-1"))
    _install_workers(monkeypatch, retrieve_call=retrieve)

    profile = await flowmesh_manager.get_worker_profile("wkr-1")

    assert "supported_task_types" in profile
    assert retrieve.calls == 2
