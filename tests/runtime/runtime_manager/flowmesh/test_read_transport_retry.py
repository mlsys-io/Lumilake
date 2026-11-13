"""FlowMesh reads retry a dropped connection; a submit never does."""

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

_DROPPED = [
    httpx.ReadError("connection dropped"),
    httpx.RemoteProtocolError("server disconnected"),
    httpx.ConnectError("refused"),
    httpx.ConnectTimeout("connect timed out"),
    FlowMeshConnectionError("Failed to connect to http://flowmesh"),
]
_NOT_RETRIED = [
    httpx.ReadTimeout("timed out"),
    httpx.PoolTimeout("no free connection"),
    ValueError("bad payload"),
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
@pytest.mark.parametrize("error", _DROPPED, ids=lambda e: type(e).__name__)
async def test_a_read_retries_a_dropped_connection(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays = _record_sleep(monkeypatch)
    call = _FlakyCall([error])

    assert await fm_mod._retry_transient_read(lambda: call(), "task t-1") == "ok"
    assert call.calls == 2
    assert delays == [_FLOWMESH_RETRY_DELAYS_S[0]]


@pytest.mark.asyncio
async def test_a_read_retries_a_gateway_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = _record_sleep(monkeypatch)
    call = _FlakyCall(
        [APIError("busy", status_code=503, method="GET", url="https://fm/x")]
    )

    assert await fm_mod._retry_transient_read(lambda: call(), "task t-1") == "ok"
    assert call.calls == 2
    assert delays == [_FLOWMESH_RETRY_DELAYS_S[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", _NOT_RETRIED, ids=lambda e: type(e).__name__)
async def test_a_read_does_not_retry_a_timeout_or_an_unrelated_error(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays = _record_sleep(monkeypatch)
    call = _FlakyCall([error])

    with pytest.raises(type(error)):
        await fm_mod._retry_transient_read(lambda: call(), "task t-1")
    assert call.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_a_read_gives_up_after_the_gateway_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays = _record_sleep(monkeypatch)
    call = _FlakyCall([httpx.ReadError("dropped")] * 20)

    with pytest.raises(httpx.ReadError):
        await fm_mod._retry_transient_read(lambda: call(), "task t-1")
    assert call.calls == len(_FLOWMESH_RETRY_DELAYS_S) + 1
    assert delays == list(_FLOWMESH_RETRY_DELAYS_S)


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


@pytest.mark.asyncio
async def test_a_task_status_read_survives_a_dropped_connection(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_sleep(monkeypatch)
    status_call = _FlakyCall([httpx.ReadError("dropped")], value="DONE")

    class _Tasks:
        status = staticmethod(status_call)

    class _FakeFm:
        tasks = _Tasks()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    assert await flowmesh_manager.fetch_task_status("t-1") == "DONE"
    assert status_call.calls == 2


@pytest.mark.asyncio
async def test_a_result_fetch_survives_a_dropped_connection(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    _record_sleep(monkeypatch)
    retrieve_call = _FlakyCall([httpx.ReadError("dropped")], value={"items": []})

    class _Results:
        retrieve = staticmethod(retrieve_call)

    class _FakeFm:
        results = _Results()

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))
    saved: dict[str, Any] = {}
    monkeypatch.setattr(
        flowmesh_manager,
        "_save_json_artifact",
        lambda _request_info, name, _data: saved.setdefault("name", name),
    )

    uri = await flowmesh_manager._archive_task_response(
        request_info=RequestInfo(
            request_id="req-1", runtime_graphs={}, data_profile_graphs={}
        ),
        tid="t-1",
        node_id="n-1",
    )

    assert retrieve_call.calls == 2
    assert uri == saved["name"]


@pytest.mark.asyncio
async def test_the_scheduler_worker_reads_survive_a_dropped_connection(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker reads sit on the dispatch path, so one dropped connection must
    not fail a dispatch."""
    _record_sleep(monkeypatch)
    listing = _FlakyCall([httpx.ReadError("dropped")], value=[_worker("wkr-1")])
    retrieve = _FlakyCall([httpx.ReadError("dropped")], value=_worker("wkr-1"))

    class _Workers:
        list = staticmethod(lambda **_: listing())
        retrieve = staticmethod(lambda *a, **_: retrieve(*a, **_))

    class _FakeFm:
        workers = _Workers()

    monkeypatch.setattr(fm_mod, "flowmesh_for_server", lambda: _FakeFm())

    assert await flowmesh_manager.get_workers() == ["wkr-1"]
    assert await flowmesh_manager.get_all_workers() == ["wkr-1"]
    assert await flowmesh_manager.get_dispatchable_workers() == ["wkr-1"]
    assert "supported_task_types" in await flowmesh_manager.get_worker_profile("wkr-1")
    assert listing.calls == 4
    assert retrieve.calls == 2


@pytest.mark.asyncio
async def test_a_submit_is_not_retried_after_a_dropped_connection(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether the server created the workflow is unknown after a transport
    error, so a submit never retries on one."""
    delays = _record_sleep(monkeypatch)
    submit_call = _FlakyCall([httpx.ReadError("dropped")])

    class _FakeFm:
        class workflows:
            submit = staticmethod(submit_call)

            @staticmethod
            async def list() -> list[Any]:
                raise AssertionError("a transport error must not trigger adoption")

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

    class _FakeFm:
        class workflows:
            submit = staticmethod(submit_call)

            @staticmethod
            async def list() -> list[Any]:
                return []

    monkeypatch.setattr(FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm()))

    assert await flowmesh_manager._submit_workflow("yaml", "sid-123") == (
        "wf-1",
        ["t1"],
    )
    assert submit_call.calls == 2
    assert delays == [_FLOWMESH_RETRY_DELAYS_S[0]]
