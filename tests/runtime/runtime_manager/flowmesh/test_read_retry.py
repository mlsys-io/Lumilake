"""Read-only FlowMesh calls retry transient gateway errors (502/503/504)."""

from typing import Any

import pytest
from flowmesh.exceptions import APIError

import lumilake_server.runtime.runtime_manager.flowmesh as fm_mod
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _api_error(status_code: int) -> APIError:
    return APIError(
        "boom",
        status_code=status_code,
        method="GET",
        url="https://flowmesh.internal/tasks/t1",
    )


class _FakeTasks:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def retrieve(self, task_id: str) -> Any:
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _TaskInfo:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data


class _FakeFm:
    def __init__(self, tasks: _FakeTasks) -> None:
        self.tasks = tasks


@pytest.mark.asyncio
async def test_fetch_task_description_retries_transient_503(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 503 twice, then success: the read is retried and the result comes
    back, with the recorded delays [1.0, 2.0]."""
    tasks = _FakeTasks(
        [_api_error(503), _api_error(503), _TaskInfo({"graph_node_name": "node-a"})]
    )
    monkeypatch.setattr(
        FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm(tasks))
    )

    delays: list[float] = []

    async def _noop_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(fm_mod.asyncio, "sleep", _noop_sleep)

    result = await flowmesh_manager.fetch_task_description("t1")
    assert result == {"graph_node_name": "node-a"}
    assert tasks.calls == 3
    assert delays == [1.0, 2.0]


@pytest.mark.asyncio
async def test_fetch_task_description_404_raises_immediately(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-transient status (404) raises after a single call, no sleep."""
    tasks = _FakeTasks([_api_error(404)])
    monkeypatch.setattr(
        FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm(tasks))
    )

    delays: list[float] = []

    async def _noop_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(fm_mod.asyncio, "sleep", _noop_sleep)

    with pytest.raises(APIError) as excinfo:
        await flowmesh_manager.fetch_task_description("t1")
    assert excinfo.value.status_code == 404
    assert tasks.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_fetch_task_description_503_every_call_raises(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 503 on every call: after the retry budget (7 calls total) it raises
    an APIError with status 503."""
    tasks = _FakeTasks([_api_error(503)] * 7)
    monkeypatch.setattr(
        FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm(tasks))
    )

    async def _noop_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(fm_mod.asyncio, "sleep", _noop_sleep)

    with pytest.raises(APIError) as excinfo:
        await flowmesh_manager.fetch_task_description("t1")
    assert excinfo.value.status_code == 503
    assert tasks.calls == 7
