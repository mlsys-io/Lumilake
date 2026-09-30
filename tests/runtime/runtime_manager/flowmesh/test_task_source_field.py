"""A FlowMesh 0.1.10 server sends a task's source as ``source``, not ``raw_yaml``.

flowmesh-sdk 0.1.9 requires ``raw_yaml``, so every task read failed validation
and every request died right after submit with "Failed to fetch FlowMesh task
description" (home, 2026-09-30, on v0.1.10-rc.2). Both shapes must read.
"""

from typing import Any

import pytest
from flowmesh.exceptions import APIError
from flowmesh.models.tasks import TaskInfo

from lumilake_server.runtime.runtime_manager import flowmesh as fm_module
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager

BASE = {
    "task_id": "tsk-1",
    "workflow_id": "wfl-1",
    "owner_id": "u-1",
    "org_id": "",
    "supplier_id": "lumid",
    "task": {},
    "status": "DISPATCHED",
    "submitted_at": "2026-09-30T00:00:00Z",
    "submitted_ts": 0.0,
    "usages": [],
    "attempts": 0,
    "max_attempts": 3,
    "load": 1,
    "graph_node_name": "node-a",
    "depends_on": [],
    "pending_dependencies": [],
    "dependents": [],
    "completed": False,
    "failed": False,
}


class _Tasks:
    """The SDK's AsyncTasks: retrieve() validates the body into TaskInfo."""

    def __init__(self, client: "_Client") -> None:
        self._client = client

    async def retrieve(self, task_id: str) -> TaskInfo:
        return TaskInfo.model_validate(
            await self._client._request("GET", f"/tasks/{task_id}")
        )


class _Client:
    def __init__(self, data: Any = None, exc: Exception | None = None) -> None:
        self.data, self.exc = data, exc
        self.calls: list[tuple[str, str]] = []

    async def _request(self, method: str, path: str) -> Any:
        self.calls.append((method, path))
        if self.exc:
            raise self.exc
        return self.data


def _manager(
    client: _Client, monkeypatch: pytest.MonkeyPatch
) -> FlowmeshRuntimeManager:
    fm = type("FM", (), {})()
    fm.tasks = _Tasks(client)
    monkeypatch.setattr(fm_module, "flowmesh_for_context", lambda: fm)
    return FlowmeshRuntimeManager()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["source", "raw_yaml"])
async def test_task_description_reads_both_server_shapes(
    key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _Client({**BASE, key: "apiVersion: mloc/v1"})
    mgr = _manager(client, monkeypatch)
    desc = await mgr.fetch_task_description("tsk-1")
    assert desc["graph_node_name"] == "node-a"
    assert desc["raw_yaml"] == "apiVersion: mloc/v1"
    assert await mgr.fetch_task_status("tsk-1") == "DISPATCHED"
    assert client.calls[0] == ("GET", "/tasks/tsk-1")


@pytest.mark.asyncio
async def test_task_read_errors_still_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    err = APIError("denied", status_code=403, method="GET", url="/tasks/tsk-1")
    mgr = _manager(_Client(exc=err), monkeypatch)
    with pytest.raises(APIError):
        await mgr.fetch_task_description("tsk-1")
