"""FlowMesh workflow submits retry transient gateway errors without
double-submitting.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from flowmesh.exceptions import APIError
from flowmesh.models.workflows import (
    Workflow,
    WorkflowSubmitResponse,
    WorkflowSubmitTaskEntry,
)

import lumilake_server.runtime.runtime_manager.flowmesh as fm_mod
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


def _api_error(status_code: int) -> APIError:
    return APIError(
        "boom",
        status_code=status_code,
        method="POST",
        url="https://flowmesh.internal/api/v1/workflows",
    )


def _submit_response(workflow_id: str, task_ids: list[str]) -> WorkflowSubmitResponse:
    return WorkflowSubmitResponse(
        ok=True,
        workflow_id=workflow_id,
        count=len(task_ids),
        tasks=[WorkflowSubmitTaskEntry(task_id=t) for t in task_ids],
    )


def _workflow(workflow_id: str, task_ids: list[str], submitted_at: str) -> Workflow:
    return Workflow(
        workflow_id=workflow_id,
        task_ids=task_ids,
        submitted_at=submitted_at,
        updated_at=submitted_at,
        status="PENDING",
        dispatched_tasks=[],
        completed_tasks=[],
        failed_tasks=[],
        cancelled_tasks=[],
    )


class _TaskInfo:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data


class _FakeTasks:
    def __init__(self, descriptions: dict[str, dict[str, Any]]) -> None:
        self._descriptions = descriptions

    async def retrieve(self, task_id: str) -> Any:
        return _TaskInfo(self._descriptions[task_id])


class _FakeWorkflows:
    def __init__(self, submit_responses: list[Any], workflows: list[Workflow]) -> None:
        self._submit_responses = list(submit_responses)
        self.workflows = workflows
        self.submit_calls = 0
        self.list_calls = 0

    async def submit(self, task_yaml: str) -> Any:
        self.submit_calls += 1
        response = self._submit_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def list(self) -> list[Workflow]:
        self.list_calls += 1
        return self.workflows


class _FakeFm:
    def __init__(self, workflows: _FakeWorkflows, tasks: _FakeTasks) -> None:
        self.workflows = workflows
        self.tasks = tasks


def _description(submit_id: str) -> dict[str, Any]:
    return {
        "task": {
            "metadata": {"annotations": {"custom": {"lumilake_submit_id": submit_id}}}
        }
    }


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
    workflows: _FakeWorkflows,
    tasks: _FakeTasks,
) -> None:
    monkeypatch.setattr(
        FlowmeshRuntimeManager, "fm", property(lambda self: _FakeFm(workflows, tasks))
    )


def _record_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def _noop_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(fm_mod.asyncio, "sleep", _noop_sleep)
    return delays


@pytest.mark.asyncio
async def test_submit_502_then_success_no_adoption(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 502, then the list has no matching workflow: submit is retried and the
    second response is returned, with the recorded delay [1.0]."""
    workflows = _FakeWorkflows(
        [_api_error(502), _submit_response("wf-2", ["t2"])],
        [],
    )
    tasks = _FakeTasks({})
    _install_fake(monkeypatch, workflows, tasks)
    delays = _record_sleep(monkeypatch)

    result = await flowmesh_manager._submit_workflow("yaml", "sid-123")
    assert result == ("wf-2", ["t2"])
    assert workflows.submit_calls == 2
    assert workflows.list_calls == 1
    assert delays == [1.0]


@pytest.mark.asyncio
async def test_submit_502_adopts_earlier_workflow(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 502, then the list holds a workflow carrying our submit_id: submit is
    called once and the adopted workflow's id and task_ids are returned. An older
    workflow (even with our submit_id) and one with a different submit_id are not
    adopted."""
    now = datetime.now(UTC)
    older = (now - timedelta(hours=1)).isoformat()
    now_iso = now.isoformat()
    workflows = _FakeWorkflows(
        [_api_error(502)],
        [
            _workflow("wf-old", ["t-old"], older),
            _workflow("wf-other", ["t-other"], now_iso),
            _workflow("wf-mine", ["t-mine"], now_iso),
        ],
    )
    tasks = _FakeTasks(
        {
            "t-old": _description("sid-123"),
            "t-other": {"task": {"metadata": {"annotations": {"custom": None}}}},
            "t-mine": _description("sid-123"),
        }
    )
    _install_fake(monkeypatch, workflows, tasks)
    delays = _record_sleep(monkeypatch)

    result = await flowmesh_manager._submit_workflow("yaml", "sid-123")
    assert result == ("wf-mine", ["t-mine"])
    assert workflows.submit_calls == 1
    assert workflows.list_calls == 1
    assert delays == [1.0]


@pytest.mark.asyncio
async def test_submit_400_raises_immediately(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-transient status (400) raises after a single submit, no sleep, no list."""
    workflows = _FakeWorkflows([_api_error(400)], [])
    tasks = _FakeTasks({})
    _install_fake(monkeypatch, workflows, tasks)
    delays = _record_sleep(monkeypatch)

    with pytest.raises(APIError) as excinfo:
        await flowmesh_manager._submit_workflow("yaml", "sid-123")
    assert excinfo.value.status_code == 400
    assert workflows.submit_calls == 1
    assert workflows.list_calls == 0
    assert delays == []
