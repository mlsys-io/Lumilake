import types
from typing import Any

import pytest

from lumilake_server.runtime.request import RequestInfo
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager


class _FakeResults:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def retrieve(self, task_id: str) -> dict[str, Any]:
        return self._payload


class _FakeFlowMeshClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.results = _FakeResults(payload)


@pytest.mark.asyncio
async def test_archive_task_response_redacts_credential_under_unexpected_key(
    flowmesh_manager: FlowmeshRuntimeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FlowMesh task response is untrusted content archived verbatim as a
    job artifact and reachable through the artifact API. A credential the
    remote endpoint reflects back under a key that isn't one of the
    recognized sensitive keys must still be scrubbed before archival."""
    leaking_payload = {
        "text": "call failed",
        "debug": {"request_headers": "Authorization: Bearer sk-live-leaked-secret"},
    }
    monkeypatch.setattr(
        "lumilake_server.runtime.runtime_manager.flowmesh.flowmesh_for_context",
        lambda: _FakeFlowMeshClient(leaking_payload),
    )
    saved: dict[str, Any] = {}

    def _fake_save_json_artifact(
        _self: FlowmeshRuntimeManager,
        _request_info: Any,
        _filename: str,
        data: Any,
    ) -> str:
        saved["data"] = data
        return "memory://archived.json"

    monkeypatch.setattr(
        flowmesh_manager,
        "_save_json_artifact",
        types.MethodType(_fake_save_json_artifact, flowmesh_manager),
    )
    request_info = RequestInfo(
        request_id="req-1", runtime_graphs={}, data_profile_graphs={}
    )
    request_info.batch_id = "batch-1"

    await flowmesh_manager._archive_task_response(request_info, "task-1", "node-a")

    assert "sk-live-leaked-secret" not in str(saved["data"])
