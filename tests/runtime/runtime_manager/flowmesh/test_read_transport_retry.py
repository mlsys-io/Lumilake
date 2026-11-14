"""A read repeats a dropped connection; a submit never does."""

from typing import Any

import httpx
import pytest

import lumilake_server.runtime.runtime_manager.flowmesh as fm_mod


@pytest.mark.asyncio
async def test_a_read_repeats_a_dropped_connection_but_a_submit_does_not(
    flowmesh_manager: fm_mod.FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The task-status read survives a refused connection at the gateway delays;
    a submit raises it, since FlowMesh may already hold the workflow."""
    calls = {"status": 0, "submit": 0}
    delays: list[float] = []

    async def _status(*_: Any) -> str:
        calls["status"] += 1
        if calls["status"] < 3:
            raise httpx.ConnectError("refused")
        return "DONE"

    async def _submit(*_: Any) -> Any:
        calls["submit"] += 1
        raise httpx.ConnectError("refused")

    async def _sleep(delay: float) -> None:
        delays.append(delay)

    fm = type(
        "Fm",
        (),
        {
            "tasks": type("T", (), {"status": staticmethod(_status)}),
            "workflows": type("W", (), {"submit": staticmethod(_submit)}),
        },
    )()
    monkeypatch.setattr(fm_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(fm_mod.FlowmeshRuntimeManager, "fm", property(lambda self: fm))

    assert await flowmesh_manager.fetch_task_status("t-1") == "DONE"
    with pytest.raises(httpx.ConnectError):
        await flowmesh_manager._submit_workflow("yaml", "sid-1")
    assert calls == {"status": 3, "submit": 1}
    assert delays == list(fm_mod._FLOWMESH_RETRY_DELAYS_S[:2])
