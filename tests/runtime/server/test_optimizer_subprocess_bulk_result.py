"""The optimizer subprocess must hand back schedules larger than a pipe buffer.

A spawned child cannot exit while its queue feeder thread is still flushing, so
the parent has to read the result queue while the child runs.
"""

from typing import Any, cast

import pytest
from support.optimizer_bulk_schedule_plugin import OPTIMIZER_TYPE, PAYLOAD_BYTES
from support.runtime_server import RecordingRuntimeManager

from lumilake import envs
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp

_BULK_PLUGIN = "support.optimizer_bulk_schedule_plugin"


@pytest.mark.asyncio
async def test_generate_schedule_in_subprocess_returns_bulk_schedule(
    server_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_factory()
    server.runtime_manager = cast(Any, RecordingRuntimeManager())
    monkeypatch.setenv("LUMILAKE_PLUGINS", _BULK_PLUGIN)
    monkeypatch.setattr(envs, "LUMILAKE_PLUGINS", [_BULK_PLUGIN])
    monkeypatch.setattr(envs, "LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS", 10)

    op = RuntimeOp(
        node_id="n1",
        task_type="inference",
        backend="vllm",
        model="gpt2",
        data_spec={},
        model_spec={},
        inference_spec={},
        dependencies=(),
    )
    graph = RuntimeGraph(
        nodes={"n1": op},
        node_order=["n1"],
        output_node_map={"n1": "out"},
        dsl_to_runtime={},
    )

    schedule = await server._generate_schedule_in_subprocess(
        request_id="req-bulk",
        batch_id="batch-bulk",
        optimizer_type=OPTIMIZER_TYPE,
        runtime_graph=graph,
        selected_workers=["worker-1"],
        worker_profiles={"worker-1": {}},
        data_profile_results={},
    )

    assert schedule.worker_assignment["worker-1"] == ["n" + "x" * PAYLOAD_BYTES]
