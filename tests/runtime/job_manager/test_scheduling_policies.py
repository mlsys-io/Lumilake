"""Every registered scheduling policy constructs and returns a valid batch."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from lumilake_server.runtime.job_manager.base import Job
from lumilake_server.runtime.job_manager.policies import SCHEDULING_POLICIES
from lumilake_server.runtime.job_manager.priority_queue import PriorityJobManager
from lumilake_server.runtime.optimizer.base import BaseOptimizer
from lumilake_server.runtime.protocol import LumilakeRequestConfig, Priority
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _slice_meta(graph_name: str) -> WorkflowSliceMeta:
    return WorkflowSliceMeta(
        public_graph_name=graph_name,
        slice_index=0,
        slice_start=0,
        slice_length=1,
        total_length=1,
        template_hash=f"hash-{graph_name}",
        varying_input_keys=(),
    )


def _graph(name: str) -> RuntimeGraph:
    op = RuntimeOp(
        node_id="a",
        task_type="http",
        backend="http",
        model="http",
        data_spec={},
        model_spec={},
        inference_spec={},
    )
    return RuntimeGraph(
        nodes={"a": op},
        node_order=["a"],
        output_node_map={"a": "output"},
    )


def _build_job(request_id: str, graph_name: str, user_id: str) -> Job:
    graph = _graph(graph_name)
    return Job(
        request_id=request_id,
        runtime_graphs={graph_name: graph},
        data_profile_graphs={graph_name: graph},
        dsl_graphs={graph_name: cast(Any, object())},
        workflow_slices={graph_name: _slice_meta(graph_name)},
        config=LumilakeRequestConfig(
            priority=Priority.MEDIUM,
            user_id=user_id,
            principal_id=user_id,
        ),
    )


@pytest.mark.parametrize("policy", sorted(SCHEDULING_POLICIES))
@pytest.mark.asyncio
async def test_registered_policy_constructs_and_returns_valid_batch(
    policy: str,
) -> None:
    manager = PriorityJobManager(
        optimizer=MagicMock(spec=BaseOptimizer),
        policy=policy,
    )
    await manager.enqueue(_build_job("r1", "g1", "u1"))
    await manager.enqueue(_build_job("r2", "g2", "u2"))

    batch = await manager.select_batch(2)
    assert batch is not None
    assert len(batch.workflows) <= 2
    assert {i.request_id for i in batch.workflows} <= {"r1", "r2"}
