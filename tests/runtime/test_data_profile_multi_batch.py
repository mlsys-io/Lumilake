"""A request split across scheduler batches reuses the submit-time SQL profile.

Each batch calls ``collect_data_profile`` on its own merged data-profile graph;
the lookup must find the rows that ``routes/jobs.py`` profiled once at submit.
"""

import asyncio
import copy
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.data_profile_models import DataProfileCostEstimate
from lumilake_server.graphs.graph import CompiledGraph, Graph
from lumilake_server.parser.n8n import parse_n8n_payload
from lumilake_server.runtime.data_profile_utils import (
    DataProfileSource,
    collect_data_profile,
)
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.server import LumilakeServer
from lumilake_server.utils import data_profile_offload

WORKFLOW = {
    "nodes": [
        {
            "parameters": {"options": {}},
            "type": "@n8n/n8n-nodes-langchain.chatTrigger",
            "typeVersion": 1.4,
            "position": [0, 0],
            "id": "stock",
            "name": "Stock",
        },
        {
            "parameters": {
                "operation": "executeQuery",
                "query": (
                    "SELECT close FROM lumilake_demo.ohlc_10m"
                    " WHERE symbol = '{{ $('Stock') }}'"
                ),
                "options": {},
            },
            "type": "n8n-nodes-base.postgres",
            "typeVersion": 2.6,
            "position": [200, 0],
            "id": "quote",
            "name": "Quote Query",
            "notes": '{"op-type": "data-retrieval", "is-output": true}',
        },
    ],
    "connections": {
        "Stock": {"main": [[{"node": "Quote Query", "type": "main", "index": 0}]]}
    },
}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")
    monkeypatch.setattr(
        data_profile_offload,
        "_estimate_plan_variants",
        lambda **_kwargs: [
            DataProfileCostEstimate(
                plan_id="default",
                description="planner default",
                raw_cost=1.0,
                estimated_rows=7,
                footprints={},
            )
        ],
    )
    data_profile_offload.data_profile_registry.clear()
    yield
    data_profile_offload.data_profile_registry.clear()


def _parse_slices(
    stocks: list[str],
) -> tuple[dict[str, CompiledGraph], dict[str, WorkflowSliceMeta]]:
    """Parse one slice per stock with ``scope`` set to the public name, as the
    submit path does."""
    graphs: dict[str, CompiledGraph] = {}
    slices: dict[str, WorkflowSliceMeta] = {}
    for idx, stock in enumerate(stocks):
        name = f"g0__slice_{idx + 1}"
        payload = {
            "graphs": [
                {
                    "workflow": copy.deepcopy(WORKFLOW),
                    "inputs": {"Stock": [stock]},
                    "name": name,
                    "scope": "g0",
                }
            ]
        }
        spec = parse_n8n_payload(payload)[name]
        graphs[name] = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
        slices[name] = WorkflowSliceMeta(
            public_graph_name="g0",
            slice_index=idx,
            slice_start=idx,
            slice_length=1,
            total_length=len(stocks),
            template_hash="th-fixture",
            varying_input_keys=("Stock",),
        )
    return graphs, slices


def test_later_batch_projects_the_submit_time_profile() -> None:
    graphs, slices = _parse_slices(["NVDA", "AAPL", "MSFT", "GOOG"])
    (task,) = data_profile_offload.build_request_data_profile_tasks(
        request_id="req-mb", graphs=graphs, workflow_slices=slices
    )
    result = data_profile_offload.run_data_profile_task(task.payload)
    data_profile_offload.data_profile_registry[task.task_key] = result.model_dump(
        mode="json"
    )

    batch_2 = ("g0__slice_3", "g0__slice_4")
    merged = LumilakeServer._merge_group_compiled_graph(
        [
            data_profile_offload._MergeGroupWorkflow(
                request_id="req-mb",
                public_graph_name="g0",
                template_hash="th-fixture",
                slice_index=slices[name].slice_index,
                slice_start=slices[name].slice_start,
                slice_length=1,
                total_length=4,
                workflow_id=name,
                varying_input_keys=("Stock",),
                dsl_graph=graphs[name],
            )
            for name in batch_2
        ]
    )
    batch_graph = RuntimeGraphBuilder().build(
        merged, task_type_override="data_profile", node_prefix=task.task_key
    )

    projected: dict[str, Any] = asyncio.run(
        collect_data_profile(
            request_id="req-mb",
            data_profile_graphs={task.task_key: batch_graph},
            data_profile_sources={
                task.task_key: [
                    DataProfileSource(task_key=task.task_key, org_id="default")
                ]
            },
        )
    )

    ((key, (row,)),) = projected.items()
    assert key == f"data_profile::{row['node_id']}::{row['query_name']}"
    assert row["node_id"] in batch_graph.nodes
    assert [c["estimated_rows"] for c in row["cost_estimates"]] == [7]
