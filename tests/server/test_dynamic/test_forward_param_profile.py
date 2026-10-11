"""Data profiling resolves a SQL param forwarded from an earlier round.

A round boundary separates ``SQL Date Planner`` (an LLM with
``structural_outputs``) from ``Market Query`` (a DataRetrievalOp whose
``data_spec.params`` references the planner's ``start_date``). ``forward_refs``
rewrites the param to a per-(node, path) workflow input; the profiling preflight
must resolve that InputOp-bound param from the forwarded input's values, the same
way a param bound to an ordinary workflow input is resolved.
"""

from typing import Any

import pytest

from lumilake import envs
from lumilake_server.data_profile_models import DataProfileCostEstimate
from lumilake_server.dynamic.blocks import INPUT_NODE_ID
from lumilake_server.dynamic.driver import forward_refs
from lumilake_server.graphs import Graph
from lumilake_server.parser.yaml_parser import parse_yaml_payload
from lumilake_server.routes.jobs import (
    _chunk_inputs,
    _dispatch_workflow_to_graph_specs,
    _input_shape,
    _workflow_template_hash,
)
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils import data_profile_offload


def _llm_config(op_id: str) -> dict:
    return {
        "id": op_id,
        "op": "LLMChatOp",
        "inputs": [],
        "prompt": {"template": "x", "format_kwargs": {}},
    }


def _sql_config(op_id: str) -> dict:
    return {
        "id": op_id,
        "op": "DataRetrievalOp",
        "inputs": [],
        "data_spec": {"type": "lumid", "mode": "sql", "template": "SELECT *"},
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


def _build_child_submission(
    subgraph: list[dict[str, Any]],
    results: dict[str, list[dict[str, Any]]],
    configs: dict[str, dict[str, Any]],
    symbols: list[str],
) -> tuple[dict[str, Any], dict[str, WorkflowSliceMeta]]:
    """Slice a forwarded round exactly as ``_submit_dynamic_child`` does:
    ``input_batch_size`` 1, one slice per row."""
    forwarded, forwarded_inputs = forward_refs(
        subgraph, results, configs, rows=len(symbols)
    )
    workflow_inputs: dict[str, list[str]] = {INPUT_NODE_ID: []}
    workflow_inputs.update(forwarded_inputs)
    workflow = {
        "name": "round_1",
        "inputs": workflow_inputs,
        "ops": forwarded,
        "outputs": [{"name": "Market Query", "ref": "Market Query"}],
    }
    parsed = parse_yaml_payload(workflow)
    graph_name = next(iter(parsed))
    native = parsed[graph_name]["graph"]

    inputs = {INPUT_NODE_ID: list(symbols)}
    inputs.update(forwarded_inputs)
    total_length, varying_input_keys = _input_shape(inputs)
    input_batches = _chunk_inputs(inputs, 1)
    template_hash = _workflow_template_hash({"graph": native}, "native")
    graph_specs: dict[str, dict[str, Any]] = {}
    workflow_slices: dict[str, WorkflowSliceMeta] = {}
    slice_start = 0
    for batch_idx, batch_inputs in enumerate(input_batches):
        slice_name = (
            graph_name
            if len(input_batches) == 1
            else f"{graph_name}__slice_{batch_idx + 1}"
        )
        slice_length, _ = _input_shape(batch_inputs)
        workflow_slices[slice_name] = WorkflowSliceMeta(
            public_graph_name=graph_name,
            slice_index=batch_idx,
            slice_start=slice_start,
            slice_length=slice_length,
            total_length=total_length,
            template_hash=template_hash,
            varying_input_keys=varying_input_keys,
        )
        _dispatch_workflow_to_graph_specs(
            workflow_format="native",
            workflow_payload={"graph": native},
            batch_inputs=batch_inputs,
            graph_name=slice_name,
            graph_specs=graph_specs,
            idx=0,
        )
        slice_start += slice_length
    graphs = {
        name: Graph.from_json(spec["graph"]).compile(**spec["inputs"])
        for name, spec in graph_specs.items()
    }
    return graphs, workflow_slices


def test_forwarded_sql_param_resolves_from_round_input() -> None:
    planner_id = "SQL Date Planner"
    results = {
        planner_id: [
            {"output": {"start_date": "2023-01-01T00:00:00Z"}},
            {"output": {"start_date": "2020-01-01T00:00:00Z"}},
        ]
    }
    configs = {planner_id: _llm_config(planner_id)}
    subgraph = [
        {
            "id": "Market Query",
            "op": "DataRetrievalOp",
            "inputs": [],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "template": ("SELECT * FROM market WHERE start_date = '{start_date}'"),
                "params": [
                    {
                        "label": "start_date",
                        "node": planner_id,
                        "path": "items.output.start_date",
                    }
                ],
            },
        }
    ]

    graphs, workflow_slices = _build_child_submission(
        subgraph, results, configs, ["NVDA", "AAPL"]
    )
    (task,) = data_profile_offload.build_request_data_profile_tasks(
        request_id="req-fwd", graphs=graphs, workflow_slices=workflow_slices
    )
    (node,) = task.payload.nodes.values()
    queries = data_profile_offload._build_sample_data_profile_queries(
        template=node.data_spec["template"],
        params=node.data_spec["params"],
        constraints=node.data_spec.get("constraints"),
        num_samples=1,
        node_id=node.node_id,
    )
    # The rendered sample query carries one of the forwarded start_date values.
    assert any(
        value in queries[0]
        for value in ("2023-01-01T00:00:00Z", "2020-01-01T00:00:00Z")
    )


def test_forwarded_sql_param_is_literal_in_execution_graph() -> None:
    """The execution runtime graph resolves a forwarded SQL param to a literal
    list of the per-row values, not a dangling node ref.

    The forwarded ``start_date`` is bound as a workflow input that is referenced
    only via ``data_spec.params[*].node``, so it is dropped from the
    topologically sorted graph. The execution path must resolve it from the
    graph's input-ops map (as the data-profile path does) so the FlowMesh worker
    receives concrete values instead of a node reference it cannot resolve.
    """
    planner_id = "SQL Date Planner"
    results = {
        planner_id: [
            {"output": {"start_date": "2023-01-01T00:00:00Z"}},
            {"output": {"start_date": "2020-01-01T00:00:00Z"}},
        ]
    }
    configs = {planner_id: _llm_config(planner_id)}
    subgraph = [
        {
            "id": "Market Query",
            "op": "DataRetrievalOp",
            "inputs": [],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "template": ("SELECT * FROM market WHERE start_date = '{start_date}'"),
                "params": [
                    {
                        "label": "start_date",
                        "node": planner_id,
                        "path": "items.output.start_date",
                    }
                ],
            },
        }
    ]

    graphs, _ = _build_child_submission(subgraph, results, configs, ["NVDA", "AAPL"])
    # One slice per symbol; each carries its own forwarded start_date value.
    assert set(graphs) == {"round_1__slice_1", "round_1__slice_2"}
    for graph in graphs.values():
        runtime_graph = RuntimeGraphBuilder().build(graph)
        (retrieval,) = [
            node
            for node in runtime_graph.nodes.values()
            if node.task_type == "data_retrieval"
        ]
        # The forwarded start_date is inlined as a literal value into the
        # template, with no node reference to a node missing from the graph.
        assert "node" not in str(retrieval.data_spec["params"])
        assert any(
            value in retrieval.data_spec["template"]
            for value in ("2023-01-01T00:00:00Z", "2020-01-01T00:00:00Z")
        )
