import sys
import types
from typing import Any, cast

import pytest

from lumilake_server.data_profile_models import DataProfileCostEstimate
from lumilake_server.graphs.graph import CompiledGraph
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils import data_profile_offload


class _DummyGraph:
    pass


def test_build_request_data_profile_tasks_preserves_slice_specific_sql_nodes(
    monkeypatch,
) -> None:
    request_id = "req-1"
    workflow_slices = {
        "g__slice_1": WorkflowSliceMeta(
            public_graph_name="g",
            slice_index=0,
            slice_start=0,
            slice_length=1,
            total_length=2,
            template_hash="template-hash",
            varying_input_keys=("q",),
        ),
        "g__slice_2": WorkflowSliceMeta(
            public_graph_name="g",
            slice_index=1,
            slice_start=1,
            slice_length=1,
            total_length=2,
            template_hash="template-hash",
            varying_input_keys=("q",),
        ),
    }
    dummy_graph = _DummyGraph()
    graphs = {
        "g__slice_1": CompiledGraph(
            cast(Any, dummy_graph),
            {"q": ["A"], "k": ["fixed"]},
        ),
        "g__slice_2": CompiledGraph(
            cast(Any, dummy_graph),
            {"q": ["B"], "k": ["fixed"]},
        ),
    }

    build_calls: list[tuple[str, list[str]]] = []

    def _build(
        self,
        compiled: CompiledGraph,
        task_type_override: str | None = None,
        node_prefix: str | None = None,
    ) -> RuntimeGraph:
        assert task_type_override == "data_profile"
        assert node_prefix is not None
        build_calls.append((node_prefix, list(compiled.inputs["q"])))
        node_id = f"{node_prefix}__db_node"
        return RuntimeGraph(
            nodes={
                node_id: RuntimeOp(
                    node_id=node_id,
                    task_type="data_profiling",
                    backend="data_profiling",
                    model="data_profiling",
                    data_spec={
                        "type": "sql",
                        "connection_string": "postgresql://user:pw@h:5432/db",
                        "template": "select * from t where q = '{q}'",
                    },
                    model_spec={},
                    inference_spec={},
                )
            },
            node_order=[node_id],
            output_node_map={},
            dsl_to_runtime={f"{node_prefix}__raw_db_node": [node_id]},
        )

    class _Builder:
        build = _build

    monkeypatch.setattr(data_profile_offload, "RuntimeGraphBuilder", _Builder)
    tasks = data_profile_offload.build_request_data_profile_tasks(
        request_id=request_id,
        graphs=graphs,
        workflow_slices=workflow_slices,
    )
    assert len(tasks) == 2
    tasks_by_key = {t.task_key: t for t in tasks}
    assert set(tasks_by_key) == {
        "request::req-1::g::template-hash::slice_0",
        "request::req-1::g::template-hash::slice_1",
    }
    assert build_calls == [("g__slice_1", ["A"]), ("g__slice_2", ["B"])]
    task0 = tasks_by_key["request::req-1::g::template-hash::slice_0"]
    task1 = tasks_by_key["request::req-1::g::template-hash::slice_1"]
    assert task0.payload.node_order == ["g__slice_1__db_node"]
    assert task1.payload.node_order == ["g__slice_2__db_node"]
    assert (
        task0.payload.nodes["g__slice_1__db_node"].raw_node_id
        == "g__slice_1__raw_db_node"
    )
    assert (
        task1.payload.nodes["g__slice_2__db_node"].raw_node_id
        == "g__slice_2__raw_db_node"
    )


def test_project_data_profile_results_to_runtime_graph_maps_raw_node_ids() -> None:
    target_graph = RuntimeGraph(
        nodes={
            "target__db_node": RuntimeOp(
                node_id="target__db_node",
                task_type="data_profiling",
                backend="data_profiling",
                model="data_profiling",
                data_spec={"type": "sql"},
                model_spec={},
                inference_spec={},
            ),
            "target__non_sql": RuntimeOp(
                node_id="target__non_sql",
                task_type="inference",
                backend="vllm",
                model="m",
                data_spec={"type": "list"},
                model_spec={},
                inference_spec={},
            ),
        },
        node_order=["target__db_node", "target__non_sql"],
        output_node_map={},
        dsl_to_runtime={"db_node": ["target__db_node"], "other": ["target__non_sql"]},
    )
    data_profile_results_payload: dict[str, list[dict[str, Any]]] = {
        "data_profile::source__db_node::source__db_node_query": [
            {
                "node_id": "source__db_node",
                "raw_node_id": "db_node",
                "query_name": "source__db_node_query",
                "connection_string": "postgresql://user:pw@h:5432/db",
                "table": "public.t",
                "cost_estimates": [{"plan_id": "default", "raw_cost": 1.2}],
            }
        ]
    }
    data_profile_results = data_profile_offload.normalize_data_profile_result(
        data_profile_results_payload
    )

    projected = data_profile_offload.project_data_profile_results_to_runtime_graph(
        data_profile_results=data_profile_results,
        target_graph=target_graph,
    )
    expected_key = "data_profile::target__db_node::target__db_node_query"
    assert list(projected) == [expected_key]
    row = projected[expected_key][0]
    assert row.node_id == "target__db_node"
    assert row.query_name == "target__db_node_query"
    assert row.raw_node_id == "db_node"


def test_build_sample_data_profile_queries_keeps_slice_fixed_values() -> None:
    queries = data_profile_offload._build_sample_data_profile_queries(
        template=(
            "select * from news where symbol='{symbol}' "
            "and start_date='{start_date}' and end_date='{end_date}'"
        ),
        params=[
            {"label": "symbol", "data": {"type": "list", "items": ["NVDA"]}},
            {"label": "start_date", "data": {"type": "list", "items": ["2025-01-01"]}},
            {
                "label": "end_date",
                "data": {"type": "list", "items": ["2025-03-31", "2025-06-30"]},
            },
        ],
        constraints=[],
        num_samples=6,
        node_id="slice_1__sql",
    )

    assert queries
    assert all("symbol='NVDA'" in query for query in queries)
    assert all("start_date='2025-01-01'" in query for query in queries)
    assert {
        "end_date='2025-03-31'" in query or "end_date='2025-06-30'" in query
        for query in queries
    } == {True}


def test_build_sample_data_profile_queries_raises_on_unresolved_placeholder() -> None:
    with pytest.raises(ValueError, match="unresolved placeholder"):
        data_profile_offload._build_sample_data_profile_queries(
            template="select * from t where symbol='{symbol}' and dt='{date}'",
            params=[
                {"label": "symbol", "data": {"type": "list", "items": ["NVDA"]}},
            ],
            constraints=[],
            num_samples=1,
            node_id="slice_1__sql",
        )


def test_run_data_profile_task_raises_when_no_valid_estimates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_psycopg = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    monkeypatch.setattr(
        data_profile_offload,
        "_estimate_plan_variants",
        lambda **kwargs: [],
    )

    payload = data_profile_offload.DataProfileTaskPayload(
        task_key="request::req-1::g::template-hash",
        request_id="req-1",
        public_graph_name="g",
        template_hash="template-hash",
        node_order=["g__slice_1__db_node"],
        nodes={
            "g__slice_1__db_node": data_profile_offload.DataProfileTaskNode(
                node_id="g__slice_1__db_node",
                raw_node_id="g__slice_1__raw_db_node",
                data_spec={
                    "type": "sql",
                    "connection_string": "postgresql://user:pw@h:5432/db",
                    "template": "select * from t where symbol='{symbol}'",
                    "params": [
                        {
                            "label": "symbol",
                            "data": {"type": "list", "items": ["NVDA"]},
                        }
                    ],
                    "num_test_queries": 2,
                },
            )
        },
    )

    with pytest.raises(RuntimeError, match="produced no valid cost estimates"):
        data_profile_offload.run_data_profile_task(payload)


def test_run_data_profile_task_passes_num_test_queries_per_sql_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_psycopg = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    samples_by_node: list[tuple[str | None, int]] = []

    def capture_queries(**kwargs: Any) -> list[str]:
        samples_by_node.append((kwargs.get("node_id"), int(kwargs["num_samples"])))
        return [kwargs["template"]]

    monkeypatch.setattr(
        data_profile_offload,
        "_build_sample_data_profile_queries",
        capture_queries,
    )
    monkeypatch.setattr(
        data_profile_offload,
        "_estimate_plan_variants",
        lambda **kwargs: [
            DataProfileCostEstimate(
                plan_id="default",
                description="planner default",
                raw_cost=1.0,
                estimated_rows=10,
                footprints={},
            )
        ],
    )

    payload = data_profile_offload.DataProfileTaskPayload(
        task_key="request::req-1::g::template-hash",
        request_id="req-1",
        public_graph_name="g",
        template_hash="template-hash",
        node_order=["sql_1", "sql_2"],
        nodes={
            "sql_1": data_profile_offload.DataProfileTaskNode(
                node_id="sql_1",
                raw_node_id="raw_sql_1",
                data_spec={
                    "type": "sql",
                    "connection_string": "postgresql://user:pw@h:5432/db",
                    "template": "select 1 from public.t",
                    "table": "public.t",
                    "num_test_queries": 3,
                },
            ),
            "sql_2": data_profile_offload.DataProfileTaskNode(
                node_id="sql_2",
                raw_node_id="raw_sql_2",
                data_spec={
                    "type": "sql",
                    "connection_string": "postgresql://user:pw@h:5432/db",
                    "template": "select 2 from public.t",
                    "table": "public.t",
                    "num_test_queries": 5,
                },
            ),
        },
    )

    data_profile_offload.run_data_profile_task(payload)
    assert samples_by_node == [("sql_1", 3), ("sql_2", 5)]
