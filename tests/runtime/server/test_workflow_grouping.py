import json
from pathlib import Path

import pytest
from lumilake import envs
from support.runtime_server import make_workflow, make_workflow_slices_from_inputs

from lumilake_server.graphs import Graph
from lumilake_server.ops import DataRetrievalOp, LLMChatOp
from lumilake_server.parser.n8n import parse_n8n_payload
from lumilake_server.runtime.server import LumilakeServer


@pytest.fixture(autouse=True)
def _direct_parser_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "DATABASE_URL", "sqlite://")
    monkeypatch.setattr(envs, "S3_URL", "s3://dummy")
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "")


def _build_image_generation_slices(
    *,
    symbols: list[str],
    varying_input_keys: tuple[str, ...] = ("Stock",),
) -> list[object]:
    template = json.loads(
        Path("examples/templates/n8n/image-generation.json").read_text()
    )
    workflows: list[object] = []
    total_length = len(symbols)
    for idx, symbol in enumerate(symbols):
        graph_name = f"image-generation__slice_{idx + 1}"
        payload = {
            "graphs": [
                {
                    "name": graph_name,
                    "workflow": template,
                    "inputs": {
                        "Stock": [symbol],
                    },
                }
            ]
        }
        parsed = parse_n8n_payload(payload)
        spec = parsed[graph_name]
        compiled_graph = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
        workflow = make_workflow(
            workflow_id=f"wf-{idx}",
            request_id="req-image",
            graph_name=graph_name,
            public_graph_name="03-image-generation",
            template_hash="template-image-generation",
            dsl_inputs=compiled_graph.inputs,
            varying_input_keys=varying_input_keys,
            slice_index=idx,
            slice_start=idx,
            slice_length=1,
            total_length=total_length,
        )
        workflow.dsl_graph = compiled_graph
        workflows.append(workflow)
    return workflows


def _build_etl_slices(
    *,
    images: list[str],
    varying_input_keys: tuple[str, ...] = ("Image",),
) -> list[object]:
    template = json.loads(
        Path("examples/templates/n8n/etl-image-news-summary.json").read_text()
    )
    workflows: list[object] = []
    total_length = len(images)
    for idx, image in enumerate(images):
        graph_name = f"etl__slice_{idx + 1}"
        payload = {
            "graphs": [
                {
                    "name": graph_name,
                    "workflow": template,
                    "inputs": {"Image": [image]},
                }
            ]
        }
        parsed = parse_n8n_payload(payload)
        spec = parsed[graph_name]
        compiled_graph = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
        workflow = make_workflow(
            workflow_id=f"etl-{idx}",
            request_id="req-etl",
            graph_name=graph_name,
            public_graph_name="04-ETL",
            template_hash="template-etl",
            dsl_inputs=compiled_graph.inputs,
            varying_input_keys=varying_input_keys,
            slice_index=idx,
            slice_start=idx,
            slice_length=1,
            total_length=total_length,
        )
        workflow.dsl_graph = compiled_graph
        workflows.append(workflow)
    return workflows


def _find_sql_symbol_items(compiled_graph: Graph) -> list[str]:
    for op in compiled_graph.as_dict().values():
        if not isinstance(op, DataRetrievalOp):
            continue
        data_spec = op.data_spec
        spec_type = data_spec.get("type")
        if spec_type != "sql":
            continue
        params = data_spec.get("params")
        if not isinstance(params, list):
            continue
        for param in params:
            if not isinstance(param, dict):
                continue
            label = param.get("label")
            data = param.get("data")
            if not isinstance(label, str) or "Stock" not in label:
                continue
            if not isinstance(data, dict):
                continue
            items = data.get("items")
            if isinstance(items, list):
                return list(items)
    raise AssertionError("Failed to locate SQL symbol param list in merged graph")


def test_group_workflows_by_parent_workflow_keeps_request_slices_together() -> None:
    workflows = [
        *make_workflow_slices_from_inputs(
            request_id="req-a",
            public_graph_name="shared",
            entities=["nvda", "msft"],
        ),
        *make_workflow_slices_from_inputs(
            request_id="req-b",
            public_graph_name="shared",
            entities=["nvda", "aapl"],
        ),
    ]

    grouped = LumilakeServer._group_workflows_by_parent_workflow(workflows)
    grouped_members = sorted(
        sorted(item.workflow_id for item in items) for items in grouped.values()
    )

    assert grouped_members == [["req-a-0", "req-a-1"], ["req-b-0", "req-b-1"]]
    assert all(key.startswith("request::") for key in grouped)


def test_group_by_parent_workflow_merges_all_slices_per_request() -> None:
    workflows = [
        *make_workflow_slices_from_inputs(
            request_id="req-a",
            public_graph_name="shared",
            entities=["nvda", "msft", "tsla"],
        ),
        *make_workflow_slices_from_inputs(
            request_id="req-b",
            public_graph_name="shared",
            entities=["nvda", "aapl"],
        ),
    ]

    grouped = LumilakeServer._group_workflows_by_parent_workflow(workflows)
    grouped_members = sorted(
        sorted(item.workflow_id for item in items) for items in grouped.values()
    )

    assert grouped_members == [
        ["req-a-0", "req-a-1", "req-a-2"],
        ["req-b-0", "req-b-1"],
    ]
    assert all(key.startswith("request::") for key in grouped)


def test_group_workflows_by_parent_workflow_merges_slices_within_each_template() -> (
    None
):
    workflows = [
        make_workflow(
            workflow_id="social-0",
            request_id="req-social",
            graph_name="social-g0",
            public_graph_name="social",
            template_hash="template-social",
            dsl_inputs={"entity": ["nvda"]},
            varying_input_keys=("entity",),
            slice_index=0,
            slice_start=0,
            slice_length=1,
            total_length=2,
        ),
        make_workflow(
            workflow_id="social-1",
            request_id="req-social",
            graph_name="social-g1",
            public_graph_name="social",
            template_hash="template-social",
            dsl_inputs={"entity": ["msft"]},
            varying_input_keys=("entity",),
            slice_index=1,
            slice_start=1,
            slice_length=1,
            total_length=2,
        ),
        make_workflow(
            workflow_id="trading-0",
            request_id="req-trading",
            graph_name="trading-g0",
            public_graph_name="trading",
            template_hash="template-trading",
            dsl_inputs={"entity": ["nvda"]},
            varying_input_keys=("entity",),
            slice_index=0,
            slice_start=0,
            slice_length=1,
            total_length=2,
        ),
        make_workflow(
            workflow_id="trading-1",
            request_id="req-trading",
            graph_name="trading-g1",
            public_graph_name="trading",
            template_hash="template-trading",
            dsl_inputs={"entity": ["msft"]},
            varying_input_keys=("entity",),
            slice_index=1,
            slice_start=1,
            slice_length=1,
            total_length=2,
        ),
    ]

    grouped = LumilakeServer._group_workflows_by_parent_workflow(workflows)
    grouped_members = sorted(
        sorted(item.workflow_id for item in items) for items in grouped.values()
    )

    assert grouped_members == [["social-0", "social-1"], ["trading-0", "trading-1"]]
    for key in grouped:
        assert key.startswith("request::")


def test_merge_group_compiled_graph_rewrites_varying_retrieval_literals() -> None:
    symbols = ["NVDA", "MSFT"]
    workflows = _build_image_generation_slices(symbols=symbols)

    merged_compiled = LumilakeServer._merge_group_compiled_graph(workflows)

    assert merged_compiled.inputs["Stock"] == symbols
    assert _find_sql_symbol_items(merged_compiled.graph) == symbols


def test_merge_group_compiled_graph_image_generation_rewrites_stock_everywhere() -> (
    None
):
    symbols = ["NVDA", "MSFT", "AAPL"]
    workflows = _build_image_generation_slices(symbols=symbols)

    merged_compiled = LumilakeServer._merge_group_compiled_graph(workflows)

    assert merged_compiled.inputs["Stock"] == symbols
    assert _find_sql_symbol_items(merged_compiled.graph) == symbols
    assert isinstance(merged_compiled.graph, Graph)
    assert any(
        isinstance(op, LLMChatOp)
        and op.rowwise_columns is not None
        and any(
            isinstance(column, dict)
            and column.get("label") == "symbol"
            and column.get("path") == "items.table.symbol"
            for column in op.rowwise_columns
        )
        for op in merged_compiled.graph.as_dict().values()
    )


def test_merge_group_compiled_graph_etl_merges_varying_input_via_input_op() -> None:
    images = ["images/a.png", "images/b.png"]
    workflows = _build_etl_slices(images=images)

    merged_compiled = LumilakeServer._merge_group_compiled_graph(workflows)

    assert merged_compiled.inputs["Image"] == images
    assert merged_compiled._coalesce_rewrite_hits == {"Image": 0}
    assert merged_compiled._coalesce_rewrite_skipped is False


def test_merge_group_compiled_graph_single_slice_skips_rewrite_strictness() -> None:
    workflows = _build_etl_slices(images=["images/a.png"])

    merged_compiled = LumilakeServer._merge_group_compiled_graph(workflows)

    assert merged_compiled.inputs == {"Image": ["images/a.png"]}
    assert merged_compiled._coalesce_rewrite_hits == {}
    assert merged_compiled._coalesce_rewrite_skipped is True
