import json
from pathlib import Path

import pytest
from support.runtime_server import make_workflow, make_workflow_slices_from_inputs

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.ops import DataRetrievalOp, LLMChatOp
from lumilake_server.parser.n8n import parse_n8n_payload
from lumilake_server.runtime.server import LumilakeServer


@pytest.fixture(autouse=True)
def _direct_parser_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


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
    varying_input_keys: tuple[str, ...] = ("Stock",),
) -> list[object]:
    template = json.loads(
        Path("examples/templates/n8n/image-generation.json").read_text()
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
                    "inputs": {"Stock": [image]},
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
        if data_spec.get("type") != "lumid" or data_spec.get("mode") != "sql":
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

    assert grouped_members == [
        ["social-0", "social-1"],
        ["trading-0", "trading-1"],
    ]
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


def test_merge_group_compiled_graph_single_slice_skips_rewrite_strictness() -> None:
    workflows = _build_etl_slices(images=["NVDA"])

    merged_compiled = LumilakeServer._merge_group_compiled_graph(workflows)

    assert merged_compiled.inputs == {"Stock": ["NVDA"]}
    assert merged_compiled._coalesce_rewrite_hits == {}
    assert merged_compiled._coalesce_rewrite_skipped is True


def _rowwise_prompt_workflow(prompt_text: str) -> dict:
    return {
        "nodes": [
            {
                "parameters": {},
                "type": "@n8n/n8n-nodes-langchain.chatTrigger",
                "name": "Paper",
                "id": "p",
                "typeVersion": 1,
            },
            {
                "parameters": {"model": "gpt-4o"},
                "type": "@n8n/n8n-nodes-langchain.lmOpenHuggingFaceInference",
                "name": "Model",
                "id": "m",
                "typeVersion": 1,
            },
            {
                "parameters": {
                    "text": prompt_text,
                    "messages": {"messageValues": [{"message": "You are a judge."}]},
                },
                "type": "@n8n/n8n-nodes-langchain.chainLlm",
                "name": "Relevance Judge",
                "id": "rj",
                "typeVersion": 1,
                "notes": '{"op-type":"text-generation","mode":"rowwise"}',
            },
        ],
        "connections": {
            "Paper": {
                "main": [[{"node": "Relevance Judge", "type": "main", "index": 0}]]
            },
            "Model": {
                "ai_languageModel": [
                    [
                        {
                            "node": "Relevance Judge",
                            "type": "ai_languageModel",
                            "index": 0,
                        }
                    ]
                ]
            },
        },
    }


def _rowwise_slices(
    *, prompt_text: str, rows: list[str], request_id: str = "req-relwork"
) -> list[object]:
    workflows: list[object] = []
    total_length = len(rows)
    for idx, row in enumerate(rows):
        graph_name = f"relwork__slice_{idx + 1}"
        payload = {
            "graphs": [
                {
                    "name": graph_name,
                    "workflow": _rowwise_prompt_workflow(prompt_text),
                    "inputs": {"Paper": [row]},
                }
            ]
        }
        parsed = parse_n8n_payload(payload)
        spec = parsed[graph_name]
        compiled_graph = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
        workflow = make_workflow(
            workflow_id=f"relwork-{idx}",
            request_id=request_id,
            graph_name=graph_name,
            public_graph_name="related-work",
            template_hash="template-relwork",
            dsl_inputs=compiled_graph.inputs,
            varying_input_keys=("Paper",),
            slice_index=idx,
            slice_start=idx,
            slice_length=1,
            total_length=total_length,
        )
        workflow.dsl_graph = compiled_graph
        workflows.append(workflow)
    return workflows


def _rowwise_columns(compiled_graph: Graph) -> list[dict]:
    for op in compiled_graph.as_dict().values():
        if not isinstance(op, LLMChatOp):
            continue
        if op.rowwise_columns:
            return op.rowwise_columns
    raise AssertionError("No rowwise LLM found")


def _runtime_columns(graph: Graph, inputs: dict[str, list[str]]) -> list[dict]:
    from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

    runtime_graph = RuntimeGraphBuilder().build(graph.compile(**inputs))
    for node in runtime_graph.nodes.values():
        data_spec = node.data_spec
        if isinstance(data_spec, dict) and data_spec.get("type") == "dataframe":
            return data_spec["columns"]
    raise AssertionError("No dataframe node found")


def test_merge_group_compiled_graph_coalesces_projected_input_columns() -> None:
    """Coalescing N single-row slices must produce the same runtime columns as
    parsing the same N rows in one graph: each field projected per row, grouped
    when more than one row, in merged order, for field, nested-field and
    whole-item refs."""
    rows = [
        '{"title": "T0", "meta": {"year": 2020}}',
        '{"title": "T1", "meta": {"year": 2021}}',
    ]
    for prompt_text in (
        "Judge: {{ $('Paper').item.title }}",
        "Judge: {{ $('Paper').item.meta.year }}",
        "Judge: {{ $('Paper').item.json }}",
    ):
        workflows = _rowwise_slices(prompt_text=prompt_text, rows=rows)
        merged = LumilakeServer._merge_group_compiled_graph(workflows)

        direct = Graph.from_json(
            parse_n8n_payload(
                {
                    "graphs": [
                        {
                            "name": "g",
                            "workflow": _rowwise_prompt_workflow(prompt_text),
                            "inputs": {"Paper": rows},
                        }
                    ]
                }
            )["g"]["graph"]
        ).compile(Paper=rows)

        assert _runtime_columns(merged.graph, merged.inputs) == _runtime_columns(
            direct.graph, direct.inputs
        )


def test_merge_group_compiled_graph_non_varying_input_ref_not_rewritten() -> None:
    """A rowwise column bound from a fixed (non-varying) input must not be
    re-projected even when its label collides with a varying key."""
    workflow = {
        "nodes": [
            {
                "parameters": {},
                "type": "@n8n/n8n-nodes-langchain.chatTrigger",
                "name": "Paper",
                "id": "p",
                "typeVersion": 1,
            },
            {
                "parameters": {},
                "type": "@n8n/n8n-nodes-langchain.chatTrigger",
                "name": "Stock",
                "id": "s",
                "typeVersion": 1,
            },
            {
                "parameters": {"model": "gpt-4o"},
                "type": "@n8n/n8n-nodes-langchain.lmOpenHuggingFaceInference",
                "name": "Model",
                "id": "m",
                "typeVersion": 1,
            },
            {
                "parameters": {
                    "text": "Judge: {{ $('Paper').item.Stock }} {{ $('Stock').item }}",
                    "messages": {"messageValues": [{"message": "You are a judge."}]},
                },
                "type": "@n8n/n8n-nodes-langchain.chainLlm",
                "name": "Relevance Judge",
                "id": "rj",
                "typeVersion": 1,
                "notes": '{"op-type":"text-generation","mode":"rowwise"}',
            },
        ],
        "connections": {
            "Paper": {
                "main": [[{"node": "Relevance Judge", "type": "main", "index": 0}]]
            },
            "Stock": {
                "main": [[{"node": "Relevance Judge", "type": "main", "index": 0}]]
            },
            "Model": {
                "ai_languageModel": [
                    [
                        {
                            "node": "Relevance Judge",
                            "type": "ai_languageModel",
                            "index": 0,
                        }
                    ]
                ]
            },
        },
    }
    workflows: list[object] = []
    for idx, stock in enumerate(["S0", "S1"]):
        graph_name = f"relwork__slice_{idx + 1}"
        payload = {
            "graphs": [
                {
                    "name": graph_name,
                    "workflow": workflow,
                    "inputs": {"Paper": ['{"Stock": "paper"}'], "Stock": [stock]},
                }
            ]
        }
        parsed = parse_n8n_payload(payload)
        spec = parsed[graph_name]
        compiled_graph = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
        wf = make_workflow(
            workflow_id=f"relwork-{idx}",
            request_id="req-relwork",
            graph_name=graph_name,
            public_graph_name="related-work",
            template_hash="template-relwork",
            dsl_inputs=compiled_graph.inputs,
            varying_input_keys=("Stock",),
            slice_index=idx,
            slice_start=idx,
            slice_length=1,
            total_length=2,
        )
        wf.dsl_graph = compiled_graph
        workflows.append(wf)

    merged = LumilakeServer._merge_group_compiled_graph(workflows)

    # The Paper-bound column keeps its single projected value; Stock is not
    # projected from it.
    columns = _rowwise_columns(merged.graph)
    paper_col = next(col for col in columns if col["label"] == "Stock")
    assert paper_col["data"]["items"] == ["paper"]
