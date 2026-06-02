import json
from pathlib import Path

import pytest
from lumilake import envs

from lumilake_server.graphs import Graph
from lumilake_server.ops import OutputOp
from lumilake_server.parser import parse_n8n_payload
from lumilake_server.parser.n8n import N8N_CHAT_TRIGGER

TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "examples" / "templates" / "n8n"


def _load_template(path: Path) -> dict:
    return json.loads(path.read_text())


def _default_input_value(name: str) -> str:
    lowered = name.lower()
    if "question" in lowered or "query" in lowered:
        return "What is 2+2?"
    if "entity" in lowered or "stock" in lowered:
        return "NVDA"
    return "test"


def _build_inputs(workflow: dict) -> dict[str, list[str]]:
    inputs: dict[str, list[str]] = {}
    for node in workflow.get("nodes", []):
        if node.get("type") != N8N_CHAT_TRIGGER:
            continue
        name = node.get("name")
        if not name:
            continue
        inputs[name] = [_default_input_value(name)]
    if not inputs:
        raise ValueError("workflow missing chat trigger inputs")
    return inputs


@pytest.mark.parametrize(
    "template_path",
    sorted(TEMPLATE_DIR.glob("*.json")),
    ids=lambda p: str(p.relative_to(TEMPLATE_DIR)),
)
def test_parse_n8n_templates(template_path: Path):
    if envs.DATABASE_URL is None:
        envs.DATABASE_URL = "sqlite://"
    if envs.S3_URL is None:
        envs.S3_URL = "s3://dummy"
    if envs.S3_WORKER_URL is None:
        envs.S3_WORKER_URL = "s3://dummy/bucket"
    if not envs.LUMID_DATA_URL:
        envs.LUMID_DATA_URL = "http://lumid-data"

    workflow = _load_template(template_path)
    inputs = _build_inputs(workflow)
    payload = {
        "graphs": [{"name": template_path.stem, "workflow": workflow, "inputs": inputs}]
    }

    graph_specs = parse_n8n_payload(payload)
    assert len(graph_specs) == 1

    spec = graph_specs[template_path.stem]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(**spec["inputs"])
    assert compiled is not None
    assert list(graph.iter_ops(OutputOp)), "expected at least one output op"


def test_image_generation_digest_uses_row_summary_aggregate_table() -> None:
    if envs.DATABASE_URL is None:
        envs.DATABASE_URL = "sqlite://"
    if envs.S3_URL is None:
        envs.S3_URL = "s3://dummy"
    if envs.S3_WORKER_URL is None:
        envs.S3_WORKER_URL = "s3://dummy/bucket"

    template_path = TEMPLATE_DIR / "image-generation.json"
    workflow = _load_template(template_path)
    payload = {
        "graphs": [
            {
                "name": "image-generation",
                "workflow": workflow,
                "inputs": {"Stock": ["NVDA", "MSFT", "AAPL"]},
            }
        ]
    }
    graph_specs = parse_n8n_payload(payload)
    graph_json = graph_specs["image-generation"]["graph"]
    llm_ops = [
        op
        for op in graph_json.values()
        if isinstance(op, dict) and op.get("_op") == "LLMChatOp"
    ]
    assert llm_ops

    row_summary = next(
        op
        for op in llm_ops
        if isinstance(op.get("rowwise_template"), str)
        and "Summarize one article" in op["rowwise_template"]
    )
    rowwise_columns = row_summary.get("rowwise_columns")
    assert isinstance(rowwise_columns, list) and rowwise_columns
    assert any(
        isinstance(col, dict)
        and col.get("label") == "symbol"
        and col.get("path") == "items.table.symbol"
        for col in rowwise_columns
    )

    digest_op = next(
        op
        for op in llm_ops
        if isinstance(op.get("aggregate_table"), list)
        and any(
            isinstance(col, dict) and col.get("label") == "summary"
            for col in op["aggregate_table"]
        )
    )
    aggregate_table = digest_op["aggregate_table"]
    summary_col = next(col for col in aggregate_table if col["label"] == "summary")
    assert summary_col["path"] == "items.output"
    assert summary_col["node"] == row_summary["_id"]
    assert row_summary["_id"] in digest_op["_inputs"]
