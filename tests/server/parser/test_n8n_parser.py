import json
from copy import deepcopy
from pathlib import Path

import pytest

from lumilake import envs
from lumilake.graphs import Graph
from lumilake.ops import OutputOp
from lumilake.server.parser import n8n as n8n_mod
from lumilake.server.parser import parse_n8n_payload
from lumilake.server.parser.n8n import N8N_CHAT_TRIGGER

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


def test_postgres_table_retrieval_prefers_lumid_data_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "DATABASE_URL", "postgresql://direct")

    op = n8n_mod._make_postgres_retrieval_op(
        "scope",
        "Read Rows",
        {"parameters": {"operation": "select", "schema": "public", "table": "items"}},
        {},
        {},
        {},
        {},
    )

    assert op["data_spec"]["connection_string"] == "http://lumid-data"


def test_s3_retrieval_prefers_lumid_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "S3_URL", "s3://direct")

    op = n8n_mod._make_s3_retrieval_op(
        "scope",
        "Read Object",
        {
            "parameters": {
                "bucketName": "bucket/prefix",
                "options": {"folderKey": "object.txt"},
            }
        },
        {"Read Object": [{"type": "main", "node": "Read Rows"}]},
        {},
        {"Read Rows": "scope.read_rows"},
        {},
    )

    assert op["data_spec"]["connection_string"] == "http://lumid-data"


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


def test_trading_agent_image_understanding_requires_llava() -> None:
    template_path = TEMPLATE_DIR / "trading-agent.json"
    workflow = _load_template(template_path)
    bad_workflow = deepcopy(workflow)
    bad_connections = bad_workflow["connections"]

    # Disconnect News Per-row Summary from any model connection.
    bad_connections["News VLM"] = {"ai_languageModel": [[]]}
    payload = {
        "graphs": [
            {
                "name": "trading-agent",
                "workflow": bad_workflow,
                "inputs": _build_inputs(bad_workflow),
            }
        ]
    }
    with pytest.raises(ValueError, match="missing model connection"):
        parse_n8n_payload(payload)


def test_trading_agent_image_understanding_uses_rowwise_columns() -> None:
    if envs.DATABASE_URL is None:
        envs.DATABASE_URL = "sqlite://"
    if envs.S3_URL is None:
        envs.S3_URL = "s3://dummy"

    template_path = TEMPLATE_DIR / "trading-agent.json"
    workflow = _load_template(template_path)
    payload = {
        "graphs": [
            {
                "name": "trading-agent",
                "workflow": workflow,
                "inputs": _build_inputs(workflow),
            }
        ]
    }
    graph_specs = parse_n8n_payload(payload)
    graph_json = graph_specs["trading-agent"]["graph"]
    vision_ops = [
        op
        for op in graph_json.values()
        if isinstance(op, dict) and op.get("_op") == "LLMVisionOp"
    ]
    assert len(vision_ops) == 1
    vision = vision_ops[0]
    assert vision.get("rowwise_template")
    rowwise_columns = vision.get("rowwise_columns")
    assert isinstance(rowwise_columns, list) and rowwise_columns
    assert any(
        isinstance(col, dict)
        and col.get("label") == "Stock"
        and col.get("data") == {"type": "list", "items": ["NVDA"]}
        for col in rowwise_columns
    )
    assert any(
        isinstance(col, dict) and col.get("path") == "items.table.title"
        for col in rowwise_columns
    )


def test_image_generation_digest_uses_row_summary_aggregate_table() -> None:
    if envs.DATABASE_URL is None:
        envs.DATABASE_URL = "sqlite://"
    if envs.S3_URL is None:
        envs.S3_URL = "s3://dummy"

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
        and "Summarize one retrieved article" in op["rowwise_template"]
    )
    rowwise_columns = row_summary.get("rowwise_columns")
    assert isinstance(rowwise_columns, list) and rowwise_columns
    assert not any(
        isinstance(col, dict)
        and col.get("label") == "Stock"
        and isinstance(col.get("data"), dict)
        for col in rowwise_columns
    )
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
    assert [col["label"] for col in aggregate_table] == [
        "title",
        "publishedDate",
        "category",
        "summary",
    ]
    summary_col = next(col for col in aggregate_table if col["label"] == "summary")
    assert summary_col["path"] == "items.output"
    assert summary_col["node"] == row_summary["_id"]
    assert row_summary["_id"] in digest_op["_inputs"]


def test_healthcare_osteoporosis_template_compiles_with_patient_prompt() -> None:
    if envs.DATABASE_URL is None:
        envs.DATABASE_URL = "sqlite://"
    if envs.S3_URL is None:
        envs.S3_URL = "s3://dummy"

    template_path = TEMPLATE_DIR / "healthcare-osteoporosis-debate.json"
    workflow = _load_template(template_path)
    payload = {
        "graphs": [
            {
                "name": "healthcare-osteoporosis-debate",
                "workflow": workflow,
                "inputs": {
                    "PatientPrompt": [
                        "Patient profile: SEQN=1; age=72; sex_female=1; "
                        "femoral_neck_tscore=-2.7; total_spine_bmd=0.81; "
                        "hip_fracture_history=1; calcium_supplement=0"
                    ]
                },
            }
        ]
    }

    graph_specs = parse_n8n_payload(payload)
    spec = graph_specs["healthcare-osteoporosis-debate"]
    graph = Graph.from_json(spec["graph"])
    compiled = graph.compile(**spec["inputs"])
    assert compiled is not None
