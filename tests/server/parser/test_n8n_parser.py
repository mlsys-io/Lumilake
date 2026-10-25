import json
from pathlib import Path

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.ops import OutputOp
from lumilake_server.parser import parse_n8n_payload
from lumilake_server.parser.n8n import (
    N8N_AGENT_NODE,
    N8N_CHAT_TRIGGER,
    N8N_POSTGRES_NODE,
    N8N_S3_NODE,
    _build_aggregate_prompt_content,
    _build_prompt_content,
    _build_rowwise_prompt_content,
    _extract_rowwise_columns,
    _make_chain_ops,
    _referenced_op_path,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

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
def test_parse_n8n_templates(template_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")

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

    # Verify every DataRetrievalOp carries the new lumid wire format
    from lumilake_server.ops import DataRetrievalOp as _DROp

    for op in graph.iter_ops(_DROp):
        ds = op.data_spec
        assert ds.get("type") == "lumid", f"op {op.id} has type {ds.get('type')!r}"
        assert ds.get("mode") in {
            "sql",
            "s3",
            "agent",
        }, f"op {op.id} has unknown mode {ds.get('mode')!r}"
        assert "connection_string" not in ds


def test_rowwise_agent_reference_uses_table_path() -> None:
    node_map = {
        "Agent": {"type": N8N_AGENT_NODE},
    }
    op_ids = {"Agent": "op_agent"}
    columns = _extract_rowwise_columns(
        "Summarize {{ $('Agent') }}",
        node_map,
        op_ids,
        inputs={},
    )
    assert any(
        col.get("node") == "op_agent" and col.get("path") == "items.table"
        for col in columns
    )


def test_aggregate_upstream_sql_reference_uses_table_path() -> None:
    node_map = {
        "SQL": {"type": N8N_POSTGRES_NODE},
    }
    op_ids = {"Main": "op_main"}
    prompt = (
        "Build a digest.\n" "```table\n" "summary: {{ $('SQL').item.synopsis }}\n" "```"
    )
    _, table_spec = _build_aggregate_prompt_content(
        prompt, op_ids, upstream_main="Main", node_map=node_map
    )
    assert any(
        col.get("node") == "op_main" and col.get("path") == "items.table.synopsis"
        for col in table_spec
    )


def test_parse_n8n_templates_agent_missing_lumid_data_url_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime graph builder raises when LUMID_DATA_URL is missing."""
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "tok")

    agent_template = TEMPLATE_DIR / "agent-retrieval.json"
    if not agent_template.exists():
        pytest.skip("agent-retrieval.json not found")
    workflow = _load_template(agent_template)
    inputs = _build_inputs(workflow)
    payload = {
        "graphs": [
            {"name": agent_template.stem, "workflow": workflow, "inputs": inputs}
        ]
    }
    graph_specs = parse_n8n_payload(payload)
    spec = graph_specs[agent_template.stem]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    with pytest.raises(ValueError, match="LUMID_DATA_URL"):
        RuntimeGraphBuilder().build(compiled)


def test_image_generation_digest_uses_row_summary_aggregate_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")

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


def test_referenced_op_path_agent_uses_table() -> None:
    assert _referenced_op_path(N8N_AGENT_NODE, None) == "items.table"
    assert _referenced_op_path(N8N_POSTGRES_NODE, "symbol") == "items.table.symbol"
    assert _referenced_op_path(N8N_S3_NODE, None) == "items.content"
    assert _referenced_op_path(None, None) == "items.output"


def _code_node_workflow(notes: str) -> dict:
    return {
        "nodes": [
            {
                "parameters": {},
                "type": N8N_CHAT_TRIGGER,
                "name": "Input",
                "id": "input-node",
                "typeVersion": 1,
            },
            {
                "parameters": {"pythonCode": "return 'hi'"},
                "type": "n8n-nodes-base.code",
                "name": "Code",
                "id": "code-node",
                "typeVersion": 1,
                "notes": notes,
            },
        ],
        "connections": {
            "Input": {"main": [[{"node": "Code", "type": "main", "index": 0}]]}
        },
    }


def test_n8n_code_node_image_reaches_the_op() -> None:
    workflow = _code_node_workflow('{"image":"my/lambda:1"}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    spec = parse_n8n_payload(payload)["graph"]
    graph = Graph.from_json(spec["graph"])
    from lumilake_server.ops import LambdaOp as _LambdaOp

    (op,) = [o for o in graph.iter_ops() if isinstance(o, _LambdaOp)]
    assert op.image == "my/lambda:1"


def test_n8n_code_node_expected_runtime_reaches_the_op() -> None:
    workflow = _code_node_workflow('{"expected-runtime-s": 5}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    spec = parse_n8n_payload(payload)["graph"]
    graph = Graph.from_json(spec["graph"])
    from lumilake_server.ops import LambdaOp as _LambdaOp

    (op,) = [o for o in graph.iter_ops() if isinstance(o, _LambdaOp)]
    assert op.expected_runtime_s == 5


def test_n8n_code_node_mode_reaches_the_op() -> None:
    workflow = _code_node_workflow('{"mode":"list"}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    spec = parse_n8n_payload(payload)["graph"]
    graph = Graph.from_json(spec["graph"])
    from lumilake_server.ops import LambdaOp as _LambdaOp

    (op,) = [o for o in graph.iter_ops() if isinstance(o, _LambdaOp)]
    assert op.mode == "list"


def test_n8n_code_node_rejects_an_unknown_mode() -> None:
    workflow = _code_node_workflow('{"mode":"rowwise"}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    with pytest.raises(ValueError, match="'mode'"):
        parse_n8n_payload(payload)


def test_n8n_code_node_limits_reach_the_op() -> None:
    workflow = _code_node_workflow(
        '{"image":"my/lambda:1","timeout-s":1800,"memory-mb":4096}'
    )
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    spec = parse_n8n_payload(payload)["graph"]
    graph = Graph.from_json(spec["graph"])
    from lumilake_server.ops import LambdaOp as _LambdaOp

    (op,) = [o for o in graph.iter_ops() if isinstance(o, _LambdaOp)]
    assert op.timeout_s == 1800
    assert op.memory_mb == 4096


def test_n8n_code_node_rejects_a_bad_timeout() -> None:
    workflow = _code_node_workflow('{"timeout-s": "long"}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    with pytest.raises(ValueError, match="timeout-s"):
        parse_n8n_payload(payload)


def test_n8n_code_node_timeout_without_image_is_capped() -> None:
    workflow = _code_node_workflow('{"timeout-s": 1800}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    with pytest.raises(ValueError, match="timeout_s"):
        Graph.from_json(parse_n8n_payload(payload)["graph"]["graph"])


def test_n8n_code_node_rejects_a_bad_expected_runtime() -> None:
    workflow = _code_node_workflow('{"expected-runtime-s": "soon"}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    with pytest.raises(ValueError, match="expected-runtime-s"):
        parse_n8n_payload(payload)


def test_n8n_code_node_cache_reaches_the_op() -> None:
    workflow = _code_node_workflow('{"cache": true}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    spec = parse_n8n_payload(payload)["graph"]
    graph = Graph.from_json(spec["graph"])
    from lumilake_server.ops import LambdaOp as _LambdaOp

    (op,) = [o for o in graph.iter_ops() if isinstance(o, _LambdaOp)]
    assert op.cache is True


def test_n8n_code_node_rejects_a_non_bool_cache() -> None:
    workflow = _code_node_workflow('{"cache": "yes"}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    with pytest.raises(ValueError, match="cache"):
        parse_n8n_payload(payload)


def test_n8n_code_node_rejects_an_empty_image() -> None:
    workflow = _code_node_workflow('{"image":"  "}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    with pytest.raises(ValueError, match="image"):
        parse_n8n_payload(payload)


def test_n8n_code_node_is_output_emits_output_op() -> None:
    workflow = _code_node_workflow(
        '{"is-output": true, "output-field": "selected_papers"}'
    )
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    spec = parse_n8n_payload(payload)["graph"]
    graph = Graph.from_json(spec["graph"])
    output_ops = list(graph.iter_ops(OutputOp))
    assert len(output_ops) == 1
    assert output_ops[0].name == "selected_papers"


def test_n8n_code_node_is_output_defaults_to_node_name() -> None:
    workflow = _code_node_workflow('{"is-output": true}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    spec = parse_n8n_payload(payload)["graph"]
    graph = Graph.from_json(spec["graph"])
    output_ops = list(graph.iter_ops(OutputOp))
    assert len(output_ops) == 1
    assert output_ops[0].name == "Code"


def test_n8n_code_node_rejects_a_non_bool_is_output() -> None:
    workflow = _code_node_workflow('{"is-output": "yes"}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    with pytest.raises(ValueError, match="is-output"):
        parse_n8n_payload(payload)


def test_n8n_code_node_rejects_an_empty_output_field() -> None:
    workflow = _code_node_workflow('{"is-output": true, "output-field": "  "}')
    payload = {
        "graphs": [
            {"name": "graph", "workflow": workflow, "inputs": {"Input": ["NVDA"]}}
        ]
    }
    with pytest.raises(ValueError, match="output-field"):
        parse_n8n_payload(payload)


def test_rowwise_prompt_escapes_literal_braces() -> None:
    """A rowwise prompt with a literal JSON example renders verbatim."""
    template = _build_rowwise_prompt_content(
        "Return JSON like {\"population\": 5} for {{ $('SQL').item.synopsis }}"
    )
    rendered = template.format_map({"synopsis": "the value"})
    assert rendered == 'Return JSON like {"population": 5} for the value'


def test_prompt_content_escapes_literal_braces_with_ref() -> None:
    """A non-rowwise prompt with a ref and a JSON example renders verbatim."""
    content, extra_ops = _build_prompt_content(
        "Return JSON like {\"population\": 5} for {{ $('SQL') }}",
        node_name="Chain",
        op_ids={"SQL": "op_sql"},
        upstream_main=None,
        scope="g0",
    )
    assert len(extra_ops) == 1
    fmt_op = extra_ops[0]
    assert fmt_op["_op"] == "FormatOp"
    rendered = fmt_op["template"].format(**fmt_op["format_kwargs"])
    assert rendered == 'Return JSON like {"population": 5} for op_sql'


def test_prompt_content_no_ref_escapes_literal_braces() -> None:
    """A no-ref prompt is a graph_template message, so literal braces are
    escaped and the message renders verbatim via format_map."""
    text = 'Return JSON like {"population": 5}'
    content, extra_ops = _build_prompt_content(
        text,
        node_name="Chain",
        op_ids={},
        upstream_main=None,
        scope="g0",
    )
    assert extra_ops == []
    assert content == 'Return JSON like {{"population": 5}}'
    assert content.format_map({}) == text


def test_prompt_content_no_ref_unescaped_when_not_format_rendered() -> None:
    """Image-generation content is not format-rendered, so it stays verbatim."""
    text = 'Return JSON like {"population": 5}'
    content, extra_ops = _build_prompt_content(
        text,
        node_name="Chain",
        op_ids={},
        upstream_main=None,
        scope="g0",
        escape=False,
    )
    assert extra_ops == []
    assert content == text


def test_prompt_content_image_ref_escapes_literal_braces() -> None:
    """An image-generation prompt with a ref and literal JSON braces escapes
    the braces, so the FormatOp renders without a KeyError."""
    content, extra_ops = _build_prompt_content(
        'Draw {"kind": "chart"} from {{ $(\'Render Prompt\') }}',
        node_name="Chain",
        op_ids={"Render Prompt": "op_render"},
        upstream_main=None,
        scope="g0",
        escape=False,
    )
    assert len(extra_ops) == 1
    fmt_op = extra_ops[0]
    assert fmt_op["_op"] == "FormatOp"
    rendered = fmt_op["template"].format(**fmt_op["format_kwargs"])
    assert rendered == 'Draw {"kind": "chart"} from op_render'


def test_prompt_content_image_ref_escapes_literal_df() -> None:
    """An image-generation prompt with a ref and a literal {df} escapes it,
    so the FormatOp renders without a KeyError for 'df'."""
    content, extra_ops = _build_prompt_content(
        "Draw {df} from {{ $('Render Prompt') }}",
        node_name="Chain",
        op_ids={"Render Prompt": "op_render"},
        upstream_main=None,
        scope="g0",
        escape=False,
    )
    assert len(extra_ops) == 1
    fmt_op = extra_ops[0]
    assert fmt_op["_op"] == "FormatOp"
    rendered = fmt_op["template"].format(**fmt_op["format_kwargs"])
    assert rendered == "Draw {df} from op_render"


def test_prompt_content_literal_df_escaped_with_ref() -> None:
    """A literal {df} in a non-aggregate prompt is escaped, so the FormatOp
    renders without a KeyError for 'df'."""
    content, extra_ops = _build_prompt_content(
        "Use the literal token {df}; context: {{ $('SQL') }}",
        node_name="Chain",
        op_ids={"SQL": "op_sql"},
        upstream_main=None,
        scope="g0",
    )
    assert len(extra_ops) == 1
    fmt_op = extra_ops[0]
    assert fmt_op["_op"] == "FormatOp"
    rendered = fmt_op["template"].format(**fmt_op["format_kwargs"])
    assert rendered == "Use the literal token {df}; context: op_sql"


def test_aggregate_prompt_escapes_literal_braces() -> None:
    """An aggregate prompt with a JSON example and a table ref renders
    verbatim through the graph_template message format_map."""
    prompt = (
        'Summarize the table and return JSON {"population": 5}.\n'
        "```table\nsummary: {{ $('SQL').item.summary }}\n```"
    )
    aggregate_prompt, table_spec = _build_aggregate_prompt_content(
        prompt,
        op_ids={"SQL": "op_sql"},
        upstream_main=None,
        node_map={"SQL": {"type": "n8n-nodes-base.postgres"}},
    )
    assert table_spec
    rendered = aggregate_prompt.format_map({"df": '{"summary": "the value"}'})
    assert rendered == (
        'Summarize the table and return JSON {"population": 5}.\n'
        '{"summary": "the value"}'
    )


def test_aggregate_prompt_no_table_escapes_literal_braces() -> None:
    """An aggregate prompt without a table block is still a graph_template
    message and escapes literal braces."""
    aggregate_prompt, table_spec = _build_aggregate_prompt_content(
        'Return JSON {"population": 5}',
        op_ids={},
        upstream_main=None,
        node_map={},
    )
    assert table_spec == []
    assert aggregate_prompt == 'Return JSON {{"population": 5}}'
    assert aggregate_prompt.format_map({}) == 'Return JSON {"population": 5}'


def test_prompt_content_field_ref_records_runtime_path() -> None:
    """A plain prompt field ref keeps the field path on the FormatOp."""
    content, extra_ops = _build_prompt_content(
        "Q: {{ $('Protocol Details').item.question }}",
        node_name="Chain",
        op_ids={"Protocol Details": "op_pd"},
        upstream_main=None,
        scope="g0",
    )
    assert len(extra_ops) == 1
    fmt_op = extra_ops[0]
    assert fmt_op["_op"] == "FormatOp"
    assert fmt_op["format_kwargs"] == {"ref0": "op_pd"}
    assert fmt_op["format_paths"] == {"ref0": "items.output.question"}
    assert fmt_op["template"] == "Q: {ref0}"


def test_prompt_content_item_output_field_ref_records_runtime_path() -> None:
    """An ``item.output.<field>`` ref binds the field path, not a doubled
    ``output`` segment."""
    content, extra_ops = _build_prompt_content(
        "Q: {{ $('Criteria Parser').item.output.population }}",
        node_name="Chain",
        op_ids={"Criteria Parser": "op_cp"},
        upstream_main=None,
        scope="g0",
    )
    assert len(extra_ops) == 1
    fmt_op = extra_ops[0]
    assert fmt_op["format_kwargs"] == {"ref0": "op_cp"}
    assert fmt_op["format_paths"] == {"ref0": "items.output.population"}
    assert fmt_op["template"] == "Q: {ref0}"


def test_prompt_content_whole_item_ref_has_no_path() -> None:
    """A whole-item ref keeps today's FormatOp behaviour with no field path."""
    content, extra_ops = _build_prompt_content(
        "Q: {{ $('Protocol Details').item }}",
        node_name="Chain",
        op_ids={"Protocol Details": "op_pd"},
        upstream_main=None,
        scope="g0",
    )
    assert len(extra_ops) == 1
    fmt_op = extra_ops[0]
    assert fmt_op["format_kwargs"] == {"ref0": "op_pd"}
    assert fmt_op.get("format_paths") in (None, {})


def _aggregate_chain_node(text: str) -> dict:
    return {
        "parameters": {
            "text": text,
            "messages": {"messageValues": []},
        },
        "type": "@n8n/n8n-nodes-langchain.chainLlm",
        "name": "Digest",
        "notes": '{"op-type": "text-generation", "mode": "aggregate"}',
    }


def _aggregate_message_content(ops: list[dict]) -> str:
    """The user message content of the aggregate chain's MessageOp."""
    for op in ops:
        if op.get("_op") == "MessageOp":
            for msg in op.get("messages") or []:
                if msg.get("role") == "user":
                    return msg["content"]
    raise AssertionError("no user MessageOp found")


def _render_aggregate_format_ops(
    ops: list[dict],
    df_value: str,
    ref_values: dict[str, str] | None = None,
) -> list[str]:
    """Render every FormatOp in the chain ops the way the runtime does, with
    the aggregate {df} column and any ref columns supplied."""
    rendered = []
    for op in ops:
        if op.get("_op") != "FormatOp":
            continue
        kwargs = dict(op.get("format_kwargs") or {})
        kwargs["df"] = df_value
        for key, value in (ref_values or {}).items():
            kwargs[key] = value
        rendered.append(op["template"].format(**kwargs))
    return rendered


def test_aggregate_chain_no_ref_escapes_once() -> None:
    """An aggregate prompt with a table block but no n8n ref escapes literal
    braces exactly once, so the message renders the JSON literally."""
    node = _aggregate_chain_node(
        'Return JSON {"a": 1}.\n```table\nx: {{ $json.x }}\n```'
    )
    ops, _ = _make_chain_ops(
        node=node,
        node_name="Digest",
        incoming={"Digest": [{"type": "ai_languageModel", "node": "Model"}]},
        node_map={},
        op_ids={},
        inputs={},
        model_specs={"Model": {"config": {"model": "m"}, "vlm_config": None}},
        scope="g0",
    )
    content = _aggregate_message_content(ops)
    assert content == 'Return JSON {{"a": 1}}.\n{df}'
    assert content.format_map({"df": '{"x": "v"}'}) == (
        'Return JSON {"a": 1}.\n{"x": "v"}'
    )


def test_aggregate_chain_with_ref_escapes_once() -> None:
    """An aggregate prompt with a table block and an n8n ref escapes literal
    braces once and keeps {df} live, rendering the JSON literally."""
    node = _aggregate_chain_node(
        "Return JSON {\"a\": 1} for {{ $('Stock') }}.\n"
        "```table\nx: {{ $('SQL').item.x }}\n```"
    )
    ops, _ = _make_chain_ops(
        node=node,
        node_name="Digest",
        incoming={
            "Digest": [{"type": "ai_languageModel", "node": "Model"}],
        },
        node_map={"SQL": {"type": N8N_POSTGRES_NODE}},
        op_ids={"Stock": "op_stock", "SQL": "op_sql"},
        inputs={},
        model_specs={"Model": {"config": {"model": "m"}, "vlm_config": None}},
        scope="g0",
    )
    rendered = _render_aggregate_format_ops(
        ops, df_value='{"x": "v"}', ref_values={"ref0": "NVDA"}
    )
    assert rendered == ['Return JSON {"a": 1} for NVDA.\n{"x": "v"}']
