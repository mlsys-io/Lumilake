import textwrap

import pytest

from lumilake import envs
from lumilake_server.common import ApiConfig, GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    LambdaOp,
    LLMChatOp,
    OpMessage,
    as_output,
    data,
    input_placeholder,
)
from lumilake_server.parser.yaml_parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraph, RuntimeGraphBuilder

_LUMID_URL = "http://lumid-data"
_LUMID_TOKEN = "test-token"
_RUNTIME_TOKEN = "test" + "-pat"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", _LUMID_URL)
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", _LUMID_TOKEN)
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", _RUNTIME_TOKEN)


def _build(yaml_text: str) -> RuntimeGraph:
    specs = parse_yaml_payload(yaml_text)
    spec = next(iter(specs.values()))
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    return RuntimeGraphBuilder().build(compiled)


def _explode_fn(items: tuple[str, ...]) -> list[dict[str, str]]:
    return [{"value": v} for v in items[0]]


def test_list_lambda_explodes_input_feeds_rowwise_llm() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_rowwise
        inputs:
          stock: ["NVDA", "AAPL"]
        ops:
          - id: explode
            op: LambdaOp
            inputs: [stock]
            fn_name: explode
            code: |
              def explode(items):
                  return [{"value": v} for v in items[0]]
            mode: list
          - id: summarize
            op: LLMChatOp
            inputs: [explode]
            messages:
              - role: user
                content: "hello"
            rowwise_template: "Summarize {value}"
            rowwise_columns:
              - { label: value, node: explode, path: "items.output.value" }
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
        outputs:
          - { name: out, ref: summarize }
        """)
    runtime_graph = _build(yaml_text)

    explode_node = next(
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    )
    assert explode_node.data_spec["mode"] == "list"
    assert explode_node.data_spec["plan"] == [
        {"kind": "literal", "values": ["NVDA", "AAPL"]}
    ]

    summarize_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" in n.data_spec
    )
    assert explode_node.node_id in summarize_node.dependencies
    columns = summarize_node.data_spec["columns"]
    assert {
        "label": "value",
        "node": explode_node.node_id,
        "path": "value.items.output.value",
    } in columns


def test_list_lambda_groups_then_collapses() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_groups_collapse
        inputs:
          stock: ["NVDA", "AAPL"]
        ops:
          - id: group
            op: LambdaOp
            inputs: [stock]
            fn_name: group
            code: |
              def group(items):
                  return [[{"value": v}] for v in items[0]]
            mode: list
          - id: summarize
            op: LLMChatOp
            inputs: [group]
            messages:
              - role: user
                content: "hello"
            rowwise_template: "Summarize {value}"
            rowwise_columns:
              - { label: value, node: group, path: "items.output.value" }
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
          - id: collapse
            op: LambdaOp
            inputs: [summarize]
            fn_name: collapse
            code: |
              def collapse(items):
                  return [{"summary": items[0]}]
            mode: list
        outputs:
          - { name: out, ref: collapse }
        """)
    runtime_graph = _build(yaml_text)

    python_nodes = [n for n in runtime_graph.nodes.values() if n.task_type == "python"]
    assert len(python_nodes) == 2
    group_node = next(
        n
        for n in python_nodes
        if n.data_spec["plan"] == [{"kind": "literal", "values": ["NVDA", "AAPL"]}]
    )
    collapse_node = next(n for n in python_nodes if n is not group_node)
    summarize_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" in n.data_spec
    )
    (collapse_plan,) = collapse_node.data_spec["plan"]
    assert collapse_plan == {"kind": "stage", "node": summarize_node.node_id}
    assert summarize_node.node_id in collapse_node.dependencies
    assert collapse_node.node_id in runtime_graph.output_node_map


def test_list_lambda_as_workflow_output() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_output
        inputs:
          stock: ["NVDA", "AAPL"]
        ops:
          - id: explode
            op: LambdaOp
            inputs: [stock]
            fn_name: explode
            code: |
              def explode(items):
                  return [{"value": v} for v in items[0]]
            mode: list
        outputs:
          - { name: out, ref: explode }
        """)
    runtime_graph = _build(yaml_text)

    (explode_node,) = [
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    ]
    assert explode_node.data_spec["mode"] == "list"
    assert explode_node.node_id in runtime_graph.output_node_map
    assert runtime_graph.output_paths[explode_node.node_id] == "value.items.output"


def test_list_lambda_in_message_chain_raises() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_in_chain
        inputs:
          stock: ["NVDA"]
        ops:
          - id: explode
            op: LambdaOp
            inputs: [stock]
            fn_name: explode
            code: |
              def explode(items):
                  return [{"value": v} for v in items[0]]
            mode: list
          - id: summarize
            op: LLMChatOp
            inputs: [explode]
            messages:
              - role: user
                content: explode
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
        outputs:
          - { name: out, ref: summarize }
        """)
    with pytest.raises(ValueError, match="node column"):
        _build(yaml_text)


def test_per_row_lambda_behaviour_unchanged() -> None:
    yaml_text = textwrap.dedent("""
        name: per_row_lambda
        inputs:
          stock: ["NVDA"]
        ops:
          - id: tag
            op: LambdaOp
            inputs: [stock]
            fn_name: tag
            code: |
              def tag(items):
                  return "tagged:" + items[0]
          - id: summarize
            op: LLMChatOp
            inputs: [tag]
            messages:
              - role: user
                content: tag
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
        outputs:
          - { name: out, ref: summarize }
        """)
    runtime_graph = _build(yaml_text)

    assert not any(n.task_type == "echo" for n in runtime_graph.nodes.values())
    assert not any(n.task_type == "python" for n in runtime_graph.nodes.values())
    summarize_node = next(
        n for n in runtime_graph.nodes.values() if n.task_type == "inference"
    )
    template = summarize_node.data_spec["template"]
    steps = template["options"]["format"]["steps"]
    assert any(step.get("function") is not None for step in steps)


def test_list_lambda_over_llm_output_feeds_rowwise_llm() -> None:
    yaml_text = textwrap.dedent("""
        name: list_lambda_over_llm
        inputs:
          stock: ["NVDA"]
        ops:
          - id: extract
            op: LLMChatOp
            inputs: [stock]
            messages:
              - role: user
                content: "extract text"
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
          - id: explode
            op: LambdaOp
            inputs: [extract]
            fn_name: explode
            code: |
              def explode(items):
                  return [{"value": v} for v in items[0]]
            mode: list
          - id: summarize
            op: LLMChatOp
            inputs: [explode]
            messages:
              - role: user
                content: "hello"
            rowwise_template: "Summarize {value}"
            rowwise_columns:
              - { label: value, node: explode, path: "items.output.value" }
            config: { model: meta-llama/Llama-3.1-8B-Instruct }
        outputs:
          - { name: out, ref: summarize }
        """)
    runtime_graph = _build(yaml_text)

    explode_node = next(
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    )
    assert explode_node.data_spec["mode"] == "list"
    extract_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" not in n.data_spec
    )
    (extract_plan,) = explode_node.data_spec["plan"]
    assert extract_plan == {"kind": "stage", "node": extract_node.node_id}
    assert extract_node.node_id in explode_node.dependencies

    summarize_node = next(
        n
        for n in runtime_graph.nodes.values()
        if n.task_type == "inference" and "columns" in n.data_spec
    )
    assert explode_node.node_id in summarize_node.dependencies
    columns = summarize_node.data_spec["columns"]
    assert {
        "label": "value",
        "node": explode_node.node_id,
        "path": "value.items.output.value",
    } in columns

    order = {node_id: idx for idx, node_id in enumerate(runtime_graph.node_order)}
    assert order[extract_node.node_id] < order[explode_node.node_id]
    assert order[explode_node.node_id] < order[summarize_node.node_id]


def test_list_lambda_data_op_input_compiles() -> None:
    constant = data(["NVDA", "AAPL"])
    explode = LambdaOp([constant], _explode_fn, mode="list")  # type: ignore[arg-type]
    output = as_output("out", explode)
    compiled = Graph.from_ops([output]).compile()

    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (explode_node,) = [
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    ]
    assert explode_node.data_spec["plan"] == [
        {"kind": "literal", "values": ["NVDA", "AAPL"]}
    ]


def test_api_op_reads_list_lambda_at_value_items_output() -> None:
    """An API-backed rowwise LLMChatOp reading a list-mode Lambda must bind the
    column to the python stage at ``value.items.output``."""
    stock = input_placeholder("Stock")
    explode = LambdaOp([stock], _explode_fn, mode="list")  # type: ignore[arg-type]
    llm = LLMChatOp(
        [OpMessage(role="user", content="Summarize the rows.")],
        config=GenerationConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            api=ApiConfig(),
        ),
        rowwise_template="Summarize {value}.",
        rowwise_columns=[
            {"label": "value", "node": explode.id, "path": "items.output.value"}
        ],
    )
    llm.inputs.append(explode)
    output = as_output("result", llm)
    compiled = Graph.from_ops([output]).compile(Stock=["NVDA", "AAPL"])
    runtime_graph = RuntimeGraphBuilder().build(compiled)

    (explode_node,) = [
        n for n in runtime_graph.nodes.values() if n.task_type == "python"
    ]
    assert explode_node.data_spec["mode"] == "list"
    (llm_node,) = runtime_graph.dsl_to_runtime[llm.id]
    node = runtime_graph.nodes[llm_node]
    assert node.data_spec["columns"][0] == {
        "label": "value",
        "node": explode_node.node_id,
        "path": "value.items.output.value",
    }
