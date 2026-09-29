"""A LambdaOp that no LLM consumes compiles to its own FlowMesh python task."""

import json
from pathlib import Path
from typing import Any

import pytest
from lumilake import envs

from lumilake_server.common import GenerationConfig
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    FormatOp,
    LambdaOp,
    LLMChatOp,
    OpMessage,
    as_output,
    input_placeholder,
)
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime import python_step
from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager

_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _python_nodes(graph: Any) -> dict[str, Any]:
    return {k: n for k, n in graph.nodes.items() if n.task_type == "python"}


def test_lambda_over_input_is_a_standalone_python_task() -> None:
    stock = input_placeholder("Stock")
    lower = LambdaOp([stock], fn=lambda inputs: str(inputs[0]).lower())
    compiled = Graph.from_ops([as_output("lowered", lower)]).compile(
        Stock=["NVDA", "AAPL"]
    )
    graph = RuntimeGraphBuilder().build(compiled)

    (node,) = _python_nodes(graph).values()
    assert node.dependencies == ()
    assert graph.output_node_map == {node.node_id: "lowered"}
    spec = node.to_flowmesh_node()["spec"]
    assert spec["taskType"] == "python"
    assert spec["entrypoint"] == "main"
    assert "inputs" not in spec and "data" not in spec


def test_lambda_over_llm_reads_the_llm_stage() -> None:
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model=_MODEL),
    )
    shout = LambdaOp([llm], fn=lambda inputs: str(inputs[0]).upper())
    compiled = Graph.from_ops([as_output("shouted", shout)]).compile(Stock=["NVDA"])
    graph = RuntimeGraphBuilder().build(compiled)

    node = _python_nodes(graph)[shout.id]
    assert node.dependencies == (llm.id,)
    fm = node.to_flowmesh_node()
    assert fm["dependsOn"] == [llm.id]
    assert fm["spec"]["inputs"] == [{"stage": llm.id}]
    assert graph.topological_order().index(llm.id) < graph.topological_order().index(
        shout.id
    )


def test_lambda_chain_compiles_both_steps() -> None:
    stock = input_placeholder("Stock")
    first = LambdaOp([stock], fn=lambda inputs: str(inputs[0]).lower())
    second = LambdaOp([first], fn=lambda inputs: str(inputs[0]) + "!")
    compiled = Graph.from_ops([as_output("out", second)]).compile(Stock=["NVDA"])
    graph = RuntimeGraphBuilder().build(compiled)

    nodes = _python_nodes(graph)
    assert set(nodes) == {first.id, second.id}
    assert nodes[second.id].dependencies == (first.id,)


def test_lambda_read_by_llm_stays_inlined() -> None:
    stock = input_placeholder("Stock")
    greeting = FormatOp("Hello, {name}!", name=stock)
    shout = LambdaOp([greeting], fn=lambda inputs: str(inputs[0]).upper())
    llm = LLMChatOp(
        [OpMessage(role="user", content=shout)],
        config=GenerationConfig(model=_MODEL),
    )
    compiled = Graph.from_ops([as_output("result", llm)]).compile(Stock=["NVDA"])
    graph = RuntimeGraphBuilder().build(compiled)
    assert _python_nodes(graph) == {}


def test_documented_yaml_example_builds() -> None:
    """docs/OPS.md's LambdaOp example: a LambdaOp as the workflow output."""
    payload = """
name: lambda-output
inputs:
  Stock: ["NVDA"]
ops:
  - id: "Lowercase"
    op: LambdaOp
    inputs: [Stock]
    fn_name: lowercase
    code: |
      def lowercase(inputs: tuple[str, ...]) -> str:
          (symbol,) = inputs
          return symbol.lower()
outputs:
  - name: lowercased
    ref: "Lowercase"
"""
    spec = parse_yaml_payload(payload)["lambda-output"]
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    graph = RuntimeGraphBuilder().build(compiled)
    assert len(_python_nodes(graph)) == 1


# ------------------------------------------------------------------ #
# The generated wrapper, executed
# ------------------------------------------------------------------ #


def _run_wrapper(code: str, plan: python_step.ColumnPlan, inputs: dict[str, str]):
    namespace: dict[str, Any] = {}
    exec(python_step.wrapper_source("op-1", code, plan), namespace)
    return namespace["main"](inputs)


def _stage(tmp_path: Path, name: str, result: dict[str, Any]) -> str:
    stage = tmp_path / name
    stage.mkdir()
    (stage / "results.json").write_text(json.dumps({"task_id": "t", "result": result}))
    return str(stage)


def test_wrapper_applies_fn_per_row_and_broadcasts(tmp_path: Path) -> None:
    llm_dir = _stage(tmp_path, "llm", {"items": [{"output": "a"}, {"output": "b"}]})
    out = _run_wrapper(
        "def join(inputs):\n    return '-'.join(inputs)\n",
        [{"kind": "stage", "name": "llm"}, {"kind": "literal", "values": ["x"]}],
        {"llm": llm_dir},
    )
    assert out == {"items": [{"output": "a-x"}, {"output": "b-x"}]}


def test_wrapper_reads_python_and_api_upstreams(tmp_path: Path) -> None:
    py_dir = _stage(tmp_path, "py", {"value": {"items": [{"output": "p"}]}})
    api_dir = _stage(tmp_path, "api", {"text": "t"})
    out = _run_wrapper(
        "lambda inputs: inputs[0] + inputs[1]",
        [{"kind": "stage", "name": "py"}, {"kind": "stage", "name": "api"}],
        {"py": py_dir, "api": api_dir},
    )
    assert out == {"items": [{"output": "pt"}]}


def test_wrapper_rejects_misaligned_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 1 or 3"):
        _run_wrapper(
            "lambda inputs: inputs[0]",
            [
                {"kind": "literal", "values": ["a", "b", "c"]},
                {"kind": "literal", "values": ["a", "b"]},
            ],
            {},
        )


def test_wrapper_requires_one_function() -> None:
    with pytest.raises(ValueError, match="exactly one function"):
        _run_wrapper("def a(x):\n    return x\ndef b(x):\n    return x\n", [], {})


# ------------------------------------------------------------------ #
# Scheduling and result collection
# ------------------------------------------------------------------ #


def test_halo_maps_python_to_its_own_cpu_engine() -> None:
    halo = HaloOptimizer.__new__(HaloOptimizer)
    assert halo._map_engine("python", "python") == "python"


def test_python_output_items_come_from_value() -> None:
    manager = FlowmeshRuntimeManager.__new__(FlowmeshRuntimeManager)
    items = manager._resolve_output_items(
        {"task_type": "python", "value": {"items": [{"output": "x"}]}},
        "node",
        task_type="python",
    )
    assert items == [{"output": "x"}]
    with pytest.raises(RuntimeError, match="returned no items"):
        manager._resolve_output_items({"value": {}}, "node", task_type="python")
