"""A LambdaOp that no LLM consumes compiles to its own FlowMesh python task."""

import json
from pathlib import Path
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.common import GenerationConfig, Message
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
from lumilake_server.runtime.optimizer.schedule.models import Node
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.runtime_manager.flowmesh import FlowmeshRuntimeManager
from lumilake_server.utils import lambda_runtime

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


def _run_wrapper(
    code: str,
    plan: python_step.ColumnPlan,
    inputs: dict[str, str],
    fn_name: str | None = None,
):
    """Run the generated wrapper; ``fn_name`` defaults to the name the code defines."""
    declared = fn_name or lambda_runtime.validate_source(code)
    namespace: dict[str, Any] = {}
    exec(python_step.wrapper_source("op-1", declared, code, plan), namespace)
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
        [{"kind": "stage", "node": "llm"}, {"kind": "literal", "values": ["x"]}],
        {"llm": llm_dir},
    )
    assert out == {"items": [{"output": "a-x"}, {"output": "b-x"}]}


def test_wrapper_reads_python_and_api_upstreams(tmp_path: Path) -> None:
    py_dir = _stage(tmp_path, "py", {"value": {"items": [{"output": "p"}]}})
    api_dir = _stage(tmp_path, "api", {"text": "t"})
    out = _run_wrapper(
        "lambda inputs: inputs[0] + inputs[1]",
        [{"kind": "stage", "node": "py"}, {"kind": "stage", "node": "api"}],
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


def test_wrapper_calls_the_declared_function_or_rejects_the_mismatch() -> None:
    code = "def a(inputs):\n    return 'a'\n"
    plan: python_step.ColumnPlan = [{"kind": "literal", "values": ["x"]}]
    assert _run_wrapper(code, plan, {}, fn_name="a") == {"items": [{"output": "a"}]}
    with pytest.raises(ValueError, match="no top-level function named 'b'"):
        _run_wrapper(code, plan, {}, fn_name="b")


def test_wrapper_treats_fn_name_as_a_label_for_a_lambda() -> None:
    plan: python_step.ColumnPlan = [{"kind": "literal", "values": ["x"]}]
    out = _run_wrapper("lambda inputs: inputs[0]", plan, {}, fn_name="anything")
    assert out == {"items": [{"output": "x"}]}


def test_wrapper_requires_a_function() -> None:
    with pytest.raises(ValueError, match="no top-level function named 'a'"):
        _run_wrapper("X = 1\n", [], {}, fn_name="a")


_PLAN_X: python_step.ColumnPlan = [{"kind": "literal", "values": ["x"]}]


@pytest.mark.parametrize(
    "code, output",
    [
        # A stdlib import inside the function body.
        ("def main(Dummy):\n    import math\n    return math.sqrt(16)\n", "4.0"),
        ("def main(Dummy):\n    import os\n    return os.sep\n", "/"),
        # ... and at module level, before the function.
        ("import os\ndef main(Dummy):\n    return os.sep\n", "/"),
        (
            "from collections import Counter\nN = 2\n"
            "def helper(s):\n    return Counter(s)['x'] * N\n"
            "def main(row):\n    return str(helper(row[0]))\n",
            "2",
        ),
        (
            "def main(row):\n    import datetime, statistics\n"
            "    return str(statistics.mean([1, 3]))\n",
            "2",
        ),
    ],
)
def test_wrapper_runs_stdlib_imports_and_a_preamble(code: str, output: str) -> None:
    out = _run_wrapper(code, _PLAN_X, {}, fn_name="main")
    assert out == {"items": [{"output": output}]}


def test_wrapper_keeps_the_restricted_namespace() -> None:
    out = _run_wrapper(
        "def f(inputs):\n    return json.dumps([math.floor(2.5), len(inputs)])",
        [{"kind": "literal", "values": ["x"]}],
        {},
    )
    assert out == {"items": [{"output": "[2, 1]"}]}
    with pytest.raises(NameError, match="open"):
        _run_wrapper("def f(inputs):\n    return open('/etc/passwd').read()", [], {})
    with pytest.raises(ImportError, match="'requests' is not available"):
        _run_wrapper("def f(inputs):\n    return __import__('requests')", [], {})


def test_wrapper_imports_numpy_and_pandas_only_when_named() -> None:
    out = _run_wrapper(
        "def f(inputs):\n    return str(np.array([1, 2]).sum() + len(pd.Series([1])))",
        [],
        {},
    )
    assert out == {"items": [{"output": "4"}]}
    with pytest.raises(ImportError, match="'requests'"):
        _run_wrapper(
            "def f(inputs):\n    np\n    return __import__('requests')", [], {}
        )


# ------------------------------------------------------------------ #
# Node prefixes, limits and scheduling
# ------------------------------------------------------------------ #


def _llm_then_lambda(**lambda_kwargs: Any) -> Any:
    stock = input_placeholder("Stock")
    llm = LLMChatOp(
        [OpMessage(role="user", content=stock)],
        config=GenerationConfig(model=_MODEL),
    )
    shout = LambdaOp(
        [llm, stock], fn=lambda inputs: str(inputs[0]).upper(), **lambda_kwargs
    )
    compiled = Graph.from_ops([as_output("shouted", shout)]).compile(Stock=["NVDA"])
    return llm, shout, compiled


def test_prefixed_python_step_mounts_the_prefixed_stage() -> None:
    llm, shout, compiled = _llm_then_lambda()
    graph = RuntimeGraphBuilder().build(compiled, node_prefix="job1")

    (llm_id,) = graph.dsl_to_runtime[llm.id]
    (py_id,) = graph.dsl_to_runtime[shout.id]
    assert llm_id != llm.id and py_id != shout.id
    node = graph.nodes[py_id]
    assert node.dependencies == (llm_id,)

    fm = node.to_flowmesh_node()
    assert fm["dependsOn"] == [llm_id]
    assert fm["spec"]["inputs"] == [{"stage": llm_id}]
    code = fm["spec"]["code"]
    assert llm_id in code
    assert llm.id not in code.replace(llm_id, "")


def test_python_step_carries_the_declared_fn_name() -> None:
    def shout(inputs: tuple[str | list[Message], ...]) -> str:
        return str(inputs[0]).upper()

    stock = input_placeholder("Stock")
    op = LambdaOp([stock], fn=shout)
    compiled = Graph.from_ops([as_output("shouted", op)]).compile(Stock=["NVDA"])
    node = RuntimeGraphBuilder().build(compiled).nodes[op.id]

    assert node.data_spec["fn_name"] == "shout"
    code = node.to_flowmesh_node()["spec"]["code"]
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    assert namespace["main"]({}) == {"items": [{"output": "NVDA"}]}


def test_python_step_timeout_and_memory_reach_the_task() -> None:
    _, shout, compiled = _llm_then_lambda(timeout_s=45, memory_mb=512)
    graph = RuntimeGraphBuilder().build(compiled)
    spec = graph.nodes[shout.id].to_flowmesh_node()["spec"]
    assert spec["timeoutSeconds"] == 45
    assert spec["resources"] == {"hardware": {"memory": "512Mi"}}


def test_python_step_defaults_carry_the_default_timeout() -> None:
    _, shout, compiled = _llm_then_lambda()
    node = RuntimeGraphBuilder().build(compiled).nodes[shout.id]
    assert node.data_spec["timeout_s"] == python_step.DEFAULT_TIMEOUT_SECONDS
    spec = node.to_flowmesh_node()["spec"]
    assert spec["timeoutSeconds"] == python_step.DEFAULT_TIMEOUT_SECONDS
    assert "resources" not in spec


@pytest.mark.parametrize("kwargs", [{"timeout_s": 0}, {"timeout_s": 601}])
def test_lambda_op_rejects_a_bad_timeout(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        LambdaOp([["x"]], fn=lambda inputs: str(inputs[0]), **kwargs)


@pytest.mark.parametrize("value", [1, 10**9, True])
def test_lambda_op_rejects_a_bad_memory_limit(value: Any) -> None:
    with pytest.raises(ValueError, match="memory_mb"):
        LambdaOp([["x"]], fn=lambda inputs: str(inputs[0]), memory_mb=value)


def test_halo_costs_a_default_python_step_at_five_percent_of_its_timeout() -> None:
    def cost(raw: dict[str, Any]) -> float:
        node = Node(id="n", type="python", engine="python", model="", raw=raw)
        return HaloOptimizer._python_exec_cost(node)

    assert cost({}) == 0.05 * python_step.DEFAULT_TIMEOUT_SECONDS == 30.0
    assert cost({"timeout_s": 100}) == 5.0


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
