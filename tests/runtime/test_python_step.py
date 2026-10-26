"""A LambdaOp that no LLM consumes compiles to its own FlowMesh python task."""

import json
from pathlib import Path
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.common import GenerationConfig, Message
from lumilake_server.graphs import Graph
from lumilake_server.ops import (
    DataRetrievalOp,
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


def test_standalone_lambda_over_rowwise_llm_flattens_plan_entry() -> None:
    """A standalone (row-mode) Lambda reading a rowwise LLM must mark the plan
    entry ``flatten`` so the python step expands the grouped item back to one
    value per row."""
    stock = input_placeholder("Stock")
    shout = LambdaOp([stock], fn=lambda inputs: str(inputs[0]).upper())
    reply = LLMChatOp(
        [OpMessage(role="user", content="Reply {r}")],
        config=GenerationConfig(model=_MODEL),
        rowwise_template="Reply {r}",
        rowwise_columns=[{"label": "r", "node": shout.id, "path": "items.output"}],
    )
    reply.inputs.append(shout)
    tally = LambdaOp([reply], fn=lambda inputs: str(len(inputs[0])))
    compiled = Graph.from_ops([as_output("n", tally)]).compile(
        Stock=["NVDA", "AAPL", "MSFT"]
    )
    graph = RuntimeGraphBuilder().build(compiled)

    node = _python_nodes(graph)[tally.id]
    (reply_plan,) = node.data_spec["plan"]
    assert reply_plan == {"kind": "stage", "node": reply.id, "flatten": True}


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


def test_lambda_fed_to_llm_and_retrieval_runs_as_python_step() -> None:
    """A row-mode LambdaOp feeding both an LLM and a DataRetrievalOp must
    materialize as a python step so the retrieval node can reference it."""
    stock = input_placeholder("Stock")
    terms = LambdaOp([stock], fn=lambda inputs: str(inputs[0]).upper())
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT * FROM t WHERE q = :q",
            "params": [{"label": "q", "node": terms.id, "path": "items.output"}],
        },
        inputs=[terms],
    )
    llm = LLMChatOp(
        [
            OpMessage(role="user", content=terms),
            OpMessage(role="user", content=retrieval),
        ],
        config=GenerationConfig(model=_MODEL),
    )
    compiled = Graph.from_ops([as_output("result", llm)]).compile(Stock=["NVDA"])
    graph = RuntimeGraphBuilder().build(compiled)

    assert terms.id in _python_nodes(graph)
    assert graph.nodes[retrieval.id].dependencies == (terms.id,)


def test_lambda_named_only_by_retrieval_dependency_materializes() -> None:
    """A row-mode LambdaOp that is a DataRetrievalOp input (so it lands in the
    retrieval node's dependencies) but is not referenced by any data_spec must
    still materialize as a python step, so no FlowMesh dependsOn points at an
    unemitted node."""
    stock = input_placeholder("Stock")
    terms = LambdaOp([stock], fn=lambda inputs: str(inputs[0]).upper())
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT 1",
        },
        inputs=[terms],
    )
    llm = LLMChatOp(
        [
            OpMessage(role="user", content=terms),
            OpMessage(role="user", content=retrieval),
        ],
        config=GenerationConfig(model=_MODEL),
    )
    compiled = Graph.from_ops([as_output("result", llm)]).compile(Stock=["NVDA"])
    graph = RuntimeGraphBuilder().build(compiled)

    assert terms.id in _python_nodes(graph)
    assert graph.nodes[retrieval.id].dependencies == (terms.id,)


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


def test_row_mode_rejects_unequal_multi_row_inputs() -> None:
    a = input_placeholder("A")
    b = input_placeholder("B")
    combine = LambdaOp([a, b], fn=lambda inputs: str(inputs[0]) + str(inputs[1]))
    compiled = Graph.from_ops([as_output("out", combine)]).compile(
        A=["a1", "a2"], B=["b1", "b2", "b3"]
    )
    with pytest.raises(ValueError, match="different row counts"):
        RuntimeGraphBuilder().build(compiled)


def test_row_mode_accepts_one_and_multi_row_inputs() -> None:
    a = input_placeholder("A")
    b = input_placeholder("B")
    combine = LambdaOp([a, b], fn=lambda inputs: str(inputs[0]) + str(inputs[1]))
    compiled = Graph.from_ops([as_output("out", combine)]).compile(
        A=["a1", "a2", "a3"], B=["b"]
    )
    graph = RuntimeGraphBuilder().build(compiled)
    assert len(_python_nodes(graph)) == 1


def test_aligned_mode_rejects_unequal_multi_row_inputs() -> None:
    a = input_placeholder("A")
    b = input_placeholder("B")
    combine = LambdaOp(
        [a, b],
        fn=lambda inputs: [
            str(inputs[0][i]) + str(inputs[1][i]) for i in range(len(inputs[0]))
        ],
        mode="aligned",
    )
    compiled = Graph.from_ops([as_output("out", combine)]).compile(
        A=["a1", "a2"], B=["b1", "b2", "b3"]
    )
    with pytest.raises(ValueError, match="different row counts"):
        RuntimeGraphBuilder().build(compiled)


def test_list_mode_accepts_unequal_multi_row_inputs() -> None:
    a = input_placeholder("A")
    b = input_placeholder("B")
    combine = LambdaOp(
        [a, b],
        fn=lambda inputs: [
            str(inputs[0][i]) + str(inputs[1][i]) for i in range(len(inputs[0]))
        ],
        mode="list",
    )
    compiled = Graph.from_ops([as_output("out", combine)]).compile(
        A=["a1", "a2"], B=["b1", "b2", "b3"]
    )
    graph = RuntimeGraphBuilder().build(compiled)
    assert len(_python_nodes(graph)) == 1


def test_python_step_records_its_row_count_with_a_runtime_hint() -> None:
    a = input_placeholder("A")
    b = input_placeholder("B")
    row = LambdaOp(
        [a, b],
        fn=lambda inputs: str(inputs[0]) + str(inputs[1]),
        expected_runtime_s=2,
    )
    whole = LambdaOp(
        [a, b],
        fn=lambda inputs: [len(inputs[0]) + len(inputs[1])],
        mode="list",
        expected_runtime_s=2,
    )
    plain = LambdaOp([a, b], fn=lambda inputs: str(inputs[0]) + str(inputs[1]))
    compiled = Graph.from_ops(
        [as_output("row", row), as_output("whole", whole), as_output("plain", plain)]
    ).compile(A=["a1", "a2", "a3"], B=["b"])
    graph = RuntimeGraphBuilder().build(compiled)

    assert graph.nodes[row.id].data_spec["row_count"] == 3
    assert graph.nodes[whole.id].data_spec["row_count"] == 3
    assert "row_count" not in graph.nodes[plain.id].data_spec
    assert "row_count" not in graph.nodes[row.id].to_flowmesh_node()["spec"]


def test_python_step_omits_row_count_when_an_input_count_is_unknown() -> None:
    a = input_placeholder("A")
    inner = LambdaOp([a], fn=lambda inputs: list(inputs[0]), mode="list")
    outer = LambdaOp(
        [inner, a],
        fn=lambda inputs: [len(inputs[0])],
        mode="list",
        expected_runtime_s=2,
    )
    compiled = Graph.from_ops([as_output("out", outer)]).compile(A=["a1", "a2"])
    graph = RuntimeGraphBuilder().build(compiled)

    assert "row_count" not in graph.nodes[outer.id].data_spec


# ------------------------------------------------------------------ #
# The generated wrapper, executed
# ------------------------------------------------------------------ #


def _run_wrapper(
    code: str,
    plan: python_step.ColumnPlan,
    inputs: dict[str, str],
    fn_name: str | None = None,
    mode: str = "row",
):
    """Run the generated wrapper; ``fn_name`` defaults to the name the code defines."""
    declared = fn_name or lambda_runtime.validate_source(code)
    namespace: dict[str, Any] = {}
    exec(python_step.wrapper_source("op-1", declared, code, plan, mode), namespace)
    return namespace["main"](inputs)


def _run_reshape(
    mode: str,
    plan: python_step.ColumnPlan,
    inputs: dict[str, str],
    template: str = "{symbol}: {title}",
):
    """Run a generated flatten/regroup step, which never calls user code."""
    namespace: dict[str, Any] = {}
    if mode == "flatten":
        code = python_step.flatten_source("reshape", plan, template)
    else:
        code = python_step.regroup_source("reshape", plan)
    exec(code, namespace)
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
    assert out == {
        "items": [{"output": "a-x", "value": "a-x"}, {"output": "b-x", "value": "b-x"}]
    }


def test_wrapper_reads_python_and_api_upstreams(tmp_path: Path) -> None:
    py_dir = _stage(tmp_path, "py", {"value": {"items": [{"output": "p"}]}})
    api_dir = _stage(tmp_path, "api", {"text": "t"})
    out = _run_wrapper(
        "lambda inputs: inputs[0] + inputs[1]",
        [{"kind": "stage", "node": "py"}, {"kind": "stage", "node": "api"}],
        {"py": py_dir, "api": api_dir},
    )
    assert out == {"items": [{"output": "pt", "value": "pt"}]}


def test_wrapper_flattens_rowwise_llm_grouped_item(tmp_path: Path) -> None:
    # A rowwise LLM upstream groups its per-row outputs back into one item
    # (FlowMesh ``_populate_table``); the ``flatten`` plan flag must expand it
    # back to one value per row so a list-mode Lambda sees N values, not one.
    llm_dir = _stage(
        tmp_path,
        "llm",
        {"items": [{"output": ["r0", "r1", "r2"]}]},
    )
    out = _run_wrapper(
        "lambda inputs: inputs[0]",
        [{"kind": "stage", "node": "llm", "flatten": True}],
        {"llm": llm_dir},
        mode="list",
    )
    assert out == {"items": [{"output": "r0"}, {"output": "r1"}, {"output": "r2"}]}


def test_wrapper_flattens_api_rowwise_llm_rows_list_mode(tmp_path: Path) -> None:
    # An API rowwise LLM fans out over ``items.rows`` (one completion per row)
    # instead of the grouped ``items.output`` list a local rowwise LLM emits.
    # The ``flatten`` plan flag must expand each row's completion to one value
    # so a list-mode Lambda sees one value per row, not one outer dict.
    llm_dir = _stage(
        tmp_path,
        "llm",
        {
            "items": [
                {
                    "rows": [
                        {
                            "json": {"choices": [{"message": {"content": "r0"}}]},
                            "text": "r0",
                        },
                        {
                            "json": {"choices": [{"message": {"content": "r1"}}]},
                            "text": "r1",
                        },
                        {
                            "json": {"choices": [{"message": {"content": "r2"}}]},
                            "text": "r2",
                        },
                    ]
                }
            ]
        },
    )
    out = _run_wrapper(
        "lambda inputs: inputs[0]",
        [{"kind": "stage", "node": "llm", "flatten": True}],
        {"llm": llm_dir},
        mode="list",
    )
    assert out == {"items": [{"output": "r0"}, {"output": "r1"}, {"output": "r2"}]}


def test_wrapper_flattens_api_rowwise_llm_rows_standalone(tmp_path: Path) -> None:
    # A standalone (row-mode) Lambda over an API rowwise LLM must also expand
    # ``items.rows`` to one completion per row, not hand the Lambda a single
    # serialized outer dict.
    llm_dir = _stage(
        tmp_path,
        "llm",
        {
            "items": [
                {
                    "rows": [
                        {
                            "json": {"choices": [{"message": {"content": "r0"}}]},
                            "text": "r0",
                        },
                        {
                            "json": {"choices": [{"message": {"content": "r1"}}]},
                            "text": "r1",
                        },
                    ]
                }
            ]
        },
    )
    out = _run_wrapper(
        "lambda inputs: inputs[0]",
        [{"kind": "stage", "node": "llm", "flatten": True}],
        {"llm": llm_dir},
        mode="row",
    )
    assert out == {
        "items": [{"output": "r0", "value": "r0"}, {"output": "r1", "value": "r1"}]
    }


def test_wrapper_flatten_keeps_a_list_value_as_one_value(tmp_path: Path) -> None:
    # A per-row value that is itself a list must stay one value per row, not be
    # expanded further. The grouped item carries one list per row.
    llm_dir = _stage(
        tmp_path,
        "llm",
        {"items": [{"output": [["a", "b"], ["c", "d"]]}]},
    )
    out = _run_wrapper(
        "lambda inputs: inputs[0]",
        [{"kind": "stage", "node": "llm", "flatten": True}],
        {"llm": llm_dir},
        mode="list",
    )
    assert out == {"items": [{"output": ["a", "b"]}, {"output": ["c", "d"]}]}


def test_wrapper_flatten_multiple_grouped_items(tmp_path: Path) -> None:
    # Several FlowMesh tables/slices each group their per-row outputs into one
    # item; flattening must expand every item, not just the first.
    llm_dir = _stage(
        tmp_path,
        "llm",
        {
            "items": [
                {"output": ["r0", "r1"]},
                {"output": ["r2", "r3"]},
            ]
        },
    )
    out = _run_wrapper(
        "lambda inputs: inputs[0]",
        [{"kind": "stage", "node": "llm", "flatten": True}],
        {"llm": llm_dir},
        mode="list",
    )
    assert out == {
        "items": [
            {"output": "r0"},
            {"output": "r1"},
            {"output": "r2"},
            {"output": "r3"},
        ]
    }


def test_wrapper_list_mode_reads_retrieval_stage(tmp_path: Path) -> None:
    # A list-mode Lambda reading a DataRetrievalOp gets one column element per
    # retrieval item, matching the standalone path.
    ret_dir = _stage(
        tmp_path,
        "ret",
        {
            "items": [
                {"table": {"df": json.dumps({"title": {"0": "t0"}})}},
                {"table": {"df": json.dumps({"title": {"0": "t1"}})}},
            ]
        },
    )
    out = _run_wrapper(
        "def titles(inputs):\n"
        "    return [json.loads(v['table']['df'])['title']['0'] for v in inputs[0]]\n",
        [{"kind": "stage", "node": "ret"}],
        {"ret": ret_dir},
        mode="list",
        fn_name="titles",
    )
    assert out == {"items": [{"output": "t0"}, {"output": "t1"}]}


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
    assert _run_wrapper(code, plan, {}, fn_name="a") == {
        "items": [{"output": "a", "value": "a"}]
    }
    with pytest.raises(ValueError, match="no top-level function named 'b'"):
        _run_wrapper(code, plan, {}, fn_name="b")


def test_wrapper_list_mode_decode_rows_decodes_each_row() -> None:
    # The grouped-list materializer's decode_rows receives tuple(columns) (a
    # 1-tuple for its single JSON input) and must decode each row's JSON back
    # to its list, one group per row.
    code = (
        "def decode_rows(columns):\n" "    return [json.loads(v) for v in columns[0]]\n"
    )
    plan: python_step.ColumnPlan = [
        {"kind": "literal", "values": [json.dumps(["a", "b"]), json.dumps(["c"])]}
    ]
    out = _run_wrapper(code, plan, {}, fn_name="decode_rows", mode="list")
    assert out == {"items": [{"output": ["a", "b"]}, {"output": ["c"]}]}


def test_wrapper_aligned_mode_returns_one_value_per_row(tmp_path: Path) -> None:
    # An aligned lambda gets whole columns and returns a list, but must return
    # exactly one value per input row; the wrapper emits one item per row.
    code = (
        "def decode_rows(columns):\n" "    return [json.loads(v) for v in columns[0]]\n"
    )
    plan: python_step.ColumnPlan = [
        {"kind": "literal", "values": [json.dumps(["a", "b"]), json.dumps(["c"])]}
    ]
    out = _run_wrapper(code, plan, {}, fn_name="decode_rows", mode="aligned")
    assert out == {"items": [{"output": ["a", "b"]}, {"output": ["c"]}]}


def test_wrapper_aligned_mode_rejects_wrong_length(tmp_path: Path) -> None:
    # An aligned lambda that returns a different number of values than there are
    # rows is rejected by the wrapper.
    code = "def bad(columns):\n    return [columns[0][0]]\n"
    plan: python_step.ColumnPlan = [{"kind": "literal", "values": ["a", "b", "c"]}]
    with pytest.raises(
        ValueError, match="aligned lambda 'op-1' returned 1 values for 3 rows"
    ):
        _run_wrapper(code, plan, {}, fn_name="bad", mode="aligned")


def test_wrapper_aligned_mode_rejects_unequal_columns(tmp_path: Path) -> None:
    code = "def f(columns):\n    return columns[0]\n"
    plan: python_step.ColumnPlan = [
        {"kind": "literal", "values": ["a", "b", "c"]},
        {"kind": "literal", "values": ["x", "y"]},
    ]
    with pytest.raises(ValueError, match="expected 1 or 3"):
        _run_wrapper(code, plan, {}, fn_name="f", mode="aligned")


def test_wrapper_treats_fn_name_as_a_label_for_a_lambda() -> None:
    plan: python_step.ColumnPlan = [{"kind": "literal", "values": ["x"]}]
    out = _run_wrapper("lambda inputs: inputs[0]", plan, {}, fn_name="anything")
    assert out == {"items": [{"output": "x", "value": "x"}]}


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
    # ``output`` is the textified result; ``value`` carries the raw value.
    assert out["items"][0]["output"] == output
    assert out["items"][0]["value"] == (float(output) if output == "4.0" else output)


def test_wrapper_keeps_the_restricted_namespace() -> None:
    out = _run_wrapper(
        "def f(inputs):\n    return json.dumps([math.floor(2.5), len(inputs)])",
        [{"kind": "literal", "values": ["x"]}],
        {},
    )
    assert out == {"items": [{"output": "[2, 1]", "value": "[2, 1]"}]}
    with pytest.raises(ValueError, match="'open'.*does not provide"):
        _run_wrapper("def f(inputs):\n    return open('/etc/passwd').read()", [], {})


def test_wrapper_allows_importing_any_module_the_container_image_provides() -> None:
    # The container is the isolation boundary for a standalone task: any module
    # the image provides may be imported, so __import__ is not restricted.
    out = _run_wrapper(
        "def f(inputs):\n    import requests\n    return requests.__name__", [], {}
    )
    assert out == {"items": [{"output": "requests", "value": "requests"}]}


def test_wrapper_imports_numpy_and_pandas_only_when_named() -> None:
    out = _run_wrapper(
        "def f(inputs):\n    return str(np.array([1, 2]).sum() + len(pd.Series([1])))",
        [],
        {},
    )
    assert out == {"items": [{"output": "4", "value": "4"}]}


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
    assert namespace["main"]({}) == {"items": [{"output": "NVDA", "value": "NVDA"}]}


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


def test_python_step_carries_image_when_set_and_omits_it_when_not() -> None:
    _, shout, compiled = _llm_then_lambda(image="ghcr.io/me/sim:latest")
    graph = RuntimeGraphBuilder().build(compiled)
    spec = graph.nodes[shout.id].to_flowmesh_node()["spec"]
    assert spec["image"] == "ghcr.io/me/sim:latest"

    _, plain, compiled_plain = _llm_then_lambda()
    plain_spec = (
        RuntimeGraphBuilder()
        .build(compiled_plain)
        .nodes[plain.id]
        .to_flowmesh_node()["spec"]
    )
    assert "image" not in plain_spec


def test_image_lambda_read_by_an_llm_message_chain_is_rejected() -> None:
    stock = input_placeholder("Stock")
    shout = LambdaOp(
        [stock], fn=lambda inputs: str(inputs[0]).upper(), image="ghcr.io/me/sim:latest"
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content=shout)],
        config=GenerationConfig(model=_MODEL),
    )
    compiled = Graph.from_ops([as_output("result", llm)]).compile(Stock=["NVDA"])
    with pytest.raises(
        ValueError, match="image-bearing LambdaOp '.*' through its message chain"
    ):
        RuntimeGraphBuilder().build(compiled)


def test_image_lambda_read_by_llm_via_node_column_compiles() -> None:
    stock = input_placeholder("Stock")
    shout = LambdaOp(
        [stock], fn=lambda inputs: str(inputs[0]).upper(), image="ghcr.io/me/sim:latest"
    )
    llm = LLMChatOp(
        [OpMessage(role="user", content="Result: {r}")],
        config=GenerationConfig(model=_MODEL),
        rowwise_template="Result: {r}",
        rowwise_columns=[{"label": "r", "node": shout.id, "path": "items.output"}],
    )
    llm.inputs.append(shout)
    compiled = Graph.from_ops([as_output("result", llm)]).compile(Stock=["NVDA"])
    graph = RuntimeGraphBuilder().build(compiled)
    assert shout.id in _python_nodes(graph)
    spec = graph.nodes[shout.id].to_flowmesh_node()["spec"]
    assert spec["image"] == "ghcr.io/me/sim:latest"


def test_image_lambda_allows_a_larger_timeout_cap() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        LambdaOp([["x"]], fn=lambda inputs: str(inputs[0]), timeout_s=601)
    LambdaOp(
        [["x"]],
        fn=lambda inputs: str(inputs[0]),
        timeout_s=3600,
        image="ghcr.io/me/sim:latest",
    )
    with pytest.raises(ValueError, match="timeout_s"):
        LambdaOp(
            [["x"]],
            fn=lambda inputs: str(inputs[0]),
            timeout_s=3601,
            image="ghcr.io/me/sim:latest",
        )


@pytest.mark.parametrize("image", ["", "   "])
def test_lambda_op_rejects_an_empty_image(image: str) -> None:
    with pytest.raises(ValueError, match="image"):
        LambdaOp([["x"]], fn=lambda inputs: str(inputs[0]), image=image)


def test_python_step_carries_expected_runtime_when_set_and_omits_it_when_not() -> None:
    _, shout, compiled = _llm_then_lambda(expected_runtime_s=5)
    graph = RuntimeGraphBuilder().build(compiled)
    node = graph.nodes[shout.id]
    assert node.data_spec["expected_runtime_s"] == 5
    # The hint is planning-only and never reaches the FlowMesh task spec.
    assert "expected_runtime_s" not in node.to_flowmesh_node()["spec"]

    _, plain, compiled_plain = _llm_then_lambda()
    plain_node = RuntimeGraphBuilder().build(compiled_plain).nodes[plain.id]
    assert "expected_runtime_s" not in plain_node.data_spec


def test_python_step_passes_cache_through_when_set_and_omits_it_when_not() -> None:
    _, shout, compiled = _llm_then_lambda(cache=True)
    graph = RuntimeGraphBuilder().build(compiled)
    spec = graph.nodes[shout.id].to_flowmesh_node()["spec"]
    assert spec["cache"] is True

    _, plain, compiled_plain = _llm_then_lambda()
    plain_spec = (
        RuntimeGraphBuilder()
        .build(compiled_plain)
        .nodes[plain.id]
        .to_flowmesh_node()["spec"]
    )
    assert "cache" not in plain_spec


def test_lambda_op_rejects_a_non_bool_cache() -> None:
    for bad in (1, "yes", 0.0):
        with pytest.raises(ValueError, match="cache"):
            LambdaOp(
                [["x"]],
                fn=lambda inputs: str(inputs[0]),
                cache=bad,  # type: ignore[call-overload]
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_runtime_s": 0},
        {"expected_runtime_s": -1},
        {"expected_runtime_s": True},
        {"expected_runtime_s": "5"},
        {"expected_runtime_s": 601},
    ],
)
def test_lambda_op_rejects_a_bad_expected_runtime(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="expected_runtime_s"):
        LambdaOp([["x"]], fn=lambda inputs: str(inputs[0]), **kwargs)


def test_lambda_op_expected_runtime_must_not_exceed_the_effective_timeout() -> None:
    with pytest.raises(ValueError, match="expected_runtime_s"):
        LambdaOp([["x"]], fn=lambda inputs: str(inputs[0]), expected_runtime_s=601)
    with pytest.raises(ValueError, match="expected_runtime_s"):
        LambdaOp(
            [["x"]],
            fn=lambda inputs: str(inputs[0]),
            timeout_s=45,
            expected_runtime_s=46,
        )
    LambdaOp(
        [["x"]],
        fn=lambda inputs: str(inputs[0]),
        timeout_s=45,
        expected_runtime_s=45,
    )


def test_halo_costs_a_default_python_step_at_five_percent_of_its_timeout() -> None:
    optimizer = HaloOptimizer()

    def cost(raw: dict[str, Any]) -> float:
        node = Node(id="n", type="python", engine="python", model="", raw=raw)
        return optimizer._python_exec_cost(node)

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


def _sql_stage(
    tmp_path: Path, name: str, tables: list[dict[str, dict[str, Any]]]
) -> str:
    """A SQL-shaped retrieval stage: one item per symbol row, each a serialized df."""
    return _stage(
        tmp_path,
        name,
        {"items": [{"table": {"df": json.dumps(t)}} for t in tables]},
    )


def _s3_stage(tmp_path: Path, name: str, keys: list[list[str]]) -> str:
    return _stage(
        tmp_path,
        name,
        {"items": [{"keys": k} for k in keys]},
    )


def _flatten_plan() -> python_step.ColumnPlan:
    return [
        {
            "kind": "stage",
            "node": "art",
            "path": "keys",
            "label": "art_batch",
            "grouped": True,
        },
        {
            "kind": "literal",
            "values": ["AAA", "BBB", "CCC"],
            "label": "symbol",
            "grouped": False,
        },
        {
            "kind": "stage",
            "node": "news",
            "path": "table.title",
            "label": "title",
            "grouped": True,
        },
    ]


def _flatten_inputs(
    tmp_path: Path, sizes: list[int], titles: list[list[str]]
) -> dict[str, str]:
    """Build per-symbol news/art stages; ``titles`` is one list per symbol."""
    news_tables = [
        {"title": {str(j): t for j, t in enumerate(sym_titles)}}
        for sym_titles in titles
    ]
    news = _sql_stage(tmp_path, "news", news_tables)
    art = _s3_stage(tmp_path, "art", [[f"k{j}" for j in range(s)] for s in sizes])
    return {"news": news, "art": art}


def test_flatten_renders_one_prompt_per_image_row_major(tmp_path: Path) -> None:
    # sizes (2, 0, 3): symbol 0 has 2 images, symbol 1 none, symbol 2 has 3.
    titles = [["t0", "t1"], [], ["t2", "t3", "t4"]]
    inputs = _flatten_inputs(tmp_path, [2, 0, 3], titles)
    out = _run_reshape("flatten", _flatten_plan(), inputs, template="{symbol}: {title}")
    items = out["items"]
    assert len(items) == 5
    # Row-major: row 0's two images first, then row 2's three.
    assert [i["row"] for i in items] == [0, 0, 2, 2, 2]
    assert items[0]["output"] == "AAA: t0"
    assert items[1]["output"] == "AAA: t1"
    assert items[2]["output"] == "CCC: t2"
    assert items[4]["output"] == "CCC: t4"


def test_flatten_renders_one_prompt_per_image_middle_empty(tmp_path: Path) -> None:
    titles = [["t0", "t1"], ["t2", "t3", "t4"], []]
    inputs = _flatten_inputs(tmp_path, [2, 3, 0], titles)
    out = _run_reshape("flatten", _flatten_plan(), inputs, template="{symbol}: {title}")
    items = out["items"]
    assert len(items) == 5
    assert [i["row"] for i in items] == [0, 0, 1, 1, 1]


def test_flatten_rejects_group_size_mismatch(tmp_path: Path) -> None:
    # 3 symbols; row 0 has 4 news titles but only 3 art keys -> grouped size mismatch.
    news = _sql_stage(
        tmp_path,
        "news",
        [
            {"title": {"0": "t0", "1": "t1", "2": "t2", "3": "t3"}},
            {"title": {"0": "t4"}},
            {"title": {"0": "t5"}},
        ],
    )
    art = _s3_stage(tmp_path, "art", [["k0", "k1", "k2"], ["k3"], ["k4"]])
    inputs = {"news": news, "art": art}
    with pytest.raises(ValueError, match="row 0 has 4 values; expected 3"):
        _run_reshape("flatten", _flatten_plan(), inputs)


def test_flatten_rejects_missing_key(tmp_path: Path) -> None:
    news = _sql_stage(tmp_path, "news", [{"title": {"0": "t0", "1": "t1"}}])
    # art item carries no "keys" key -> _walk_path raises naming node and path.
    art = _stage(tmp_path, "art", {"items": [{"nokeys": []}]})
    inputs = {"news": news, "art": art}
    with pytest.raises(ValueError, match="missing 'keys'"):
        _run_reshape("flatten", _flatten_plan(), inputs)


def test_flatten_checks_grouped_columns_against_image_counts(tmp_path: Path) -> None:
    # The image column (first) has groups [2, 1]; the title column has [2, 2],
    # so row 1's title count (2) mismatches the image count (1).
    news = _sql_stage(
        tmp_path,
        "news",
        [
            {"title": {"0": "t0", "1": "t1"}},
            {"title": {"0": "t2", "1": "t3"}},
        ],
    )
    art = _s3_stage(tmp_path, "art", [["k0", "k1"], ["k2"]])
    plan: python_step.ColumnPlan = [
        {
            "kind": "stage",
            "node": "art",
            "path": "keys",
            "label": "art_batch",
            "grouped": True,
        },
        {
            "kind": "stage",
            "node": "news",
            "path": "table.title",
            "label": "title",
            "grouped": True,
        },
    ]
    inputs = {"news": news, "art": art}
    with pytest.raises(ValueError, match="row 1 has 2 values; expected 1"):
        _run_reshape("flatten", plan, inputs)


def test_flatten_with_only_literal_symbol_emits_prompts_without_image_text(
    tmp_path: Path,
) -> None:
    # Images [2, 1] and only a literal symbol column: 3 prompts, rows [0,0,1],
    # and no image content is rendered into the prompt.
    art = _s3_stage(tmp_path, "art", [["k0", "k1"], ["k2"]])
    plan: python_step.ColumnPlan = [
        {
            "kind": "stage",
            "node": "art",
            "path": "keys",
            "label": "art_batch",
            "grouped": True,
        },
        {
            "kind": "literal",
            "values": ["AAA", "BBB"],
            "label": "symbol",
            "grouped": False,
        },
    ]
    out = _run_reshape("flatten", plan, {"art": art}, template="{symbol}")
    items = out["items"]
    assert len(items) == 3
    assert [i["row"] for i in items] == [0, 0, 1]
    assert items[0]["output"] == "AAA"
    assert items[1]["output"] == "AAA"
    assert items[2]["output"] == "BBB"
    assert all("k0" not in i["output"] and "k1" not in i["output"] for i in items)


def test_regroup_groups_outputs_back_per_symbol(tmp_path: Path) -> None:
    # flatten produced 5 prompts with rows [0,0,2,2,2]; regroup must give 3 groups.
    infer = _stage(
        tmp_path,
        "infer",
        {
            "items": [
                {"output": "o0"},
                {"output": "o1"},
                {"output": "o2"},
                {"output": "o3"},
                {"output": "o4"},
            ]
        },
    )
    flatten = _stage(
        tmp_path,
        "flatten",
        {
            "items": [
                {"output": "p", "row": 0},
                {"output": "p", "row": 0},
                {"output": "p", "row": 2},
                {"output": "p", "row": 2},
                {"output": "p", "row": 2},
            ]
        },
    )
    news = _sql_stage(
        tmp_path,
        "news",
        [
            {"title": {"0": "t0", "1": "t1"}},
            {"title": {}},
            {"title": {"0": "t2", "1": "t3", "2": "t4"}},
        ],
    )
    plan: python_step.ColumnPlan = [
        {"kind": "stage", "node": "infer"},
        {"kind": "stage", "node": "flatten", "path": "row"},
        {"kind": "stage", "node": "news", "path": "table.title", "grouped": True},
    ]
    out = _run_reshape(
        "regroup", plan, {"infer": infer, "flatten": flatten, "news": news}
    )
    assert out == {
        "items": [
            {"output": ["o0", "o1"]},
            {"output": []},
            {"output": ["o2", "o3", "o4"]},
        ]
    }


def test_regroup_rejects_row_out_of_range(tmp_path: Path) -> None:
    infer = _stage(tmp_path, "infer", {"items": [{"output": "o0"}]})
    flatten = _stage(tmp_path, "flatten", {"items": [{"output": "p", "row": 5}]})
    news = _sql_stage(
        tmp_path,
        "news",
        [{"title": {"0": "t0"}}, {"title": {"0": "t1"}}, {"title": {"0": "t2"}}],
    )
    plan: python_step.ColumnPlan = [
        {"kind": "stage", "node": "infer"},
        {"kind": "stage", "node": "flatten", "path": "row"},
        {"kind": "stage", "node": "news", "path": "table.title", "grouped": True},
    ]
    with pytest.raises(ValueError, match="row 5 out of range for 3 symbols"):
        _run_reshape(
            "regroup", plan, {"infer": infer, "flatten": flatten, "news": news}
        )


def test_regroup_rejects_output_count_mismatch(tmp_path: Path) -> None:
    infer = _stage(tmp_path, "infer", {"items": [{"output": "o0"}]})
    flatten = _stage(
        tmp_path,
        "flatten",
        {"items": [{"output": "p", "row": 0}, {"output": "p", "row": 0}]},
    )
    news = _sql_stage(tmp_path, "news", [{"title": {"0": "t0", "1": "t1", "2": "t2"}}])
    plan: python_step.ColumnPlan = [
        {"kind": "stage", "node": "infer"},
        {"kind": "stage", "node": "flatten", "path": "row"},
        {"kind": "stage", "node": "news", "path": "table.title", "grouped": True},
    ]
    with pytest.raises(ValueError, match="1 outputs but 2 rows"):
        _run_reshape(
            "regroup", plan, {"infer": infer, "flatten": flatten, "news": news}
        )


def _walk(value: Any, parts: list[str]) -> Any:
    for idx, part in enumerate(parts):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list):
            rest = parts[idx:]
            return [_walk(v, rest) for v in value]
        else:
            raise KeyError(part)
    return value


def test_retrieval_param_paths_resolve_against_python_result() -> None:
    """The emitted retrieval param path must resolve against a python step's
    PythonResult-shaped result (``{"value": {"items": [...]}}``), so a wrong
    path fails the way it fails in FlowMesh."""
    stock = input_placeholder("Stock")
    terms = LambdaOp([stock], fn=lambda inputs: str(inputs[0]).upper())
    retrieval = DataRetrievalOp(
        data_spec={
            "type": "lumid",
            "mode": "sql",
            "template": "SELECT * FROM t WHERE q = :q",
            "params": [
                {"label": "whole", "node": terms.id, "path": "items.output"},
            ],
        },
        inputs=[terms],
    )
    llm = LLMChatOp(
        [
            OpMessage(role="user", content=terms),
            OpMessage(role="user", content=retrieval),
        ],
        config=GenerationConfig(model=_MODEL),
    )
    compiled = Graph.from_ops([as_output("result", llm)]).compile(Stock=["NVDA"])
    graph = RuntimeGraphBuilder().build(compiled)

    params = graph.nodes[retrieval.id].data_spec["params"]
    (whole,) = [p for p in params if p.get("label") == "whole"]

    # The whole-item read maps to the python step's textified output.
    assert whole["path"] == "value.items.output"
    result = {"value": {"items": [{"output": "NVDA", "value": {"field": "x"}}]}}
    assert _walk(result, whole["path"].split(".")) == ["NVDA"]


def _flowmesh_walk(value: Any, expr: str) -> Any:
    """Resolve ``expr`` against a result with FlowMesh's graph-template walker
    semantics: attribute access maps over lists of dicts and never JSON-decodes
    strings (a JSON string is opaque, so a field path into it fails)."""
    parts = expr.split(".")
    for token in parts[1:]:
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and all(
            isinstance(v, dict) and token in v for v in value
        ):
            value = [v[token] for v in value]
        else:
            raise KeyError(
                f"{token} in {parts} is not a valid key - "
                f"{type(value).__name__}, {value}"
            )
    return value


def test_row_wrapper_field_path_resolves_without_json_decode(tmp_path: Path) -> None:
    # A row-mode LambdaOp whose result is a dict must expose the raw value so a
    # prompt field ref can read a field of it. FlowMesh's graph-template walker
    # never JSON-decodes strings, so the step must emit the raw value under its
    # own key (``value``) rather than only the textified ``output``.
    out = _run_wrapper(
        "def q(row):\n    return {'question': 'what?'}\n",
        [{"kind": "literal", "values": ["x"]}],
        {},
    )
    assert _flowmesh_walk(out, "value.items.value.question") == ["what?"]
