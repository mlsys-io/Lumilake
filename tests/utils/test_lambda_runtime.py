"""Submitted LambdaOp code is parsed by the server, never executed by it."""

import json
from typing import Any

import pytest
from lumilake import envs

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp
from lumilake_server.ops.util_ops import SubmittedFunction
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils import lambda_runtime

_SHOUT = "def shout(inputs):\n    (text,) = inputs\n    return text.upper()"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def test_materialized_function_keeps_the_contract() -> None:
    assert lambda_runtime.materialize(_SHOUT)(("hello",)) == "HELLO"
    assert lambda_runtime.materialize("lambda a: a[0] + a[1]")(("x", "y")) == "xy"
    code = "def f(a):\n    return str({'n': math.floor(2.5), 'j': json.dumps(a[0])})"
    assert lambda_runtime.materialize(code)(("q",)) == str({"n": 2, "j": '"q"'})


def test_materialized_function_sees_only_the_restricted_builtins() -> None:
    with pytest.raises(NameError, match="open"):
        lambda_runtime.materialize("def f(a):\n    return open('/x')")(("x",))
    with pytest.raises(NameError, match="eval"):
        lambda_runtime.materialize("lambda a: eval('1')")(("x",))


def test_validation_parses_but_never_executes() -> None:
    code = "def f(a):\n    return a[0]\nran_module_body"
    assert lambda_runtime.validate_source(code) == "f"
    with pytest.raises(NameError, match="ran_module_body"):
        lambda_runtime.materialize(code)


def test_declared_fn_name_must_match_the_first_function() -> None:
    assert lambda_runtime.validate_source(_SHOUT, "shout") == "shout"
    assert lambda_runtime.materialize(_SHOUT, "shout")(("hi",)) == "HI"
    with pytest.raises(ValueError, match="does not match"):
        lambda_runtime.validate_source(_SHOUT, "other")
    with pytest.raises(ValueError, match="does not match"):
        lambda_runtime.materialize(_SHOUT, "other")


def test_declared_fn_name_only_labels_a_lambda() -> None:
    assert lambda_runtime.validate_source("lambda a: a[0]", "label") == "<lambda>"
    assert lambda_runtime.materialize("lambda a: a[0]", "label")(("x",)) == "x"


def test_the_first_function_binding_grammar_is_unchanged() -> None:
    code = "def other(a):\n    return 'other'\ndef wanted(a):\n    return 'wanted'"
    assert lambda_runtime.validate_source(code) == "other"
    with pytest.raises(ValueError, match="does not match"):
        lambda_runtime.validate_source(code, "wanted")


@pytest.mark.parametrize(
    "code, message",
    [
        ("def f(a, b):\n    return a", "exactly 1 parameter"),
        ("X = 1\ndef f(a):\n    return a", "first binding"),
        ("def f(a:\n", "does not parse"),
        ("async def f(a):\n    return a", "async"),
        ("lambda a, b: a", "exactly 1 parameter"),
    ],
)
def test_invalid_source_is_rejected(code: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        lambda_runtime.validate_source(code)


def test_deserialized_lambda_op_never_runs_its_code() -> None:
    code = "def f(a):\n    return a[0]\nraise SystemExit('ran in the server')"
    op = LambdaOp._from_json(
        {"fn_name": "f", "_code": code, "_inputs": [], "timeout_s": 5}, {}
    )
    assert isinstance(op.fn, SubmittedFunction)
    assert op.timeout_s == 5
    assert op._serialize()["timeout_s"] == 5
    with pytest.raises(RuntimeError, match="does not execute submitted code"):
        op.fn(("x",))


def test_deserialize_rejects_a_fn_name_the_code_does_not_define() -> None:
    with pytest.raises(ValueError, match="does not match"):
        LambdaOp._from_json({"fn_name": "other", "_code": _SHOUT, "_inputs": []}, {})
    op = LambdaOp._from_json(
        {"fn_name": "label", "_code": "lambda a: a[0]", "_inputs": []}, {}
    )
    assert op._serialize()["fn_name"] == "label"


def test_yaml_fn_name_that_the_code_does_not_define_is_rejected() -> None:
    payload = _CHAIN.replace("fn_name: shout", "fn_name: whisper")
    with pytest.raises(ValueError, match="does not match"):
        _build(payload)


def test_deserialize_rejects_bad_source_and_limits() -> None:
    with pytest.raises(ValueError, match="Invalid LambdaOp function"):
        LambdaOp._from_json(
            {"fn_name": "f", "_code": "def f(a, b):\n    return a", "_inputs": []}, {}
        )
    with pytest.raises(ValueError, match="memory_mb"):
        LambdaOp._from_json(
            {"fn_name": "shout", "_code": _SHOUT, "_inputs": [], "memory_mb": 10**9},
            {},
        )


_CHAIN = """
name: chain
inputs:
  Name: ["world"]
ops:
  - id: Greeting
    op: FormatOp
    inputs: [Name]
    template: "Hello, {name}!"
    format_kwargs: {name: Name}
  - id: Shout
    op: LambdaOp
    inputs: [Greeting]
    fn_name: shout
    timeout_s: 10
    memory_mb: 256
    code: |
      def shout(inputs):
          (greeting,) = inputs
          return greeting.upper()
  - id: Reply
    op: LLMChatOp
    inputs: [Shout]
    messages:
      - {role: user, content: Shout}
    config: {model: meta-llama/Llama-3.1-8B-Instruct}
outputs:
  - {name: reply, ref: Reply}
"""


def _build(payload: str) -> Any:
    spec = parse_yaml_payload(payload)["chain"]
    graph = Graph.from_json(spec["graph"])
    return RuntimeGraphBuilder().build(graph.compile(**spec["inputs"]))


def test_yaml_limits_reach_the_op() -> None:
    spec = parse_yaml_payload(_CHAIN)["chain"]
    (lambda_dict,) = [d for d in spec["graph"].values() if d["_op"] == "LambdaOp"]
    assert lambda_dict["timeout_s"] == 10 and lambda_dict["memory_mb"] == 256
    graph = Graph.from_json(spec["graph"])
    (op,) = [o for o in graph.iter_ops() if isinstance(o, LambdaOp)]
    assert isinstance(op.fn, SubmittedFunction)
    assert (op.timeout_s, op.memory_mb) == (10, 256)


def test_submitted_literal_lambda_is_not_folded_by_the_server() -> None:
    runtime = _build(_CHAIN)
    rendered = json.dumps([node.data_spec for node in runtime.nodes.values()])
    assert "HELLO, WORLD!" not in rendered
    assert "def shout" in rendered


def test_submitted_lambda_in_api_mode_fails_closed() -> None:
    payload = _CHAIN.replace(
        "config: {model: meta-llama/Llama-3.1-8B-Instruct}",
        "config: {model: meta-llama/Llama-3.1-8B-Instruct, api: {timeout_sec: 60}}",
    )
    with pytest.raises(ValueError, match="Lambda message transform"):
        _build(payload)


def test_yaml_rejects_a_non_numeric_limit() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        parse_yaml_payload(_CHAIN.replace("timeout_s: 10", "timeout_s: soon"))
