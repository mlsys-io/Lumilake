"""LambdaOp user code runs in a sandboxed child, never in the server process."""

import json
import os
import sys

import pytest

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils.sandbox_exec import (
    SandboxedFunction,
    SandboxError,
    run_lambda,
    validate_lambda_source,
)

_SHOUT = "def shout(inputs):\n    (text,) = inputs\n    return text.upper()"

# Reaches os.environ without a single builtin — the escape the restricted
# namespace never stopped. Inside the sandbox it must find nothing of ours.
_ESCAPE = (
    "def leak(inputs):\n"
    "    g = [c for c in ().__class__.__base__.__subclasses__()\n"
    "         if c.__name__ == '_wrap_close'][0].__init__.__globals__\n"
    "    return str(dict(g['environ']))\n"
)


def test_batch_runs_in_one_child_and_keeps_the_contract() -> None:
    assert run_lambda(_SHOUT, [("hello",), ("world",)]) == ["HELLO", "WORLD"]
    assert run_lambda("lambda a: a[0] + a[1]", [("x", "y")]) == ["xy"]
    code = "def f(a):\n    return {'n': math.floor(2.5), 'j': json.dumps(a[0])}"
    assert run_lambda(code, [("q",)]) == [str({"n": 2, "j": '"q"'})]


def test_empty_batch_starts_no_child() -> None:
    assert run_lambda(_SHOUT, []) == []


def test_user_exception_comes_back_with_its_traceback() -> None:
    code = "def f(a):\n    return a[5]"
    with pytest.raises(SandboxError, match="IndexError") as exc:
        run_lambda(code, [("x",)])
    assert "<lambda_op>" in str(exc.value)


def test_restricted_namespace_is_unchanged() -> None:
    with pytest.raises(SandboxError, match="NameError: name 'open'"):
        run_lambda("def f(a):\n    return open('/etc/passwd').read()", [("x",)])


def test_infinite_loop_is_killed_at_the_timeout() -> None:
    with pytest.raises(SandboxError, match="time limit"):
        run_lambda("def f(a):\n    while True:\n        pass", [("x",)], timeout_s=1)


def test_memory_over_the_limit_fails() -> None:
    code = "def f(a):\n    return str(len('x' * (600 * 1024 * 1024)))"
    with pytest.raises(SandboxError, match="memory"):
        run_lambda(code, [("x",)], memory_mb=256)


def test_server_environment_is_not_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMILAKE_TEST_SECRET", "s3cr3t-value")
    (seen,) = run_lambda(_ESCAPE, [("x",)])
    assert "s3cr3t-value" not in seen
    assert "LUMILAKE_TEST_SECRET" not in seen
    assert "PATH" in seen  # the minimal environment, nothing more


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc")
def test_escaped_code_cannot_read_the_server_environ() -> None:
    code = _ESCAPE.replace(
        "    return str(dict(g['environ']))\n",
        "    fd = g['open']('/proc/%d/environ' % g['getppid'](), 0)\n"
        "    return str(g['read'](fd, 64))\n",
    )
    with pytest.raises(SandboxError, match="PermissionError"):
        run_lambda(code, [("x",)])
    assert os.getpid()  # the server process itself is untouched


def test_validation_parses_but_never_executes() -> None:
    # A module body runs on exec; validation must not. The trailing raise would
    # have fired inside the server under the old materializer.
    code = "def f(a):\n    return a[0]\nraise SystemExit('ran in the server')"
    assert validate_lambda_source(code) == "f"
    with pytest.raises(SandboxError, match="SystemExit"):
        run_lambda(code, [("x",)])


@pytest.mark.parametrize(
    "code, message",
    [
        ("def f(a, b):\n    return a", "exactly 1 parameter"),
        ("X = 1\ndef f(a):\n    return a", "first binding"),
        ("def f(a:\n", "does not parse"),
        ("async def f(a):\n    return a", "async"),
    ],
)
def test_invalid_source_is_rejected_without_running(code: str, message: str) -> None:
    with pytest.raises(SandboxError, match=message):
        validate_lambda_source(code)


def test_limits_are_bounded() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        run_lambda(_SHOUT, [("x",)], timeout_s=0)
    with pytest.raises(ValueError, match="memory_mb"):
        SandboxedFunction(_SHOUT, memory_mb=1)


def test_deserialized_lambda_op_holds_a_sandboxed_function() -> None:
    op = LambdaOp._from_json(
        {"fn_name": "shout", "_code": _SHOUT, "_inputs": [], "timeout_s": 5},
        {},
    )
    assert isinstance(op.fn, SandboxedFunction)
    assert op.fn(("hi",)) == "HI"
    assert op.timeout_s == 5
    assert op._serialize()["timeout_s"] == 5


def test_deserialize_rejects_a_bad_limit() -> None:
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


def test_yaml_limits_reach_the_op_and_the_fold_runs_sandboxed() -> None:
    spec = parse_yaml_payload(_CHAIN)["chain"]
    (lambda_dict,) = [d for d in spec["graph"].values() if d["_op"] == "LambdaOp"]
    assert lambda_dict["timeout_s"] == 10 and lambda_dict["memory_mb"] == 256

    graph = Graph.from_json(spec["graph"])
    (op,) = [o for o in graph.iter_ops() if isinstance(o, LambdaOp)]
    assert isinstance(op.fn, SandboxedFunction)

    runtime = RuntimeGraphBuilder().build(graph.compile(**spec["inputs"]))
    rendered = json.dumps([node.data_spec for node in runtime.nodes.values()])
    assert "HELLO, WORLD!" in rendered


def test_yaml_rejects_a_non_numeric_limit() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        parse_yaml_payload(_CHAIN.replace("timeout_s: 10", "timeout_s: soon"))
