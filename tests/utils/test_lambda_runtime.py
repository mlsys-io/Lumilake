"""Submitted LambdaOp code is parsed by the server, never executed by it."""

import builtins
import inspect
import json
import math
import re
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp, as_output
from lumilake_server.ops.util_ops import SubmittedFunction
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.lambda_task_helpers import decode
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
    with pytest.raises(ValueError, match="'open'.*does not provide"):
        lambda_runtime.validate_source("def f(a):\n    return open('/x')", "f")
    with pytest.raises(ValueError, match="'eval'.*does not provide"):
        lambda_runtime.validate_source("lambda a: eval('1')")


def test_materialized_function_can_raise_a_builtin_exception() -> None:
    with pytest.raises(ValueError, match="boom"):
        lambda_runtime.materialize("def f(a):\n    raise ValueError('boom')")(("x",))


def test_materialized_function_can_catch_a_builtin_exception() -> None:
    code = (
        "def f(a):\n"
        "    try:\n"
        "        raise KeyError('k')\n"
        "    except KeyError:\n"
        "        return 'handled'"
    )
    assert lambda_runtime.materialize(code)(("x",)) == "handled"


def test_materialized_function_still_cannot_reference_systemexit() -> None:
    with pytest.raises(ValueError, match="'SystemExit'.*does not provide"):
        lambda_runtime.validate_source("def f(a):\n    raise SystemExit('x')", "f")


def test_validation_parses_but_never_executes() -> None:
    code = "def f(a):\n    return a[0]\n1/0"
    assert lambda_runtime.validate_source(code) == "f"
    with pytest.raises(ZeroDivisionError):
        lambda_runtime.materialize(code)


def test_declared_fn_name_must_be_a_defined_function() -> None:
    assert lambda_runtime.validate_source(_SHOUT, "shout") == "shout"
    assert lambda_runtime.materialize(_SHOUT, "shout")(("hi",)) == "HI"
    with pytest.raises(ValueError, match="no top-level function named 'other'"):
        lambda_runtime.validate_source(_SHOUT, "other")
    with pytest.raises(ValueError, match="functions defined: 'shout'"):
        lambda_runtime.materialize(_SHOUT, "other")


def test_declared_fn_name_only_labels_a_lambda() -> None:
    assert lambda_runtime.validate_source("lambda a: a[0]", "label") == "<lambda>"
    assert lambda_runtime.materialize("lambda a: a[0]", "label")(("x",)) == "x"


def test_fn_name_selects_the_function_among_several() -> None:
    code = "def other(a):\n    return 'other'\ndef wanted(a):\n    return 'wanted'"
    assert lambda_runtime.validate_source(code) == "other"
    assert lambda_runtime.validate_source(code, "wanted") == "wanted"
    assert lambda_runtime.materialize(code, "wanted")(("x",)) == "wanted"


def test_imports_and_constants_may_precede_the_function() -> None:
    code = "import os\nimport math as m\nK = 3\ndef main(a):\n    return m.sqrt(K * 3)"
    assert lambda_runtime.validate_source(code, "main") == "main"
    assert lambda_runtime.materialize(code, "main")(("x",)) == 3.0


def test_stdlib_imports_inside_the_function_work() -> None:
    code = (
        "def main(a):\n    import math\n    from os import path\n"
        "    return path.join('a', str(math.sqrt(16)))"
    )
    assert lambda_runtime.materialize(code, "main")(("x",)) == "a/4.0"


@pytest.mark.parametrize(
    "code",
    [
        "import requests\ndef f(a):\n    return a",
        "def f(a):\n    from httpx import get\n    return a",
    ],
)
def test_non_stdlib_imports_are_accepted_for_standalone_code(code: str) -> None:
    # A standalone task runs in its own container, so any module the image
    # provides may be imported; validation only checks the shape of the code.
    assert lambda_runtime.validate_source(code, "f") == "f"


def test_relative_imports_are_rejected() -> None:
    with pytest.raises(ValueError, match="relative imports"):
        lambda_runtime.validate_source(
            "def f(a):\n    from . import x\n    return a", "f"
        )


def test_inline_source_keeps_the_flowmesh_safe_eval_grammar() -> None:
    assert lambda_runtime.validate_inline_source(_SHOUT, "shout") == "shout"
    assert lambda_runtime.validate_inline_source("lambda a: a[0]", "x") == "<lambda>"
    with pytest.raises(ValueError, match="imports 'math'.*evaluated inline"):
        lambda_runtime.validate_inline_source(
            "def f(a):\n    import math\n    return a", "f"
        )
    with pytest.raises(ValueError, match="imports 'os'"):
        lambda_runtime.validate_inline_source("import os\ndef f(a):\n    return a", "f")
    with pytest.raises(ValueError, match="annotates a top-level assignment"):
        lambda_runtime.validate_inline_source(
            "PROMPT: str = 'x'\ndef f(a):\n    return a", "f"
        )
    with pytest.raises(ValueError, match="references 'Message'"):
        lambda_runtime.validate_inline_source("def f(a: Message):\n    return a", "f")


@pytest.mark.parametrize("name", sorted(lambda_runtime._inline_names()))
def test_inline_source_rejects_a_name_the_wrapper_binds(name: str) -> None:
    code = f"def main(x):\n    return x\n{name} = 1"
    with pytest.raises(ValueError, match="generated wrapper that already binds"):
        lambda_runtime.validate_inline_source(code, "main")


@pytest.mark.parametrize(
    "code",
    [
        "def main(x):\n    raise ValueError('boom')",
        "def main(x):\n    try:\n        return x\n"
        "    except KeyError:\n        return 'k'",
        "def main(x):\n    return isinstance(x, RuntimeError)",
    ],
)
def test_inline_source_rejects_exception_class_references(code: str) -> None:
    """FlowMesh's inline ``safe_eval`` namespace has no exception classes, so
    inline code that references one must be rejected at submit rather than
    fail on the worker with a NameError."""
    with pytest.raises(ValueError, match="evaluated inline.*not defined"):
        lambda_runtime.validate_inline_source(code, "main")


def test_inline_source_without_exception_names_is_accepted() -> None:
    assert (
        lambda_runtime.validate_inline_source(
            "def main(x):\n    return x.upper()", "main"
        )
        == "main"
    )
    assert (
        lambda_runtime.validate_inline_source(
            "def main(x):\n    return str(len(x))", "main"
        )
        == "main"
    )


def test_standalone_source_still_accepts_exception_classes() -> None:
    assert (
        lambda_runtime.validate_source(
            "def main(x):\n    raise ValueError('boom')", "main"
        )
        == "main"
    )
    assert (
        lambda_runtime.validate_source(
            "def main(x):\n    try:\n        return x\n"
            "    except KeyError:\n        return 'k'",
            "main",
        )
        == "main"
    )


@pytest.mark.parametrize(
    "code, name",
    [
        ("def f(a):\n    return open(a[0])", "open"),
        ("def f(a):\n    return eval(a[0])", "eval"),
        ("def f(a):\n    return a[0] + os.sep", "os"),
        ("lambda a: a[0] + foo", "foo"),
        ("def f(a):\n    return a[0] + np_typo", "np_typo"),
        ('PREFIX = os.environ["X"]\ndef f(a):\n    return PREFIX', "os"),
        (
            "def f(a):\n    def helper(x):\n        return missing\n"
            "    return helper(a)",
            "missing",
        ),
    ],
)
def test_standalone_source_rejects_unprovided_free_names(code: str, name: str) -> None:
    """A name the LambdaOp namespace does not provide would raise NameError on
    the worker, so it must be rejected at submit with the name."""
    with pytest.raises(ValueError, match=f"'{name}'.*does not provide"):
        lambda_runtime.validate_source(code, "f")


@pytest.mark.parametrize(
    "code",
    [
        "def f(a):\n    return len(a) + str(a[0])",
        "def f(a):\n    return math.sqrt(a[0])",
        "def f(a):\n    return json.dumps(a[0])",
        "def f(a):\n    return np.array(a)",
        "def f(a):\n    return pd.Series(a)",
        "K = 3\ndef f(a):\n    return a[0] + K",
        "def helper(a):\n    return a\ndef f(a):\n    return helper(a)",
        "import math\ndef f(a):\n    return math.sqrt(a[0])",
        "def f(a):\n    return isinstance(a[0], ValueError)",
    ],
)
def test_standalone_source_accepts_provided_or_bound_names(code: str) -> None:
    """Safe builtins, namespace globals, np/pd, module constants, helper defs
    and imports all resolve in the standalone namespace."""
    assert lambda_runtime.validate_source(code, "f") == "f"


@pytest.mark.parametrize(
    "code, message",
    [
        ("def f(a, b):\n    return a", "exactly 1 parameter"),
        ("X = 1\n", "must define a top-level function"),
        ("def f(a:\n", "does not parse"),
        ("async def f(a):\n    return a", "async"),
        ("lambda a, b: a", "exactly 1 parameter"),
    ],
)
def test_invalid_source_is_rejected(code: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        lambda_runtime.validate_source(code)


def test_deserialized_lambda_op_never_runs_its_code() -> None:
    code = "def f(a):\n    return a[0]\n1/0"
    op = LambdaOp._from_json(
        {"_id": "f", "fn_name": "f", "_code": code, "_inputs": [], "timeout_s": 5}, {}
    )
    assert isinstance(op.fn, SubmittedFunction)
    assert op.timeout_s == 5
    assert op._serialize()["timeout_s"] == 5
    with pytest.raises(RuntimeError, match="does not execute submitted code"):
        op.fn(("x",))


def test_deserialize_rejects_a_fn_name_the_code_does_not_define() -> None:
    with pytest.raises(ValueError, match="no top-level function named 'other'"):
        LambdaOp._from_json(
            {"_id": "other", "fn_name": "other", "_code": _SHOUT, "_inputs": []}, {}
        )
    op = LambdaOp._from_json(
        {"_id": "label", "fn_name": "label", "_code": "lambda a: a[0]", "_inputs": []},
        {},
    )
    assert op._serialize()["fn_name"] == "label"


def test_yaml_fn_name_that_the_code_does_not_define_is_rejected() -> None:
    payload = _CHAIN.replace("fn_name: shout", "fn_name: whisper")
    with pytest.raises(ValueError, match="no top-level function named 'whisper'"):
        _build(payload)


def test_deserialize_rejects_bad_source_and_limits() -> None:
    with pytest.raises(ValueError, match="Invalid LambdaOp function"):
        LambdaOp._from_json(
            {
                "_id": "f",
                "fn_name": "f",
                "_code": "def f(a, b):\n    return a",
                "_inputs": [],
            },
            {},
        )
    with pytest.raises(ValueError, match="lambda_shout': memory_mb"):
        LambdaOp._from_json(
            {
                "_id": "lambda_shout",
                "fn_name": "shout",
                "_code": _SHOUT,
                "_inputs": [],
                "memory_mb": 10**9,
            },
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
    rendered = json.dumps(
        [
            node.data_spec.model_dump(exclude_none=True)
            for node in runtime.nodes.values()
        ]
    )
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


def test_yaml_image_reaches_the_op() -> None:
    spec = parse_yaml_payload(
        _CHAIN.replace("memory_mb: 256", "memory_mb: 256\n    image: my/lambda:1")
    )["chain"]
    (lambda_dict,) = [d for d in spec["graph"].values() if d["_op"] == "LambdaOp"]
    assert lambda_dict["image"] == "my/lambda:1"
    graph = Graph.from_json(spec["graph"])
    (op,) = [o for o in graph.iter_ops() if isinstance(o, LambdaOp)]
    assert op.image == "my/lambda:1"


def test_yaml_rejects_an_empty_image() -> None:
    with pytest.raises(ValueError, match="image"):
        parse_yaml_payload(
            _CHAIN.replace("memory_mb: 256", "memory_mb: 256\n    image: '  '")
        )


def test_native_graph_image_round_trip() -> None:
    graph = Graph.from_ops(
        [
            as_output(
                "out",
                LambdaOp(
                    [["x"]],
                    fn=SubmittedFunction("def f(a):\n    return a[0]", "f"),
                    code="def f(a):\n    return a[0]",
                    image="my/lambda:1",
                ),
            )
        ]
    )
    (op,) = [o for o in graph.iter_ops() if isinstance(o, LambdaOp)]
    assert op.image == "my/lambda:1"
    (round_tripped,) = [
        o
        for o in Graph.from_json(graph.serialize()).iter_ops()
        if isinstance(o, LambdaOp)
    ]
    assert round_tripped.image == "my/lambda:1"


def test_yaml_expected_runtime_reaches_the_op() -> None:
    payload = _CHAIN.replace(
        "    memory_mb: 256\n",
        "    memory_mb: 256\n    expected_runtime_s: 5\n",
    )
    spec = parse_yaml_payload(payload)["chain"]
    (lambda_dict,) = [d for d in spec["graph"].values() if d["_op"] == "LambdaOp"]
    assert lambda_dict["expected_runtime_s"] == 5
    graph = Graph.from_json(spec["graph"])
    (op,) = [o for o in graph.iter_ops() if isinstance(o, LambdaOp)]
    assert op.expected_runtime_s == 5
    assert op._serialize()["expected_runtime_s"] == 5


def test_yaml_rejects_a_bad_expected_runtime() -> None:
    with pytest.raises(ValueError, match="expected_runtime_s"):
        parse_yaml_payload(
            _CHAIN.replace(
                "    memory_mb: 256\n",
                "    memory_mb: 256\n    expected_runtime_s: true\n",
            )
        )


def test_native_graph_expected_runtime_round_trip() -> None:
    op = LambdaOp._from_json(
        {
            "_id": "lambda_shout",
            "fn_name": "shout",
            "_code": _SHOUT,
            "_inputs": [],
            "timeout_s": 5,
            "expected_runtime_s": 3,
        },
        {},
    )
    assert op.expected_runtime_s == 3
    assert op._serialize()["expected_runtime_s"] == 3


def test_yaml_cache_reaches_the_op() -> None:
    payload = _CHAIN.replace(
        "    memory_mb: 256\n",
        "    memory_mb: 256\n    cache: true\n",
    )
    spec = parse_yaml_payload(payload)["chain"]
    (lambda_dict,) = [d for d in spec["graph"].values() if d["_op"] == "LambdaOp"]
    assert lambda_dict["cache"] is True
    graph = Graph.from_json(spec["graph"])
    (op,) = [o for o in graph.iter_ops() if isinstance(o, LambdaOp)]
    assert op.cache is True
    assert op._serialize()["cache"] is True


def test_yaml_rejects_a_non_bool_cache() -> None:
    with pytest.raises(ValueError, match="cache"):
        parse_yaml_payload(
            _CHAIN.replace(
                "    memory_mb: 256\n",
                "    memory_mb: 256\n    cache: 1\n",
            )
        )


def test_native_graph_cache_round_trip() -> None:
    op = LambdaOp._from_json(
        {
            "_id": "lambda_shout",
            "fn_name": "shout",
            "_code": _SHOUT,
            "_inputs": [],
            "timeout_s": 5,
            "cache": False,
        },
        {},
    )
    assert op.cache is False
    assert op._serialize()["cache"] is False


_STANDALONE = """
name: lambda-import-test
inputs:
  Dummy: ["run"]
outputs:
  - name: result
    ref: Check
ops:
  - id: Check
    op: LambdaOp
    inputs: [Dummy]
    fn_name: main
    code: |
      import os
      def main(Dummy):
          import math
          return math.sqrt(16)
"""


def test_yaml_standalone_lambda_with_imports_compiles() -> None:
    """Imports before and inside ``main`` compile to a python task."""
    spec = parse_yaml_payload(_STANDALONE)["lambda-import-test"]
    graph = Graph.from_json(spec["graph"])
    runtime = RuntimeGraphBuilder().build(graph.compile(**spec["inputs"]))
    (node,) = [n for n in runtime.nodes.values() if n.task_type == "python"]
    assert "import math" in node.data_spec.model_dump(exclude_none=True)["code"]


def _sdk_imports(inputs: tuple[float, ...]) -> str:
    import math

    return str(math.sqrt(inputs[0]))


def _run_inlined(body: str, row: str) -> str:
    shout = "      def shout(inputs):\n          (greeting,) = inputs\n"
    payload = _CHAIN.replace('Name: ["world"]', f"Name: [{json.dumps(row)}]")
    body = "\n".join("      " + s for s in body.splitlines())
    payload = payload.replace(shout + "          return greeting.upper()", body)
    nodes = _build(payload).nodes.values()
    reply = next(n for n in nodes if n.task_type == "inference")
    steps = reply.data_spec.model_dump(exclude_none=True)["template"]["options"][
        "format"
    ]["steps"]
    function = next(s["function"] for s in steps if s.get("function"))
    local: dict[str, Any] = {}
    safe = {n: getattr(builtins, n) for n in lambda_runtime._SAFE_BUILTIN_BASE_NAMES}
    exec(function, {"__builtins__": safe, "json": json, "re": re, "math": math}, local)
    assert list(local) == ["adapt"]
    assert isinstance(result := local["adapt"]((row,)), str)
    return result


@pytest.mark.parametrize(
    "body, expected",
    [
        ('def shout(a):\n    return f"""Q\n  {a[0][\'name\']}\nA:"""', "Q\n  q\nA:"),
        ("lambda a: a[0]['name']  # first", "q"),
        ('def shout(a):\n    return {"k": a[0]}', '{"k": {"name": "q"}}'),
        ("lambda a: a[0].upper()", "HI"),
    ],
)
def test_yaml_inlined_step_runs_the_op_wrapped(body: str, expected: str) -> None:
    row = "hi" if expected == "HI" else '{"name": "q"}'
    assert _run_inlined(body, row) == expected


def test_an_sdk_callable_is_validated_before_wrapping() -> None:
    sdk = lambda inputs: str(inputs[0]).lower()  # noqa: E731
    assert lambda_runtime.callable_source(sdk, "f(lambda a: a)") == "lambda a: a"
    code = lambda_runtime.callable_source(_sdk_imports, inspect.getsource(_sdk_imports))
    with pytest.raises(ValueError, match="imports 'math'"):
        lambda_runtime.validate_inline_source(code, _sdk_imports.__name__)


_TABLE = json.dumps({"name": {"0": "a", "2": "b", "10": "c"}, "tag": "x"})
_VALUES = ['{"a": 1}', "plain {x", "1", "[1, 2]", {"table": {"df": _TABLE}}]
_VALUES += [{"table": json.dumps({"df": _TABLE})}, {"df": "not json"}]
_VALUES += [{"df": "[1]"}, {"df": _TABLE}]
_VALUES += [{"table": "not json"}, {"df": json.dumps({"n": {"a": 1, "b": 2}})}]


@pytest.mark.parametrize("value", _VALUES)
def test_inline_wrapper_and_shared_decoder_agree(value: Any) -> None:
    namespace: dict[str, Any] = {"json": json}
    exec(lambda_runtime.DECODER_SOURCE, namespace)
    assert namespace["_decode"](value) == decode(value)
