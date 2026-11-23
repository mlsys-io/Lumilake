from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

_SEED = 'def seed_value(row):\n    return PREFIX + row["name"]\nPREFIX = "seed:"\n'
_LAMBDA: dict[str, Any] = {"op": "LambdaOp", "inputs": ["rows"], "uses": ["seed"]}
_CHAT: dict[str, Any] = {
    "id": "chat",
    "op": "LLMChatOp",
    "config": {"model": "m"},
    "messages": [{"role": "user", "content": "op1"}],
}
_TRAILING = "def op1(i):\n    return helper(i)\n\n\ndef helper(r):\n    return r\n"
_NESTED = "def op1(i):\n    def helper(r):\n        return r\n    return helper(i)\n"


def _spec(code: str, inline: bool = False) -> dict[str, Any]:
    ops = [{**_LAMBDA, "id": "op1", "fn_name": "op1", "code": code}]
    if inline:
        ops = ops[:1] + [_CHAT]
    return {
        "functions": {"seed": _SEED, "paths": "import os.path\n"},
        "inputs": {"rows": ["a"]},
        "ops": ops,
        "outputs": [{"name": "out", "ref": ops[-1]["id"]}],
    }


def _run(spec: dict[str, Any]) -> list[str]:
    text = yaml.safe_dump(spec, sort_keys=False)
    parsed = next(iter(parse_yaml_payload(text).values()))
    graph = Graph.from_json(parsed["graph"])
    RuntimeGraphBuilder().build(graph.compile(**parsed["inputs"]))
    return [op.code for op in graph.iter_ops() if isinstance(op, LambdaOp)]


def test_blocks_are_appended_in_use_order_and_each_using_op_runs_them() -> None:
    code = "def op(inputs):\n    return TAIL + HEAD"
    spec = _spec(code)
    spec["functions"] = {"tail": "TAIL = 't'\n", "head": "HEAD = 'h'\n"}
    spec["ops"] = [
        {**spec["ops"][0], "id": f"op{i}", "uses": ["tail", "head"], "fn_name": "op"}
        for i in (1, 2)
    ]
    for source in _run(spec):
        assert source == f"{code}\n\nTAIL = 't'\n\nHEAD = 'h'"
        namespace: dict[str, Any] = {}
        exec(compile(source, "<shared>", "exec"), namespace)
        assert namespace["op"](None) == "th"


def test_a_dotted_import_and_its_package_are_the_same_target() -> None:
    spec = _spec("import os\ndef op1(i):\n    return i\n")
    spec["ops"][0]["uses"] = ["paths"]
    (source,) = _run(spec)
    assert source.endswith("import os.path")


@pytest.mark.parametrize(
    "edit, match",
    [
        (
            {
                "code": "def op1(i):\n    return PREFIX + i[0]\n",
                "uses": ["prefix"],
                "inline": True,
            },
            "it binds 'PREFIX'",
        ),
        ({"code": _TRAILING, "uses": [], "inline": True}, "it binds 'helper'"),
        ({"code": _NESTED, "uses": [], "inline": True}, None),
        ({"uses": ["nope"]}, "unknown function"),
        ({"uses": ["seed", "seed"]}, "more than once"),
        ({"op": "DataOp"}, "only a LambdaOp"),
        ({"uses": ["seed"], "code": "PREFIX = 'x'\n"}, "it uses also defines"),
        ({"uses": ["seed"], "code": "lambda r: r[0]"}, "lambda-expression code"),
        ({"functions": {"bad": "def bad(:\n"}}, "does not parse"),
        ({"functions": {"o": "PREFIX = 'x'\n"}, "uses": ["seed", "o"]}, "both define"),
    ],
)
def test_uses_and_the_inline_rule_are_enforced_at_submit(
    edit: dict[str, Any], match: str | None
) -> None:
    spec = _spec(
        edit.get("code", "def op1(i):\n    return seed_value(i)\n"),
        edit.get("inline", False),
    )
    spec["functions"]["prefix"] = "PREFIX = 'x'\n"
    spec["functions"].update(edit.get("functions", {}))
    spec["ops"][0].update(
        {k: v for k, v in edit.items() if k not in ("functions", "inline")}
    )
    if match is None:
        _run(spec)
        return
    with pytest.raises(ValueError, match=match):
        _run(spec)


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"op": "DataOp", "data": ["x"]}, "id: Field required"),
        ({"id": "a", "data": ["x"]}, "op: Field required"),
        ({"id": "  ", "op": "DataOp"}, "id: must be a non-empty string"),
        ({"id": "a", "op": "DataOp", "inputs": "nope"}, "inputs: Input should be a"),
        ({"id": "a", "op": "DataOp", "max_iter": "soon"}, "max_iter: Input should"),
    ],
)
def test_an_op_entry_field_is_refused_naming_the_entry_and_field(
    entry: dict[str, Any], match: str
) -> None:
    text = yaml.safe_dump({"ops": [entry]}, sort_keys=False)
    with pytest.raises(ValueError, match=match):
        parse_yaml_payload(text)
