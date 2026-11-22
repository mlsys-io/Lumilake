from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

_SEED = 'def seed_value(row):\n    return PREFIX + row["name"]\nPREFIX = "seed:"\n'


def _workflow() -> dict[str, Any]:
    return {
        "functions": {"seed": _SEED},
        "inputs": {"rows": ["a"]},
        "ops": [
            {
                "id": f"op{i}",
                "op": "LambdaOp",
                "fn_name": f"op{i}",
                "inputs": ["rows"],
                "uses": ["seed"],
                "code": f"def op{i}(inputs):\n    return seed_value(inputs)\n",
            }
            for i in (1, 2)
        ],
        "outputs": [{"name": "result", "ref": "op2"}],
    }


def _lambda_ops(spec: dict[str, Any]) -> list[LambdaOp]:
    specs = parse_yaml_payload(yaml.safe_dump(spec, sort_keys=False))
    graph = Graph.from_json(next(iter(specs.values()))["graph"])
    return [op for op in graph.iter_ops() if isinstance(op, LambdaOp)]


def _build_runtime(spec: dict[str, Any]) -> Any:
    parsed = parse_yaml_payload(yaml.safe_dump(spec, sort_keys=False))
    compiled = next(iter(parsed.values()))
    graph = Graph.from_json(compiled["graph"])
    return RuntimeGraphBuilder().build(graph.compile(**compiled["inputs"]))


def test_blocks_are_appended_in_use_order_and_each_using_op_runs_them() -> None:
    code = "def op(inputs):\n    return TAIL + HEAD"
    spec = _workflow()
    spec["functions"] = {"tail": "TAIL = 't'\n", "head": "HEAD = 'h'\n"}
    for entry in spec["ops"]:
        entry.update(fn_name="op", uses=["tail", "head"], code=code)
    ops = _lambda_ops(spec)
    for op in ops:
        assert op.code == f"{code}\n\nTAIL = 't'\n\nHEAD = 'h'"
        namespace: dict[str, Any] = {}
        exec(compile(op.code, "<shared>", "exec"), namespace)
        assert namespace["op"](None) == "th"


def test_an_inline_lambda_cannot_use_a_block() -> None:
    spec = _workflow()
    spec["functions"] = {"prefix": "PREFIX = 'seed:'\n"}
    spec["ops"] = spec["ops"][:1]
    spec["ops"][0].update(
        uses=["prefix"], code="def op1(inputs):\n    return PREFIX + inputs[0]\n"
    )
    spec["ops"].append(
        {
            "id": "chat",
            "op": "LLMChatOp",
            "inputs": ["op1"],
            "messages": [{"role": "user", "content": "op1"}],
            "config": {"model": "meta-llama/Llama-3.1-8B-Instruct"},
        }
    )
    spec["outputs"] = [{"name": "reply", "ref": "chat"}]
    with pytest.raises(ValueError, match=r"Invalid LambdaOp '[^']*': it uses the"):
        _build_runtime(spec)


def test_a_dotted_import_and_its_package_are_the_same_target() -> None:
    spec = _workflow()
    spec["functions"] = {"paths": "import os.path\n"}
    spec["ops"] = spec["ops"][:1]
    spec["ops"][0].update(
        uses=["paths"], code="import os\ndef op1(inputs):\n    return inputs\n"
    )
    spec["outputs"] = [{"name": "result", "ref": "op1"}]
    (op,) = _lambda_ops(spec)
    assert op.code.endswith("import os.path")


@pytest.mark.parametrize(
    "edit, match",
    [
        ({"uses": ["nope"]}, "unknown function"),
        ({"uses": ["seed", "seed"]}, "more than once"),
        ({"op": "DataOp"}, "only a LambdaOp"),
        ({"uses": ["seed"], "code": "PREFIX = 'x'\n"}, "it uses also defines"),
        ({"uses": ["seed"], "code": "def op(inputs)\n"}, "does not parse"),
        ({"uses": ["seed"], "code": "lambda r: r[0]"}, "lambda-expression code"),
        ({"functions": {"bad": "def bad(:\n"}}, "does not parse"),
        ({"functions": {"o": "PREFIX = 'x'\n"}, "uses": ["seed", "o"]}, "both define"),
    ],
)
def test_a_bad_use_is_refused(edit: dict[str, Any], match: str) -> None:
    spec = _workflow()
    spec["functions"].update(edit.get("functions", {}))
    spec["ops"][0].update({k: v for k, v in edit.items() if k != "functions"})
    with pytest.raises(ValueError, match=match):
        _lambda_ops(spec)
