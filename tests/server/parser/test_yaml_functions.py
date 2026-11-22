from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.utils import lambda_runtime

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


def test_an_inline_lambda_passes_the_inline_source_check_with_its_blocks() -> None:
    spec = _workflow()
    spec["functions"] = {"prefix": "PREFIX = 'seed:'\n"}
    spec["ops"] = [
        {
            "id": "op",
            "op": "LambdaOp",
            "fn_name": "op",
            "inputs": ["rows"],
            "uses": ["prefix"],
            "code": "def op(inputs):\n    return PREFIX + inputs[0]\n",
        }
    ]
    spec["outputs"] = [{"name": "result", "ref": "op"}]
    assert lambda_runtime.validate_inline_source(_lambda_ops(spec)[0].code, "op")


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
