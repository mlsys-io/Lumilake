from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp
from lumilake_server.parser import parse_yaml_payload

_SEED = 'def seed_value(row):\n    return PREFIX + row["name"]\nPREFIX = "seed:"\n'


def _workflow() -> dict[str, Any]:
    return {
        "name": "shared",
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


def test_blocks_are_prepended_in_use_order_and_each_using_op_runs_them() -> None:
    spec = _workflow()
    code = "def op(inputs):\n    return TAIL + HEAD"
    spec["functions"] = {"tail": "TAIL = 't'\n", "head": "HEAD = 'h'\n"}
    for entry in spec["ops"]:
        entry.update(fn_name="op", uses=["tail", "head"], code=code)
    ops = _lambda_ops(spec)
    assert len(ops) == 2
    for op in ops:
        assert op.code == f"TAIL = 't'\n\nHEAD = 'h'\n\n{code}"
        namespace: dict[str, Any] = {}
        exec(compile(op.code, "<shared>", "exec"), namespace)
        assert namespace["op"](None) == "th"


@pytest.mark.parametrize(
    "edit, match",
    [
        ({"uses": ["nope"]}, "unknown function"),
        ({"uses": ["seed", "seed"]}, "more than once"),
        ({"op": "DataOp"}, "only a LambdaOp"),
        ({"code": "PREFIX = 'x'\n"}, "it uses also defines"),
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
