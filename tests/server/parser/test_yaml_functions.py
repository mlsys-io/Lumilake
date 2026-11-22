from typing import Any

import pytest
import yaml

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp
from lumilake_server.parser import parse_yaml_payload

_SEED = """\
def seed_value(row):
    return SEED_PREFIX + row["name"]
SEED_PREFIX = "seed:"
"""


def _workflow() -> dict[str, Any]:
    return {
        "name": "shared_functions",
        "functions": {"seed": _SEED},
        "inputs": {"rows": ["a", "b"]},
        "ops": [
            {
                "id": op_id,
                "op": "LambdaOp",
                "inputs": ["rows"],
                "fn_name": op_id,
                "uses": ["seed"],
                "code": (
                    f"def {op_id}(inputs):\n"
                    "    return seed_value({'name': inputs[0]})\n"
                ),
            }
            for op_id in ("first", "second")
        ],
        "outputs": [{"name": "result", "ref": "second"}],
    }


def _lambda_ops(spec: dict[str, Any]) -> list[LambdaOp]:
    parsed = parse_yaml_payload(yaml.safe_dump(spec, sort_keys=False))
    graph = Graph.from_json(next(iter(parsed.values()))["graph"])
    return [op for op in graph.iter_ops() if isinstance(op, LambdaOp)]


def test_a_used_function_is_prepended_to_each_op_that_uses_it() -> None:
    ops = _lambda_ops(_workflow())
    assert len(ops) == 2
    for op in ops:
        namespace: dict[str, Any] = {}
        exec(compile(op.code, "<shared>", "exec"), namespace)
        assert namespace[op.fn.__name__](("x",)) == "seed:x"


def test_used_functions_are_prepended_in_the_listed_order() -> None:
    spec = _workflow()
    spec["functions"] = {"tail": "TAIL = 't'\n", "head": "HEAD = 'h'\n"}
    for entry in spec["ops"]:
        entry.update(
            uses=["tail", "head"],
            code=f"def {entry['id']}(inputs):\n    return HEAD\n",
        )
    code = _lambda_ops(spec)[0].code
    assert code.index("TAIL = ") < code.index("HEAD = ") < code.index("def first")


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda spec: spec["ops"][0].update(uses=["missing"]), "unknown function"),
        (lambda spec: spec["ops"][0].update(uses=["seed", "seed"]), "more than once"),
        (lambda spec: spec["ops"][0].update(op="DataOp"), "only a LambdaOp"),
        (
            lambda spec: spec["functions"].update(broken="def broken(:\n"),
            "does not parse as Python",
        ),
    ],
)
def test_a_bad_use_is_refused(change: Any, match: str) -> None:
    spec = _workflow()
    change(spec)
    with pytest.raises(ValueError, match=match):
        parse_yaml_payload(yaml.safe_dump(spec, sort_keys=False))


@pytest.mark.parametrize(
    "change, match",
    [
        (
            lambda spec: spec["ops"][0].update(code="seed_value = None\n"),
            "which the function 'seed' it uses also defines",
        ),
        (
            lambda spec: spec["functions"].update(other="SEED_PREFIX = 'x'\n"),
            "both define 'SEED_PREFIX'",
        ),
    ],
)
def test_a_redefined_name_is_refused(change: Any, match: str) -> None:
    spec = _workflow()
    change(spec)
    if "other" in spec["functions"]:
        spec["ops"][0]["uses"] = ["seed", "other"]
    with pytest.raises(ValueError, match=match):
        parse_yaml_payload(yaml.safe_dump(spec, sort_keys=False))
