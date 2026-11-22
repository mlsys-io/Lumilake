"""``Repeat`` unrolling through the YAML parser and the graph build."""

import pytest

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp
from lumilake_server.parser import parse_yaml_payload

_HEADER = "name: repeat\ninputs:\n  Start: ['seed']\n"
_OUTPUT = "outputs:\n  - name: result\n    ref: R\n"
_LAMBDA = (
    "{{id: {id}, op: LambdaOp, inputs: {inputs}, fn_name: f, "
    "code: 'def f(x): return x'}}"
)
_FIRST = _LAMBDA.format(id="First", inputs="[$in]")
_SECOND = _LAMBDA.format(id="Second", inputs="[First]")
_OUTER = """\
  - id: Outer
    op: Repeat
    times: 2
    input: Start
    output: R
    body:
{body}"""


def _repeat(times: str = "3", source: str = "Start", output: str = "Second") -> str:
    body = f"      - {_FIRST}\n      - {_SECOND}\n"
    return (
        f"  - id: R\n    op: Repeat\n    times: {times}\n"
        f"    input: {source}\n    output: {output}\n    body:\n{body}"
    )


def _graph(ops: str, outputs: str = "") -> dict:
    return parse_yaml_payload(f"{_HEADER}ops:\n{ops}{outputs}")["repeat"]["graph"]


def _lambdas(graph: dict) -> list[dict]:
    """The body ops in iteration order, read off their ``R.<n>.<bodyId>`` ids."""
    return sorted(
        (op for op in graph.values() if op["_op"] == "LambdaOp"),
        key=lambda op: op["_id"].removeprefix("lambda_repeat_"),
    )


def _input_id(graph: dict) -> str:
    return next(op["_id"] for op in graph.values() if op["_op"] == "InputOp")


def test_a_two_op_body_runs_three_times_with_each_iteration_feeding_the_next() -> None:
    graph = _graph(_repeat(), _OUTPUT)
    ops = _lambdas(graph)
    assert ops[0]["_inputs"] == [_input_id(graph)]
    assert [op["_inputs"] for op in ops[1:]] == [[prev["_id"]] for prev in ops[:-1]]
    output = next(op for op in graph.values() if op["_op"] == "OutputOp")
    assert output["_inputs"] == [ops[-1]["_id"]]
    compiled = Graph.from_json(graph).compile(Start=["seed"])
    assert len(list(compiled.iter_ops(LambdaOp))) == 6


def test_a_single_iteration_reads_its_input_once() -> None:
    graph = _graph(_repeat(times="1"))
    ops = _lambdas(graph)
    assert [op["_inputs"] for op in ops] == [[_input_id(graph)], [ops[0]["_id"]]]


@pytest.mark.parametrize(
    ("ops", "message"),
    [
        pytest.param(_repeat(times="0"), "positive integer", id="times-zero"),
        pytest.param(_repeat(times='"3"'), "positive integer", id="times-string"),
        pytest.param(_repeat(output="Missing"), "is not a body id", id="bad-output"),
        pytest.param(
            _repeat().replace("id: Second", "id: A.B"), "needs no '.'", id="dotted-id"
        ),
        pytest.param(_repeat(source="Nowhere"), "is unknown", id="unknown-input"),
        pytest.param(
            _repeat().replace("    times: 3", "    times: 3\n    steps: 3"),
            "unknown fields",
            id="unknown-field",
        ),
        pytest.param(
            _repeat() + f"  - {_LAMBDA.format(id='After', inputs='[Second]')}\n",
            r"a body id of 'R'",
            id="body-id-read-outside",
        ),
        pytest.param(_repeat(source="$in"), r"\$in", id="input-ref-as-source"),
        pytest.param(
            f"  - {_LAMBDA.format(id='First', inputs='[Start]')}\n" + _repeat(),
            "is outside",
            id="body-id-collides-outside",
        ),
        pytest.param(
            _repeat().replace(
                f"      - {_FIRST}",
                "      - {id: First, op: RouterOp, inputs: [$in], path: text, "
                "default_branch: b, rules: [{branch: b, match: contains, value: x}]}",
            ),
            "may not hold a 'RouterOp'",
            id="router-in-body",
        ),
        pytest.param(
            _OUTER.format(
                body="      - id: Inner\n        op: Repeat\n        times: 2\n"
                "        input: Start\n        output: T\n        body:\n"
                "          - {id: T, op: LambdaOp, inputs: [$in], fn_name: t, "
                "code: 'def t(x): return x'}\n"
            ),
            "may not hold a 'Repeat'",
            id="nested-repeat",
        ),
    ],
)
def test_a_repeat_that_cannot_be_unrolled_is_refused(ops: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _graph(ops)
