"""``Repeat`` unrolling through the YAML parser and the graph build."""

import pytest

from lumilake_server.graphs import Graph
from lumilake_server.ops import LambdaOp
from lumilake_server.parser import parse_yaml_payload

_HEADER = "name: repeat\ninputs:\n  Start: ['seed']\n"
_OUTPUT = "outputs:\n  - name: result\n    ref: R\n"
_STEP = (
    "{{id: {id}, op: LambdaOp, inputs: {inputs}, fn_name: f, "
    "code: 'def f(x): return x'}}"
)
_ROUTER = (
    "      - {id: First, op: RouterOp, inputs: [$in], path: text, "
    "default_branch: b, rules: [{branch: b, match: contains, value: x}]}\n"
)


def _op(step_id: str, inputs: str, indent: str = "  ") -> str:
    return f"{indent}- {_STEP.format(id=step_id, inputs=inputs)}\n"


def _repeat(
    times: str = "3",
    source: str = "Start",
    output: str = "Second",
    extra: str = "",
    body: str | None = None,
) -> str:
    if body is None:
        body = _op("First", "[$in]", "      ") + _op("Second", "[First]", "      ")
    return (
        f"  - id: R\n    op: Repeat\n    times: {times}\n    input: {source}\n"
        f"    output: {output}\n{extra}    body:\n{body}"
    )


def _graph(ops: str, outputs: str = "") -> dict:
    return parse_yaml_payload(f"{_HEADER}ops:\n{ops}{outputs}")["repeat"]["graph"]


def _named(graph: dict, name: str) -> dict:
    return next(
        op
        for op in graph.values()
        if op["_op"] == "LambdaOp" and op["_id"].startswith(f"lambda_repeat_{name}_")
    )


def _second(source: str, body: str | None = None) -> str:
    if body is None:
        body = _op("Lead", f"[{source}]", "      ") + _op("Last", "[Lead]", "      ")
    return (
        f"  - id: S\n    op: Repeat\n    times: 2\n    input: {source}\n"
        f"    output: Last\n    body:\n{body}"
    )


def test_a_chained_repeat_reads_another_repeats_last_output() -> None:
    outputs = _OUTPUT.replace("ref: R", "ref: S")
    graph = _graph(_repeat(times="2") + _second("R"), outputs)
    last_of_r = _named(graph, "R_2_Second")["_id"]
    assert _named(graph, "S_1_Lead")["_inputs"] == [last_of_r]
    assert _named(graph, "S_2_Lead")["_inputs"] == [last_of_r]
    output = next(op for op in graph.values() if op["_op"] == "OutputOp")
    assert output["_inputs"] == [_named(graph, "S_2_Last")["_id"]]


def test_a_body_reads_another_repeat_by_its_last_output() -> None:
    body = _op("Lead", "[R]", "      ") + _op("Last", "[Lead]", "      ")
    graph = _graph(_repeat(times="2") + _second("Start", body))
    lead = _named(graph, "S_2_Lead")
    assert lead["_inputs"] == [_named(graph, "R_2_Second")["_id"]]


def test_a_literal_whose_first_part_is_a_body_id_is_read_as_text() -> None:
    fmt = (
        "  - {id: Fmt, op: FormatOp, inputs: [Start], template: '{doc}', "
        "format_kwargs: {doc: 'Step.json'}}\n"
    )
    graph = _graph(_repeat() + fmt, _OUTPUT)
    op = next(op for op in graph.values() if op["_op"] == "FormatOp")
    assert op["format_kwargs"] == {"doc": "Step.json"}


def test_a_two_op_body_runs_three_times_with_each_iteration_feeding_the_next() -> None:
    graph = _graph(_repeat(), _OUTPUT)
    lambdas = sorted(
        (op for op in graph.values() if op["_op"] == "LambdaOp"),
        key=lambda op: op["_id"].removeprefix("lambda_repeat_"),
    )
    source = next(op["_id"] for op in graph.values() if op["_op"] == "InputOp")
    assert lambdas[0]["_inputs"] == [source]
    assert [op["_inputs"] for op in lambdas[1:]] == [[p["_id"]] for p in lambdas[:-1]]
    output = next(op for op in graph.values() if op["_op"] == "OutputOp")
    assert output["_inputs"] == [lambdas[-1]["_id"]]
    compiled = Graph.from_json(graph).compile(Start=["seed"])
    assert len(list(compiled.iter_ops(LambdaOp))) == 6


@pytest.mark.parametrize(
    ("ops", "message"),
    [
        pytest.param(_repeat(times="0"), "positive integer"),
        pytest.param(_repeat(extra="    steps: 3\n"), "unknown fields"),
        pytest.param(_repeat(body="      - [First]\n"), "must be a mapping"),
        pytest.param(
            _repeat().replace("  - id: R\n", "  - op: Repeat\n"), "missing 'id'"
        ),
        pytest.param(_repeat(body=""), "requires a non-empty 'body' list"),
        pytest.param(_repeat(body=_ROUTER), "may not hold a 'RouterOp'"),
        pytest.param(
            _repeat(body=_op("A.B", "[$in]", "      ")), "may not contain '.'"
        ),
        pytest.param(_repeat(output="Missing"), "is not one of its body ids"),
        pytest.param(_repeat(source="Nowhere"), "is unknown"),
        pytest.param(_repeat(source=""), "requires a non-empty 'input'"),
        pytest.param(_repeat(source="$in"), r"\$in"),
        pytest.param(_repeat() + _op("R", "[Start]"), "duplicate id 'R'"),
        pytest.param(
            _op("First", "[Start]") + _repeat(), "collides with an id outside"
        ),
        pytest.param(_op("After", "[$in]") + _repeat(), r"may not read '\$in'"),
        pytest.param(_repeat() + _op("After", "[Second]"), r"a body id of 'R'"),
        pytest.param(
            _repeat(source="S") + _second("R"),
            "cycle detected",
        ),
    ],
)
def test_a_repeat_that_cannot_be_unrolled_is_refused(ops: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _graph(ops)
