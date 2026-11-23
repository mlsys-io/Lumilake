"""``RepeatOp`` unrolling through the YAML parser and the graph build."""

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
_JOIN = (
    "      - {id: Gated, op: JoinOp, inputs: [$in, Status.fail, Answer], by: unit}\n"
)


def _router(source: str, router_id: str = "Status", indent: str = "      ") -> str:
    return (
        f"{indent}- {{id: {router_id}, op: RouterOp, inputs: [{source}], path: text, "
        "default_branch: ok, rules: [{branch: fail, match: contains, value: x}]}\n"
    )


def _chat(chat_id: str, source: str, indent: str = "      ") -> str:
    return (
        f"{indent}- {{id: {chat_id}, op: LLMChatOp, inputs: [{source}], config:"
        f" {{model: m}}, messages: [{{role: user, content: x}}], rowwise_template: x,"
        f" rowwise_columns: [{{label: p, node: {source}}}]}}\n"
    )


def _op(step_id: str, inputs: str, indent: str = "  ") -> str:
    return f"{indent}- {_STEP.format(id=step_id, inputs=inputs)}\n"


def _repeat(
    times: str = "3",
    source: str = "Start",
    output: str = "Second",
    body: str | None = None,
    repeat_id: str = "R",
) -> str:
    if body is None:
        body = _op("First", "[$in]", "      ") + _op("Second", "[First]", "      ")
    return (
        f"  - id: {repeat_id}\n    op: RepeatOp\n    times: {times}\n"
        f"    input: {source}\n    output: {output}\n    body:\n{body}"
    )


def _graph(ops: str, outputs: str = "", *, round_number: int | None = None) -> dict:
    text = f"{_HEADER}ops:\n{ops}{outputs}"
    return parse_yaml_payload(text, round_number=round_number)["repeat"]["graph"]


def _named(graph: dict, name: str) -> dict:
    start = f"lambda_repeat_{name}_"
    return next(op for op in graph.values() if op["_id"].startswith(start))


def _second(source: str, body: str = "") -> str:
    body = body or _op("A", f"[{source}]", "      ") + _op("B", "[A]", "      ")
    return _repeat("2", source, "B", body, "S")


def test_a_repeat_reads_another_by_its_last_output() -> None:
    chained = _graph(_repeat(times="2") + _second("R"), _OUTPUT.replace("R", "S"))
    last_of_r = _named(chained, "R_2_Second")["_id"]
    assert _named(chained, "S_1_A")["_inputs"] == [last_of_r]
    output = next(op for op in chained.values() if op["_op"] == "OutputOp")
    assert output["_inputs"] == [_named(chained, "S_2_B")["_id"]]
    body = _op("A", "[R]", "      ") + _op("B", "[A]", "      ")
    in_body = _graph(_repeat(times="2") + _second("Start", body))
    assert _named(in_body, "S_1_A")["_inputs"] == [_named(in_body, "R_2_Second")["_id"]]


def test_a_message_starting_with_a_body_id_is_read_as_text() -> None:
    body = _op("Step", "[$in]", "      ")
    chat = "  - {id: Ask, op: LLMChatOp, inputs: [Start], config: {model: m},"
    chat += " messages: [{role: user, content: 'Step. Then rewrite it.'}]}\n"
    graph = _graph(_repeat(output="Step", body=body) + chat)
    op = next(op for op in graph.values() if op["_op"] == "LLMChatOp")
    message = graph[op["messages"]]["messages"]
    assert message == [{"role": "user", "content": "Step. Then rewrite it."}]


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


def _draws(round_number: int | None) -> list[str]:
    body = _chat("Ask", "$in").replace(
        "rowwise_template: x,",
        "rowwise_template: x, rowwise_seed: {key: {node: $in,"
        " path: items.output.value}, pairing: shared},",
    )
    graph = _graph(_repeat("2", "Start", "Ask", body), round_number=round_number)
    return [o["_code"] for o in graph.values() if o.get("fn_name") == "seed_for_row"]


def test_a_body_keyed_seed_draws_apart_per_iteration_and_round() -> None:
    static, round_2, round_3 = _draws(None), _draws(2), _draws(3)
    step = _chat("Ask", "$in").replace("model: m}", "model: m, seed: 7}")
    graph = _graph(_repeat("2", "Start", "Ask", step))
    literal = [o["config"]["seed"] for o in graph.values() if o["_op"] == "LLMChatOp"]
    assert "[0, 'Ask', 1, value]" in static[0] and "[0, 'Ask', 2, value]" in static[1]
    assert round_2[0] != round_2[1] and round_2[0] != round_3[0]
    assert literal[0] != literal[1] and max(literal) < 2**31


def test_a_repeat_reads_a_router_branch_outside_it_beside_its_own_branch() -> None:
    body = _router("$in") + _chat("Answer", "Status.fail") + _JOIN
    gate = _router("Start", "Gate", "  ")
    graph = _graph(gate + _repeat("3", "Gate.fail", "Gated", body), _OUTPUT)
    ops = list(graph.values())
    fails = [op for op in ops if op.get("fn_name") == "route_fail"][1:]
    chats = [op for op in ops if op["_op"] == "LLMChatOp"]
    joins = [op for op in ops if op["_op"] == "JoinOp"]
    assert (len(fails), len(chats), len(joins)) == (3, 3, 3)
    nodes = [column["node"] for op in chats for column in op["rowwise_columns"]]
    assert nodes == [fail["_id"] for fail in fails]
    source = [next(op["_id"] for op in ops if op.get("fn_name") == "route_fail")]
    source += [join["_id"] for join in joins[:-1]]
    for index, join in enumerate(joins):
        assert join["_inputs"] == [
            source[index],
            fails[index]["_id"],
            chats[index]["_id"],
        ]


@pytest.mark.parametrize(
    ("ops", "message"),
    [
        pytest.param(_repeat(times="0"), "positive integer"),
        pytest.param(_repeat(times="1000"), "over the limit of 1000"),
        pytest.param(
            _repeat().replace("    times:", "    steps: 3\n    times:"), "unknown"
        ),
        pytest.param(_repeat(body="      - [First]\n"), "must be a mapping"),
        pytest.param(
            _repeat().replace("  - id: R\n", "  - op: RepeatOp\n"), "missing 'id'"
        ),
        pytest.param(_repeat(body=""), "requires a non-empty 'body' list"),
        pytest.param(
            _repeat(body=_router("$in") + "      - {id: Inner, op: RepeatOp}\n"),
            "may not hold a 'RepeatOp'",
        ),
        pytest.param(
            _repeat(body=_op("A.B", "[$in]", "      ")), "may not contain '.'"
        ),
        pytest.param(_repeat(output="Missing"), "is not one of its body ids"),
        pytest.param(_repeat(output="Status", body=_router("$in")), "is a body router"),
        pytest.param(
            _repeat(
                output="Ask",
                body="      - {id: Ask, op: LLMChatOp, inputs: [$in], config: {model:"
                " m, seed: '7'}, messages: [{role: user, content: x}]}\n",
            ),
            "config.seed: Input should be a valid integer",
        ),
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
            _repeat(output="B", body=_router("$in") + _op("B", "[$in]", "      "))
            + _op("After", "[Status.fail]"),
            "a body id of 'R'",
        ),
        pytest.param(
            _repeat(
                body=_op("First", "[R]", "      ") + _op("Second", "[First]", "      ")
            ),
            "body reads its own id",
        ),
        pytest.param(
            _repeat(source="S") + _second("R"),
            "cycle detected",
        ),
    ],
)
def test_a_repeat_that_cannot_be_unrolled_is_refused(ops: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _graph(ops)
