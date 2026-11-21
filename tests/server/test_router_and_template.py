"""Router expansion and template-driven rounds through the production path."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from lumilake_server.dynamic.driver import DriverProtocolError
from lumilake_server.dynamic.spec import DynamicSpec
from lumilake_server.dynamic.template import (
    archived_element_count,
    plan_for_round,
    stop_export,
    validate_template,
)
from lumilake_server.ops.router_ops import RouterOp, expand_routers
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.utils.lambda_runtime import materialize

_ROWS = [
    {"case_id": "a", "text": "hello"},
    {"case_id": "b", "text": "DIAGNOSIS READY: flu"},
    {"case_id": "c", "text": "REQUEST TEST: cbc"},
    {"case_id": "d", "text": "REQUEST TEST"},
]


def _router(**extra: Any) -> dict[str, Any]:
    return {
        "id": "route",
        "op": "RouterOp",
        "inputs": ["norm"],
        "path": "text",
        "default_branch": "patient",
        "rules": [
            {"branch": "diagnose", "match": "contains", "value": "DIAGNOSIS READY"},
            {"branch": "test", "match": "contains", "value": "REQUEST TEST"},
        ],
        **extra,
    }


def _branch_rows(
    entry: dict[str, Any], branch_id: str, column: list[str]
) -> list[dict[str, Any]]:
    """Run one expanded branch's generated source over ``column``."""
    op = next(item for item in expand_routers([entry]) if item["id"] == branch_id)
    return materialize(op["code"], op["fn_name"])((column,))


def test_expand_routers_emits_one_list_lambda_per_branch() -> None:
    for op in expand_routers([_router()]):
        assert (op["op"], op["mode"], op["inputs"]) == ("LambdaOp", "list", ["norm"])
    assert [op["id"] for op in expand_routers([_router()])] == [
        "route.patient",
        "route.diagnose",
        "route.test",
    ]


def test_branch_keeps_its_rule_rows_and_the_default_takes_the_rest() -> None:
    column = [json.dumps(row) for row in _ROWS]
    # A wrapped whole list is unwrapped by the same decoder; no rows yields none.
    assert _branch_rows(_router(), "route.patient", [json.dumps(_ROWS)])[0] == _ROWS[0]
    assert _branch_rows(_router(), "route.patient", []) == []
    for branch_id, expected in (
        ("route.patient", ["a"]),
        ("route.diagnose", ["b"]),
        ("route.test", ["c", "d"]),
    ):
        got = _branch_rows(_router(), branch_id, column)
        assert [row["case_id"] for row in got] == expected


def test_a_text_row_mentioning_the_marker_key_keeps_its_own_branch() -> None:
    rows = [{"case_id": "e", "text": 'the field "_fail" is reserved'}]
    column = [json.dumps(row) for row in rows]
    assert _branch_rows(_router(), "route.patient", column) == rows
    field = [{"case_id": "f", "text": "x", "_fail": "a user field"}]
    assert _branch_rows(_router(), "route.patient", [json.dumps(field[0])]) == field


def test_a_router_names_the_error_branch_only_when_it_declares_one() -> None:
    rule = [{"branch": "error", "match": "equals", "value": "1"}]
    RouterOp.from_entry(_router(rules=rule))
    with pytest.raises(ValueError, match="error"):
        RouterOp.from_entry(_router(rules=rule, error_branch=True))


def test_router_rejects_a_default_rule_a_second_input_and_twin_branches() -> None:
    with pytest.raises(ValueError, match="default branch"):
        RouterOp.from_entry(_router(default_branch="diagnose"))
    with pytest.raises(ValueError, match="exactly one input"):
        RouterOp.from_entry(_router(inputs=[]))
    twin = [{"branch": "x", "match": "equals", "value": "1"}] * 2
    with pytest.raises(ValueError, match="branches must be unique"):
        RouterOp.from_entry(_router(rules=twin))


def test_router_expands_before_the_graph_is_built() -> None:
    workflow = {
        "name": "r",
        "inputs": {"Symbols": ["x"]},
        "ops": [
            {
                "id": "norm",
                "op": "LambdaOp",
                "mode": "list",
                "inputs": ["Symbols"],
                "fn_name": "norm",
                "code": "def norm(c):\n    return c[0]\n",
            },
            _router(),
            {
                "id": "turn",
                "op": "LambdaOp",
                "mode": "list",
                "inputs": ["route.test"],
                "fn_name": "turn",
                "code": "def turn(c):\n    return c[0]\n",
            },
        ],
        "outputs": [{"name": "out", "ref": "turn"}],
    }
    graph = parse_yaml_payload(workflow)["r"]["graph"]
    kinds = [node["_op"] for node in graph.values()]
    assert "RouterOp" not in kinds and kinds.count("LambdaOp") == 5
    (test_branch,) = [op_id for op_id in graph if "route_test" in op_id]
    turn = next(node for node in graph.values() if "turn" in node["_id"])
    assert turn["_inputs"] == [test_branch]


def _state(op_id: str, inputs: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "id": op_id,
        "op": "LambdaOp",
        "inputs": inputs,
        "mode": "list",
        "fn_name": "state",
        "code": "def state(c):\n    return c[0]\n",
        **extra,
    }


def _spec(first: list[dict[str, Any]], next_ops: list[dict[str, Any]]) -> DynamicSpec:
    return DynamicSpec.model_validate(
        {
            "type": "dynamic",
            "goal": "g",
            "driver": {
                "model": "m",
                "template": {
                    "first": {"ops": first, "export": ["Live_r{{round}}"]},
                    "next": {"ops": next_ops, "export": ["Live_r{{round}}"]},
                    "stop_when_empty": "Live_r{{prev}}",
                },
            },
        }
    )


def test_template_renders_the_round_suffix_and_the_stop_export() -> None:
    echo = _state("Live_r{{round}}", ["Symbols"], code="# {{round}}\ndef echo(c):\n")
    template = _spec(
        [echo],
        [_state("Live_r{{round}}", ["Live_r{{prev}}"])],
    ).driver.template
    assert template is not None
    (op,) = plan_for_round(template, 1).ops
    assert plan_for_round(template, 1).export == ["Live_r0"]
    assert plan_for_round(template, 2).ops[0]["inputs"] == ["Live_r0"]
    assert (stop_export(template, 2), stop_export(template, 3)) == (
        "Live_r0",
        "Live_r1",
    )
    # A Lambda's ``code`` is Python source, not a node reference, so it is kept.
    assert (op["id"], op["code"]) == ("Live_r0", "# {{round}}\ndef echo(c):\n")


def test_template_and_api_are_mutually_exclusive() -> None:
    body = _spec([_state("Live_r{{round}}", ["Symbols"])], []).model_dump()
    body["driver"]["api"] = {"url": "http://planner.test"}
    with pytest.raises(ValidationError, match="mutually exclusive"):
        DynamicSpec.model_validate(body)


@pytest.mark.parametrize(
    ("first", "next_ops", "message"),
    [
        (
            [_state("Live_r{{round}}", ["Symbols"], mode="row")],
            [_state("Live_r{{round}}", ["Symbols"])],
            "list, expand or reduce",
        ),
        (
            [_state("Live_r{{round}}", ["Symbols"])],
            [_state("Live_r{{round}}", ["Symbols"], mode="row")],
            "list, expand or reduce",
        ),
        (
            [
                {
                    "id": "Live_r{{round}}",
                    "op": "DataRetrievalOp",
                    "inputs": ["Symbols"],
                    "data_spec": {"type": "lumid", "mode": "sql", "template": "S"},
                }
            ],
            [_state("Live_r{{round}}", ["Symbols"])],
            "must be a LambdaOp",
        ),
        (
            [_state("Other_r{{round}}", ["Symbols"])],
            [_state("Live_r{{round}}", ["Symbols"])],
            "exported",
        ),
    ],
    ids=["first-row", "next-row", "retrieval", "not-exported"],
)
def test_template_refuses_a_body_at_submit(
    first: list[dict[str, Any]], next_ops: list[dict[str, Any]], message: str
) -> None:
    template = _spec(first, next_ops).driver.template
    assert template is not None
    with pytest.raises(DriverProtocolError, match=message):
        validate_template(template, None, max_nodes_per_round=8)


def test_template_accepts_a_list_lambda_export_and_counts_its_rows() -> None:
    template = _spec(
        [_state("Live_r{{round}}", ["Symbols"])],
        [_state("Live_r{{round}}", ["Live_r{{prev}}"])],
    ).driver.template
    assert template is not None
    validate_template(template, None, max_nodes_per_round=8)
    # A list-mode export archives one item holding the list its step returned.
    assert archived_element_count([{"output": [{"case_id": 1}, {"case_id": 2}]}]) == 2
    assert archived_element_count([{"output": []}]) == 0


def _merged(
    op_id: str, inputs: list[str], attach: dict[str, str] | None = None
) -> dict[str, Any]:
    op: dict[str, Any] = {
        "id": op_id,
        "op": "JoinOp",
        "inputs": inputs,
        "by": "case_id",
    }
    if attach is not None:
        op["attach"] = attach
    return op


def test_template_accepts_a_join_as_the_round_export_and_stop_leaf() -> None:
    first = [
        _state("Live_r{{round}}", ["Symbols"]),
        _state("Turn_r{{round}}", ["Live_r{{round}}"], mode="row"),
        _merged(
            "Merged_r{{round}}",
            ["Live_r{{round}}", "Turn_r{{round}}"],
            {"note": "Turn_r{{round}}"},
        ),
    ]
    template = _spec(
        first,
        [
            _state("Live_r{{round}}", ["Live_r{{prev}}"]),
            _state("Turn_r{{round}}", ["Live_r{{round}}"], mode="row"),
            _merged(
                "Merged_r{{round}}",
                ["Live_r{{round}}", "Turn_r{{round}}"],
                {"note": "Turn_r{{round}}"},
            ),
        ],
    ).driver.template
    assert template is not None
    template.first.export = ["Merged_r{{round}}"]
    template.next.export = ["Merged_r{{round}}"]
    template.stop_when_empty = "Merged_r{{prev}}"
    validate_template(template, None, max_nodes_per_round=8)
