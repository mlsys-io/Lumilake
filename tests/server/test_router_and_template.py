"""Router expansion and template-driven rounds through the production path."""

import json
from pathlib import Path
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
from lumilake_server.ops.router_ops import expand_routers
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime import python_step

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
    tmp_path: Path, entry: dict[str, Any], branch_id: str, column: list[Any]
) -> list[Any]:
    """Run a branch's list step over ``column`` as JSON text."""
    op = next(item for item in expand_routers([entry]) if item["id"] == branch_id)
    plan = [{"kind": "literal", "values": [json.dumps(v) for v in column]}]
    source = python_step.wrapper_source(
        op["id"], op["fn_name"], op["code"], plan, "list"
    )
    mount = tmp_path / "runtime.py"
    mount.write_bytes(python_step.runtime_blob().body)
    namespace: dict[str, Any] = {}
    exec(source.replace(python_step.TASK_RUNTIME_MOUNT, str(mount)), namespace)
    return [i["output"] for i in namespace["main"]({})["items"]]


def test_expand_routers_emits_one_list_lambda_per_branch() -> None:
    ops, cleared = expand_routers([_router()])
    for op in ops:
        assert (op["op"], op["mode"], op["inputs"]) == ("LambdaOp", "list", ["norm"])
    assert [op["id"] for op in ops] == [
        "route.patient",
        "route.diagnose",
        "route.test",
    ]
    assert cleared == frozenset(op["id"] for op in ops)
    error, error_cleared = expand_routers([_router(error_branch=True)])
    assert [op["id"] in error_cleared for op in error] == [True, True, True, False]


def test_branch_keeps_its_rule_rows_and_the_default_takes_the_rest(tmp_path) -> None:
    for branch_id, expected in (
        ("route.patient", ["a"]),
        ("route.diagnose", ["b"]),
        ("route.test", ["c", "d"]),
    ):
        got = _branch_rows(tmp_path, _router(), branch_id, list(_ROWS))
        assert [row["case_id"] for row in got] == expected
    assert _branch_rows(tmp_path, _router(), "route.patient", []) == []


def test_a_text_row_mentioning_the_marker_key_keeps_its_own_branch() -> None:
    # Only an object whose sole key is `_fail` is a marker, and `error` is a
    # branch name only when the op declares the branch.
    rows = [{"case_id": "e", "text": 'the field "_fail" is reserved'}]
    field = [{"case_id": "f", "text": "x", "_fail": "a user field"}]
    for row in (*rows, *field):
        assert _branch_rows(_router(), "route.patient", [json.dumps(row)]) == [row]
    rule = [{"branch": "error", "match": "equals", "value": "1"}]
    expand_routers([_router(rules=rule)])
    with pytest.raises(ValueError, match="error"):
        expand_routers([_router(rules=rule, error_branch=True)])


_EMPTY_RULE: list[dict[str, Any]] = [{"branch": "no_ret", "match": "empty"}]
_EMPTY_ROWS: list[tuple[str, Any]] = [
    ("null", None),
    ("blank", ""),
    ("list", []),
    ("map", {}),
    ("null-text", "null"),
    ("full", "q"),
]


def test_an_empty_rule_refuses_a_value() -> None:
    rule = {"branch": "no_ret", "match": "empty", "value": "x"}
    with pytest.raises(ValueError, match="branch 'no_ret'.*takes no value"):
        expand_routers([_router(rules=[rule])])


def test_an_empty_rule_routes_null_blank_and_empty_containers_only() -> None:
    router = _router(rules=_EMPTY_RULE, default_branch="ret")
    column = [json.dumps({"case_id": n, "text": t}) for n, t in _EMPTY_ROWS]
    got = _branch_rows(router, "route.no_ret", column)
    assert [row["case_id"] for row in got] == ["null", "blank", "list", "map"]
    rest = _branch_rows(router, "route.ret", column)
    assert [row["case_id"] for row in rest] == ["null-text", "full"]


def test_a_row_missing_an_empty_rule_path_fails_naming_the_part() -> None:
    router = _router(path="ctx.question", rules=_EMPTY_RULE, default_branch="ret")
    with pytest.raises(ValueError, match=r"route\.no_ret row has no 'question'"):
        _branch_rows(router, "route.no_ret", [json.dumps({"ctx": {}})])


def test_router_rejects_a_default_rule_a_second_input_and_twin_branches() -> None:
    with pytest.raises(ValueError, match="default branch"):
        expand_routers([_router(default_branch="diagnose")])
    with pytest.raises(ValueError, match="exactly one input"):
        expand_routers([_router(inputs=[])])
    twin = [{"branch": "x", "match": "equals", "value": "1"}] * 2
    with pytest.raises(ValueError, match="branches must be unique"):
        expand_routers([_router(rules=twin)])


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
    with pytest.raises((DriverProtocolError, ValueError), match=message):
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
