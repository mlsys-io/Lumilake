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


def _run_branch(entry: dict[str, Any], column: list[str]) -> list[dict[str, Any]]:
    """Run one expanded branch's generated source over ``column``."""
    ops = expand_routers([entry])
    op = next(item for item in ops if item["id"] == "route.patient")
    fn = materialize(op["code"], op["fn_name"])
    return fn((column,))


def test_expand_routers_emits_one_list_lambda_per_branch() -> None:
    expanded = expand_routers([_router()])
    assert [op["id"] for op in expanded] == [
        "route.patient",
        "route.diagnose",
        "route.test",
    ]
    for op in expanded:
        assert op["op"] == "LambdaOp"
        assert op["mode"] == "list"
        assert op["inputs"] == ["norm"]
    assert [op for op in expand_routers([])] == []


def test_branch_keeps_its_rule_rows_and_the_default_takes_the_rest() -> None:
    column = [json.dumps(row) for row in _ROWS]
    assert [row["case_id"] for row in _run_branch(_router(), column)] == ["a"]
    diagnose = _run_branch(_router(), column) and [
        row["case_id"] for row in _branch_rows(_router(), "route.diagnose", column)
    ]
    assert diagnose == ["b"]
    assert [
        row["case_id"] for row in _branch_rows(_router(), "route.test", column)
    ] == ["c", "d"]


def _branch_rows(
    entry: dict[str, Any], branch_id: str, column: list[str]
) -> list[dict[str, Any]]:
    op = next(item for item in expand_routers([entry]) if item["id"] == branch_id)
    return materialize(op["code"], op["fn_name"])((column,))


def test_branch_reads_a_wrapped_list_and_no_rows() -> None:
    # A list-mode source archives its whole list as one item; the same decoder
    # handles that shape, and zero rows yield zero output rows.
    entry = _router()
    column = [json.dumps(_ROWS)]
    (patient,) = [op for op in expand_routers([entry]) if op["id"] == "route.patient"]
    fn = materialize(patient["code"], patient["fn_name"])
    assert [row["case_id"] for row in fn((column,))] == ["a"]
    assert fn(([],)) == []


def test_router_rejects_a_duplicate_branch_a_rule_and_a_default() -> None:
    with pytest.raises(ValueError, match="default branch"):
        RouterOp.from_entry(_router(default_branch="diagnose"))
    with pytest.raises(ValueError, match="exactly one input"):
        RouterOp.from_entry(_router(inputs=[]))
    with pytest.raises(ValueError, match="branches must be unique"):
        RouterOp.from_entry(
            _router(
                rules=[
                    {"branch": "x", "match": "equals", "value": "1"},
                    {"branch": "x", "match": "equals", "value": "2"},
                ]
            )
        )


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
                "code": "def norm(c):\n    return [{'text': t} for t in c[0]]\n",
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
    assert "RouterOp" not in kinds
    assert kinds.count("LambdaOp") == 5
    branch_ids = sorted(
        op_id for op_id in graph if "route_patient" in op_id or "route_test" in op_id
    )
    assert len(branch_ids) == 2
    turn = next(node for node in graph.values() if "turn" in node["_id"])
    assert turn["_inputs"] == [
        next(op_id for op_id in branch_ids if "route_test" in op_id)
    ]


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
    template = _spec(
        [_state("Live_r{{round}}", ["Symbols"])],
        [_state("Live_r{{round}}", ["Live_r{{prev}}"])],
    ).driver.template
    assert template is not None
    assert plan_for_round(template, 1).export == ["Live_r0"]
    assert plan_for_round(template, 2).ops[0]["inputs"] == ["Live_r0"]
    assert stop_export(template, 2) == "Live_r0"
    assert stop_export(template, 3) == "Live_r1"


def test_template_leaves_lambda_code_alone_but_renders_every_other_string() -> None:
    template = _spec(
        [
            {
                "id": "Live_r{{round}}",
                "op": "LambdaOp",
                "mode": "list",
                "inputs": ["Symbols"],
                "fn_name": "echo",
                "code": "# {{round}}\ndef echo(c):\n    return c[0]\n",
            }
        ],
        [_state("Live_r{{round}}", ["Live_r{{prev}}"])],
    ).driver.template
    assert template is not None
    (op,) = plan_for_round(template, 1).ops
    assert op["id"] == "Live_r0"
    assert op["code"] == "# {{round}}\ndef echo(c):\n    return c[0]\n"


def test_template_rejects_an_api_driver() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        DynamicSpec.model_validate(
            {
                "type": "dynamic",
                "goal": "g",
                "driver": {
                    "model": "m",
                    "template": {
                        "first": {"ops": [_state("Live_r{{round}}", ["Symbols"])]},
                        "next": {"ops": [_state("Live_r{{round}}", ["Symbols"])]},
                        "stop_when_empty": "Live_r{{prev}}",
                    },
                    "api": {"url": "http://planner.test"},
                },
            }
        )


@pytest.mark.parametrize(
    ("first", "next_ops", "message"),
    [
        (
            [_state("Live_r{{round}}", ["Symbols"], mode="row")],
            [_state("Live_r{{round}}", ["Symbols"])],
            "list-mode",
        ),
        (
            [_state("Live_r{{round}}", ["Symbols"])],
            [_state("Live_r{{round}}", ["Symbols"], mode="row")],
            "list-mode",
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
    assert archived_element_count([{"output": "one"}]) == 1
