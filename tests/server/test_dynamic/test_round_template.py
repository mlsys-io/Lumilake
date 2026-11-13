"""Tests for fixed-round dynamic templates and their data-driven stop."""

import json

import pytest
from pydantic import ValidationError

from lumilake_server.dynamic.driver import (
    DriverProtocolError,
    archived_export_has_rows,
    forward_refs,
    template_plan_for_round,
    template_stop_export,
    validate_template_spec,
)
from lumilake_server.dynamic.spec import DriverTemplate, DynamicSpec
from lumilake_server.routes.jobs import _decode_archived_items, _extract_leaf_outputs


def _state_op(op_id: str, inputs: list[str], **extra: object) -> dict:
    return {
        "id": op_id,
        "op": "LambdaOp",
        "inputs": inputs,
        "mode": "list",
        "fn_name": "state",
        "code": "def state(columns):\n    return columns[0]\n",
        **extra,
    }


def _body(export_previous: bool = False) -> dict:
    return {
        "ops": [
            _state_op("Input_r{{round}}", ["Symbols"]),
            _state_op(
                "Live_r{{round}}",
                ["Input_r{{prev}}" if export_previous else "Input_r{{round}}"],
            ),
        ],
        "export": ["Live_r{{round}}"],
    }


def _spec(**driver: object) -> dict:
    return {
        "type": "dynamic",
        "goal": "one fixed shape",
        "driver": {"model": "Qwen/Qwen3-8B", **driver},
    }


def _template(**overrides: object) -> dict:
    return {
        "first": _body(),
        "next": _body(export_previous=True),
        "stop_when_empty": "Live_r{{prev}}",
        **overrides,
    }


def test_template_renders_the_round_suffix_and_the_stop_export() -> None:
    # A template and an API planner are alternatives.
    with pytest.raises(ValidationError):
        DynamicSpec.model_validate(
            _spec(template=_template(), api={"url": "http://planner.test"})
        )

    spec = DynamicSpec.model_validate(_spec(template=_template()))
    assert spec.driver.template is not None
    first = template_plan_for_round(spec.driver.template, 1)
    assert [op["id"] for op in first.ops] == ["Input_r0", "Live_r0"]
    assert first.ops[1]["inputs"] == ["Input_r0"]
    twentieth = template_plan_for_round(spec.driver.template, 20)
    assert [op["id"] for op in twentieth.ops] == ["Input_r19", "Live_r19"]
    assert twentieth.ops[1]["inputs"] == ["Input_r18"]
    assert template_stop_export(spec.driver.template, 20) == "Live_r18"


def test_collected_export_counts_rows_and_forwards_them_to_the_next_round() -> None:
    # A collected export holds one archived item per row, whatever the row's
    # value holds, and next round reads each of them as a forwarded input.
    name = "leaf_lambda_round_1_Live_r0_7634"

    def collected(rows: list[dict]) -> list[dict]:
        leaf = _extract_leaf_outputs(
            {"round_1": {name: [json.dumps({"output": row}) for row in rows]}}, [name]
        )
        return _decode_archived_items(leaf[name[len("leaf_") :]])

    live_case = {"case_id": "0", "turn": 1, "completed": False}
    assert archived_export_has_rows(
        collected([{"output": []}]), export_id="Live_r0", op={"op": "LambdaOp"}
    )
    assert not archived_export_has_rows(
        collected([]), export_id="Live_r0", op={"op": "LambdaOp"}
    )
    with pytest.raises(DriverProtocolError, match="not a Lambda result item"):
        archived_export_has_rows(
            [{"value": {"items": [{"case_id": "0"}]}}],
            export_id="Live_r0",
            op={"op": "LambdaOp"},
        )
    ops, inputs = forward_refs(
        [{"id": "CaseState_r1", "op": "LambdaOp", "inputs": ["Live_r0"]}],
        {"Live_r0": collected([live_case])},
        {"Live_r0": {"op": "LambdaOp"}},
    )
    assert ops[0]["inputs"] == ["Live_r0"]
    assert inputs == {"Live_r0": [json.dumps(live_case)]}


@pytest.mark.parametrize(
    ("first", "next_body", "message"),
    [
        # Only a list-mode export archives one item per row, so only it can read
        # as empty at the stop check. ``next`` runs from round 2, so the round-one
        # check sees ``first`` and the later check sees ``next``.
        (_state_op("Live_r{{round}}", ["Symbols"], mode="row"), None, "list-mode"),
        (
            None,
            _state_op("Live_r{{round}}", ["Live_r{{prev}}"], mode="row"),
            "list-mode",
        ),
        # The stop check reads a Lambda's archived rows, so no other op stands as
        # the stop export.
        (
            {
                "id": "Live_r{{round}}",
                "op": "DataRetrievalOp",
                "inputs": ["Symbols"],
                "data_spec": {"type": "lumid", "mode": "sql", "template": "SELECT 1"},
            },
            None,
            "must be a LambdaOp",
        ),
        # Every export and every stop export names a node the body builds.
        (_state_op("Other_r{{round}}", ["Symbols"]), None, "exports unknown ids"),
        # A library ref resolves, and one nothing declares is rejected.
        (
            {"ref": "Missing", "id": "Live_r{{round}}", "inputs": []},
            None,
            "library entry",
        ),
    ],
    ids=[
        "round-one-row",
        "round-two-row",
        "retrieval-export",
        "not-exported",
        "missing-ref",
    ],
)
def test_template_validator_rejects_an_unusable_body(
    first: dict | None, next_body: dict | None, message: str
) -> None:
    list_op = _state_op("Live_r{{round}}", ["Symbols"])
    with pytest.raises(DriverProtocolError, match=message):
        validate_template_spec(
            DriverTemplate.model_validate(
                {
                    "first": {"ops": [first or list_op], "export": ["Live_r{{round}}"]},
                    "next": {
                        "ops": [next_body or list_op],
                        "export": ["Live_r{{round}}"],
                    },
                    "stop_when_empty": "Live_r{{prev}}",
                }
            ),
            {"Known": _state_op("Known", [])},
        )


def test_template_body_renders_the_node_refs_of_a_gated_op() -> None:
    # The node refs a body carries -- a gate's target, a column's node, a
    # message content, a named argument, a retrieval param -- all take the round
    # suffix. A Lambda's code is Python source, so a placeholder in it stays
    # literal.
    body = {
        "ops": [
            _state_op("Source_r{{round}}", ["Symbols"]),
            {
                "id": "Turn_r{{round}}",
                "op": "LLMChatOp",
                "inputs": ["Source_r{{round}}"],
                "messages": [
                    {"role": "user", "content": "Advance {row}"},
                    {"role": "user", "content": "Source_r{{round}}"},
                ],
                "config": {"model": "Qwen/Qwen3-8B"},
                "rowwise_template": "{row}",
                "rowwise_columns": [
                    {
                        "label": "row",
                        "node": "Source_r{{round}}",
                        "path": "items.output",
                    }
                ],
                "row_gate": {"node": "Source_r{{round}}", "branch": "finish"},
            },
            {
                "id": "Lookup_r{{round}}",
                "op": "DataRetrievalOp",
                "inputs": ["Source_r{{round}}"],
                "data_spec": {
                    "type": "s3",
                    "connection_string": "s3://bucket",
                    "template": "docs/{name}.json",
                    "params": [
                        {
                            "label": "name",
                            "node": "Source_r{{round}}",
                            "path": "items.output",
                        }
                    ],
                },
            },
            _state_op("Echo_r{{round}}", ["Symbols"], code="# {{round}}\npass\n"),
        ],
        "export": ["Echo_r{{round}}"],
    }
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "Echo_r{{prev}}"}
    )
    validate_template_spec(template)
    ops = {op["id"]: op for op in template_plan_for_round(template, 2).ops}
    assert ops["Turn_r1"]["row_gate"]["node"] == "Source_r1"
    assert [column["node"] for column in ops["Turn_r1"]["rowwise_columns"]] == [
        "Source_r1"
    ]
    assert [message["content"] for message in ops["Turn_r1"]["messages"]] == [
        "Advance {row}",
        "Source_r1",
    ]
    assert ops["Lookup_r1"]["data_spec"]["params"] == [
        {"label": "name", "node": "Source_r1", "path": "items.output"}
    ]
    assert ops["Echo_r1"]["code"] == "# {{round}}\npass\n"
