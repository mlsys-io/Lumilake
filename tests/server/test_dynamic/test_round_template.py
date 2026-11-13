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
                "LiveState_r{{round}}",
                ["Input_r{{prev}}" if export_previous else "Input_r{{round}}"],
            ),
        ],
        "export": ["LiveState_r{{round}}"],
    }


def _template() -> dict:
    return {
        "first": _body(),
        "next": _body(export_previous=True),
        "stop_when_empty": "LiveState_r{{prev}}",
    }


def _spec(template: dict, **extra: object) -> dict:
    return {
        "type": "dynamic",
        "goal": "one fixed shape",
        "driver": {"model": "Qwen/Qwen3-8B", "template": template, **extra},
    }


@pytest.mark.parametrize(
    ("template", "extra"),
    [
        (_template(), {"api": {"url": "http://planner.test"}}),
        ({**_template(), "unknown": True}, {}),
        ({**_template(), "first": {**_body(), "unknown": True}}, {}),
    ],
    ids=["api-and-template", "outer-key", "nested-key"],
)
def test_template_spec_rejects_bad_shapes(template: dict, extra: dict) -> None:
    with pytest.raises(ValidationError):
        DynamicSpec.model_validate(_spec(template, **extra))


def test_template_renders_the_round_suffix_and_the_stop_export() -> None:
    spec = DynamicSpec.model_validate(_spec(_template()))
    assert spec.driver.template is not None
    first = template_plan_for_round(spec.driver.template, 1)
    assert [op["id"] for op in first.ops] == ["Input_r0", "LiveState_r0"]
    assert first.ops[1]["inputs"] == ["Input_r0"]
    twentieth = template_plan_for_round(spec.driver.template, 20)
    assert [op["id"] for op in twentieth.ops] == ["Input_r19", "LiveState_r19"]
    assert twentieth.ops[1]["inputs"] == ["Input_r18"]
    assert template_stop_export(spec.driver.template, 20) == "LiveState_r18"


def _collected_export(rows: list[dict]) -> list[dict]:
    outputs = {
        "round_1": {
            "leaf_lambda_round_1_LiveState_r0_7634": [
                json.dumps({"output": row}) for row in rows
            ]
        }
    }
    name = "leaf_lambda_round_1_LiveState_r0_7634"
    leaf_outputs = _extract_leaf_outputs(outputs, [name])
    return _decode_archived_items(leaf_outputs[name[len("leaf_") :]])


def test_stop_check_counts_archived_rows() -> None:
    # One row per archived item, whatever the row's value holds.
    assert archived_export_has_rows(
        _collected_export([{"output": []}]),
        export_id="LiveState_r0",
        op={"op": "LambdaOp"},
    )
    assert not archived_export_has_rows(
        _collected_export([]), export_id="LiveState_r0", op={"op": "LambdaOp"}
    )
    with pytest.raises(DriverProtocolError, match="not a Lambda result item"):
        archived_export_has_rows(
            [{"value": {"items": [{"case_id": "0"}]}}],
            export_id="LiveState_r0",
            op={"op": "LambdaOp"},
        )


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
                "data_spec": {
                    "type": "lumid",
                    "mode": "sql",
                    "template": "SELECT 1",
                },
            },
            None,
            "must be a LambdaOp",
        ),
    ],
    ids=["round-one-row", "round-two-row", "retrieval-export"],
)
def test_template_validator_rejects_a_stop_export_that_cannot_stop(
    first: dict | None, next_body: dict | None, message: str
) -> None:
    list_op = _state_op("Live_r{{round}}", ["Symbols"])
    template = DriverTemplate.model_validate(
        {
            "first": {"ops": [first or list_op], "export": ["Live_r{{round}}"]},
            "next": {"ops": [next_body or list_op], "export": ["Live_r{{round}}"]},
            "stop_when_empty": "Live_r{{prev}}",
        }
    )
    with pytest.raises(DriverProtocolError, match=message):
        validate_template_spec(template)


def test_collected_export_forwards_live_case_rows_to_next_round() -> None:
    live_case = {"case_id": "0", "turn": 1, "completed": False}
    ops, inputs = forward_refs(
        [{"id": "CaseState_r1", "op": "LambdaOp", "inputs": ["LiveState_r0"]}],
        {"LiveState_r0": _collected_export([live_case])},
        {"LiveState_r0": {"op": "LambdaOp"}},
    )
    assert ops[0]["inputs"] == ["LiveState_r0"]
    assert inputs == {"LiveState_r0": [json.dumps(live_case)]}


def _library_ref_spec(ref: str) -> DynamicSpec:
    body = {
        "ops": [{"ref": ref, "id": "x_r{{round}}", "inputs": []}],
        "export": ["x_r{{round}}"],
    }
    return DynamicSpec.model_validate(
        {
            "type": "dynamic",
            "goal": "library ref",
            "driver": {
                "model": "Qwen/Qwen3-8B",
                "template": {
                    "first": body,
                    "next": body,
                    "stop_when_empty": "x_r{{prev}}",
                },
            },
            "library": {"Known": _state_op("Known", [])},
        }
    )


def test_template_validator_resolves_a_library_ref() -> None:
    known = _library_ref_spec("Known")
    assert known.driver.template is not None
    validate_template_spec(known.driver.template, known.library)
    unknown = _library_ref_spec("Missing")
    assert unknown.driver.template is not None
    with pytest.raises(DriverProtocolError, match="unknown library entry"):
        validate_template_spec(unknown.driver.template, unknown.library)


def _gated_body() -> dict:
    return {
        "ops": [
            _state_op("Source_r{{round}}", ["Symbols"]),
            {
                "id": "Route_r{{round}}",
                "op": "RouterOp",
                "inputs": ["Source_r{{round}}"],
                "input_path": "items.output",
                "rules": [
                    {"branch": "finish", "match": "contains", "value": "DONE"},
                    {"branch": "pending", "match": "equals", "value": "__never__"},
                ],
                "default_branch": "pending",
            },
            {
                "id": "Turn_r{{round}}",
                "op": "LLMChatOp",
                "inputs": ["Source_r{{round}}", "Route_r{{round}}"],
                "messages": [
                    {"role": "user", "content": "Advance {row}"},
                    {"role": "user", "content": "Route_r{{round}}"},
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
                "row_gate": {
                    "node": "Route_r{{round}}",
                    "branch": "finish",
                    "terminal": True,
                },
            },
            _state_op("Live_r{{round}}", ["Source_r{{round}}"]),
        ],
        "export": ["Live_r{{round}}"],
    }


def test_template_body_renders_the_node_refs_of_a_gated_op() -> None:
    # The router a gate names, the node a column reads, a message content that
    # reads an op, a node ref inside a named argument and one nested in a
    # retrieval param all carry templated ids, so the round suffix renders into
    # each before the body is parsed.
    body = _gated_body()
    body["ops"].extend(
        [
            {
                "id": "Fmt_r{{round}}",
                "op": "FormatOp",
                "inputs": ["Source_r{{round}}"],
                "template": "{doc}",
                "format_kwargs": {"doc": "Source_r{{round}}"},
            },
            {
                "id": "Lookup_r{{round}}",
                "op": "DataRetrievalOp",
                "inputs": ["Fmt_r{{round}}"],
                "data_spec": {
                    "type": "s3",
                    "connection_string": "s3://bucket",
                    "template": "docs/{name}.json",
                    "params": [
                        {
                            "label": "name",
                            "node": "Fmt_r{{round}}",
                            "path": "items.output",
                        }
                    ],
                },
            },
        ]
    )
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "Live_r{{prev}}"}
    )
    validate_template_spec(template)
    plan = template_plan_for_round(template, 2)
    ops = {op["id"]: op for op in plan.ops}
    assert ops["Turn_r1"]["row_gate"]["node"] == "Route_r1"
    assert [column["node"] for column in ops["Turn_r1"]["rowwise_columns"]] == [
        "Source_r1"
    ]
    assert [message["content"] for message in ops["Turn_r1"]["messages"]] == [
        "Advance {row}",
        "Route_r1",
    ]
    assert ops["Fmt_r1"]["format_kwargs"] == {"doc": "Source_r1"}
    assert ops["Lookup_r1"]["data_spec"]["params"] == [
        {"label": "name", "node": "Fmt_r1", "path": "items.output"}
    ]


def test_template_body_leaves_a_lambda_code_untouched() -> None:
    # A Lambda's code is Python source, not a node reference, so a placeholder
    # inside it stays literal even while the op's id renders.
    body = {
        "ops": [_state_op("Echo_r{{round}}", ["Symbols"], code="# {{round}}\npass\n")],
        "export": ["Echo_r{{round}}"],
    }
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "Echo_r{{prev}}"}
    )
    validate_template_spec(template)
    echo = next(
        op for op in template_plan_for_round(template, 2).ops if op["id"] == "Echo_r1"
    )
    assert echo["code"] == "# {{round}}\npass\n"
