"""Tests for fixed-round dynamic templates and their data-driven stop."""

import pytest
from pydantic import ValidationError

from lumilake_server.dynamic.driver import (
    DriverProtocolError,
    template_plan_for_round,
    template_stop_export,
    validate_template_spec,
)
from lumilake_server.dynamic.spec import DriverTemplate, DynamicSpec


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


_TEMPLATE = {
    "first": {
        "ops": [_state_op("Live_r{{round}}", ["Symbols"])],
        "export": ["Live_r{{round}}"],
    },
    "next": {
        "ops": [_state_op("Live_r{{round}}", ["Live_r{{prev}}"])],
        "export": ["Live_r{{round}}"],
    },
    "stop_when_empty": "Live_r{{prev}}",
}


def test_template_renders_the_round_suffix_and_the_stop_export() -> None:
    template = DynamicSpec.model_validate(
        {
            "type": "dynamic",
            "goal": "g",
            "driver": {"model": "m", "template": _TEMPLATE},
        }
    ).driver.template
    assert template is not None
    # Round 2 on renders ``next``, whose previous round is round 1's.
    assert [
        (op["id"], op["inputs"]) for op in template_plan_for_round(template, 2).ops
    ] == [("Live_r1", ["Live_r0"])]
    assert template_stop_export(template, 2) == "Live_r0"
    # A template and an API planner are alternatives.
    with pytest.raises(ValidationError):
        DynamicSpec.model_validate(
            {
                "type": "dynamic",
                "goal": "g",
                "driver": {
                    "model": "m",
                    "template": _TEMPLATE,
                    "api": {"url": "http://planner.test"},
                },
            }
        )


_LIST = _state_op("Live_r{{round}}", ["Symbols"])


@pytest.mark.parametrize(
    ("first", "next_ops", "message"),
    [
        # Only a list-mode export archives one item per row, so only it can read
        # as empty at the stop check; ``next`` runs from round 2 on. Round 3
        # renders ``next`` again, so a row-mode ``next`` export is refused at
        # that read first.
        ([dict(_LIST, mode="row")], [_LIST], "list-mode"),
        (
            [_LIST],
            [dict(_LIST, mode="row", inputs=["Live_r{{prev}}"])],
            "not available",
        ),
        # The stop check reads a Lambda's archived rows, so no other op stands as
        # the stop export.
        (
            [
                {
                    "id": "Live_r{{round}}",
                    "op": "DataRetrievalOp",
                    "inputs": ["Symbols"],
                    "data_spec": {"type": "lumid", "mode": "sql", "template": "S"},
                }
            ],
            [_LIST],
            "must be a LambdaOp",
        ),
        # Every export names a node its body builds, and a library ref resolves.
        ([_state_op("Other_r{{round}}", ["Symbols"])], [_LIST], "exported op id"),
        (
            [{"ref": "Missing", "id": "Live_r{{round}}", "inputs": []}],
            [_LIST],
            "library",
        ),
        # A ``next`` body reads an earlier round only through a list-mode Lambda
        # export.
        (
            [_state_op("Echo_r{{round}}", ["Symbols"]), _LIST],
            [
                _state_op("Echo_r{{round}}", ["Echo_r{{prev}}"]),
                dict(_LIST, inputs=["Echo_r{{prev}}"]),
            ],
            "result is not available",
        ),
    ],
    ids=[
        "round-one-row",
        "round-two-row",
        "retrieval-export",
        "not-exported",
        "missing-ref",
        "reads-a-nonexport",
    ],
)
def test_template_validator_rejects_an_unusable_body(
    first: list[dict], next_ops: list[dict], message: str
) -> None:
    with pytest.raises(DriverProtocolError, match=message):
        validate_template_spec(
            DriverTemplate.model_validate(
                {
                    "first": {"ops": first, "export": ["Live_r{{round}}"]},
                    "next": {"ops": next_ops, "export": ["Live_r{{round}}"]},
                    "stop_when_empty": "Live_r{{prev}}",
                }
            ),
            {"Known": _state_op("Known", [])},
            max_nodes_per_round=8,
        )


def test_template_validator_rejects_a_round_three_read() -> None:
    # Round 2 reads an op round 1 exports, so it validates; round 3 reads the
    # same op from round 2, which never produced it.
    template = DriverTemplate.model_validate(
        {
            "first": {
                "ops": [
                    _state_op("Live_r{{round}}", ["Symbols"]),
                    _state_op("Keep_r{{round}}", ["Live_r{{round}}"]),
                ],
                "export": ["Live_r{{round}}", "Keep_r{{round}}"],
            },
            "next": {
                "ops": [_state_op("Live_r{{round}}", ["Keep_r{{prev}}"])],
                "export": ["Live_r{{round}}"],
            },
            "stop_when_empty": "Live_r{{prev}}",
        }
    )
    with pytest.raises(DriverProtocolError, match="unknown node id 'Keep_r1'"):
        validate_template_spec(template, max_nodes_per_round=8)


def test_template_body_renders_every_string_it_carries() -> None:
    # Every string a body carries renders its placeholders, except a Lambda's
    # code, which is Python source, and single braces, which are part of a row
    # template.
    body = {
        "ops": [
            _state_op("Source_r{{round}}", ["Symbols"]),
            {
                "id": "Turn_r{{round}}",
                "op": "LLMChatOp",
                "inputs": ["Source_r{{round}}"],
                "messages": [{"role": "user", "content": "Source_r{{prev}}"}],
                "config": {"model": "Qwen/Qwen3-8B"},
                "rowwise_template": "{row}",
                "rowwise_columns": [
                    {"label": "row", "node": "Source_r{{round}}", "path": "out"}
                ],
            },
            _state_op("Echo_r{{round}}", ["Symbols"], code="# {{round}}\npass\n"),
        ],
        "export": ["Echo_r{{round}}"],
    }
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "Echo_r{{prev}}"}
    )
    turn = {op["id"]: op for op in template_plan_for_round(template, 2).ops}["Turn_r1"]
    assert turn["messages"][0]["content"] == "Source_r0"
    assert turn["rowwise_columns"][0]["node"] == "Source_r1"
    assert turn["rowwise_template"] == "{row}"
