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


def _body(prev: bool) -> dict:
    """One round body; ``prev`` makes it read the previous round's export."""
    return {
        "ops": [
            _state_op("Input_r{{round}}", ["Symbols"]),
            _state_op(
                "Live_r{{round}}", ["Input_r{{prev}}" if prev else "Input_r{{round}}"]
            ),
        ],
        "export": ["Live_r{{round}}"],
    }


def _template() -> dict:
    return {
        "first": _body(prev=False),
        "next": _body(prev=True),
        "stop_when_empty": "Live_r{{prev}}",
    }


def test_template_renders_the_round_suffix_and_the_stop_export() -> None:
    # A template and an API planner are alternatives.
    driver: dict = {"model": "Qwen/Qwen3-8B", "template": _template()}
    spec = {"type": "dynamic", "goal": "one fixed shape", "driver": driver}
    with pytest.raises(ValidationError):
        DynamicSpec.model_validate(
            {**spec, "driver": {**driver, "api": {"url": "http://p.test"}}}
        )

    template = DynamicSpec.model_validate(spec).driver.template
    assert template is not None
    first = template_plan_for_round(template, 1)
    assert [(op["id"], op["inputs"]) for op in first.ops] == [
        ("Input_r0", ["Symbols"]),
        ("Live_r0", ["Input_r0"]),
    ]
    # Round 2 on renders ``next``, whose previous round is round 1's.
    second = template_plan_for_round(template, 2)
    assert [(op["id"], op["inputs"]) for op in second.ops] == [
        ("Input_r1", ["Symbols"]),
        ("Live_r1", ["Input_r0"]),
    ]
    assert template_stop_export(template, 2) == "Live_r0"


_LIST = _state_op("Live_r{{round}}", ["Symbols"])
_ROW = dict(_LIST, mode="row")


@pytest.mark.parametrize(
    ("first", "next_ops", "message"),
    [
        # Only a list-mode export archives one item per row, so only it can read
        # as empty at the stop check. ``next`` runs from round 2, so the round-one
        # check sees ``first`` and the later check sees ``next``.
        ([_ROW], [_LIST], "list-mode"),
        ([_LIST], [dict(_ROW, inputs=["Live_r{{prev}}"])], "list-mode"),
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
        # export; a non-exported read is refused.
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


def test_template_body_renders_every_string_it_carries() -> None:
    # Every string a body carries renders the placeholders, except a Lambda's
    # code, which is Python source, and single braces, which belong to a row
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
    validate_template_spec(template, max_nodes_per_round=8)
    ops = {op["id"]: op for op in template_plan_for_round(template, 2).ops}
    assert ops["Turn_r1"]["messages"][0]["content"] == "Source_r0"
    assert ops["Turn_r1"]["rowwise_columns"][0]["node"] == "Source_r1"
    # A row template's single braces are not placeholders.
    assert ops["Turn_r1"]["rowwise_template"] == "{row}"
    assert ops["Echo_r1"]["code"] == "# {{round}}\npass\n"
