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


def _state(op_id: str, inputs: list[str], **extra: object) -> dict:
    return {
        "id": op_id,
        "op": "LambdaOp",
        "inputs": inputs,
        "mode": "list",
        "fn_name": "state",
        "code": "def state(columns):\n    return columns[0]\n",
        **extra,
    }


_LIST = _state("Live_r{{round}}", ["Symbols"])
_TEMPLATE = {
    "first": {"ops": [_LIST], "export": ["Live_r{{round}}"]},
    "next": {
        "ops": [_state("Live_r{{round}}", ["Live_r{{prev}}"])],
        "export": ["Live_r{{round}}"],
    },
    "stop_when_empty": "Live_r{{prev}}",
}


def _validate(first: list[dict], next_ops: list[dict]) -> None:
    export = ["Live_r{{round}}"]
    spec = {
        "first": {"ops": first, "export": export},
        "next": {"ops": next_ops, "export": export},
        "stop_when_empty": "Live_r{{prev}}",
    }
    validate_template_spec(
        DriverTemplate.model_validate(spec),
        {"Known": _state("Known", [])},
        max_nodes_per_round=8,
    )


def _driver(template: dict, **extra: object) -> dict:
    return {
        "type": "dynamic",
        "goal": "g",
        "driver": {"model": "m", "template": template, **extra},
    }


def test_template_renders_the_round_suffix_and_the_stop_export() -> None:
    template = DynamicSpec.model_validate(_driver(_TEMPLATE)).driver.template
    assert template is not None
    # Round 2 on renders ``next``, whose previous round is round 1's.
    (op,) = template_plan_for_round(template, 2).ops
    assert (op["id"], op["inputs"]) == ("Live_r1", ["Live_r0"])
    assert template_stop_export(template, 2) == "Live_r0"
    # A template and an API planner are alternatives.
    with pytest.raises(ValidationError):
        DynamicSpec.model_validate(_driver(_TEMPLATE, api={"url": "http://p.test"}))


_RETRIEVAL = {
    "id": "Live_r{{round}}",
    "op": "DataRetrievalOp",
    "inputs": ["Symbols"],
    "data_spec": {"type": "lumid", "mode": "sql", "template": "S"},
}
_ROW_NEXT = dict(_LIST, mode="row", inputs=["Live_r{{prev}}"])
_MISSING = {"ref": "Missing", "id": "Live_r{{round}}", "inputs": []}
_STATE_ECHO = _state("Echo_r{{round}}", ["Echo_r{{prev}}"])


@pytest.mark.parametrize(
    ("first", "next_ops", "message"),
    [
        # Only a list-mode export archives one item per row, so only it can read
        # as empty at the stop check.
        ([dict(_LIST, mode="row")], [_LIST], "list-mode"),
        ([_LIST], [_ROW_NEXT], "not available"),
        # No other op stands as the stop export.
        ([_RETRIEVAL], [_LIST], "must be a LambdaOp"),
        # Every export names a node its body builds, and a library ref resolves.
        ([_state("Other_r{{round}}", ["Symbols"])], [_LIST], "exported op id"),
        ([_MISSING], [_LIST], "library"),
    ],
    ids=["row-1", "row-2", "retrieval", "no-export", "no-ref"],
)
def test_template_validator_rejects_an_unusable_body(
    first: list[dict], next_ops: list[dict], message: str
) -> None:
    with pytest.raises(DriverProtocolError, match=message):
        _validate(first, next_ops)


def _leaf_read(op: dict) -> DriverTemplate:
    """A template whose next body reads a leaf of the round before it."""
    later = [dict(_LIST, inputs=["Live_r{{prev}}"]), dict(op, inputs=["T_r{{prev}}"])]
    export = {"export": ["Live_r{{round}}"]}
    return DriverTemplate.model_validate(
        {
            "first": {**export, "ops": [_LIST, op]},
            "next": {**export, "ops": later},
            "stop_when_empty": "Live_r{{prev}}",
        }
    )


def test_template_validator_accepts_a_later_read_of_a_list_lambda_leaf() -> None:
    # The route archives every leaf, so a next body may read a list-mode Lambda.
    validate_template_spec(
        _leaf_read(_state("T_r{{round}}", ["Symbols"])), max_nodes_per_round=8
    )


def test_template_validator_refuses_a_later_read_of_a_chat_leaf() -> None:
    # A row-wise chat leaf archives one grouped item per table, so binding it
    # per archived item would misread it.
    chat = {
        "id": "T_r{{round}}",
        "op": "LLMChatOp",
        "inputs": ["Symbols"],
        "messages": [{"role": "user", "content": "x"}],
        "config": {"model": "m"},
        "rowwise_template": "{row}",
        "rowwise_columns": [{"label": "r", "node": "Symbols", "path": "items.output"}],
    }
    with pytest.raises(DriverProtocolError, match="'T_r1'"):
        validate_template_spec(_leaf_read(chat), max_nodes_per_round=8)


def test_template_validator_rejects_a_round_three_read() -> None:
    # Round 2 reads what round 1 exports, so it validates; round 3 reads the
    # same op from round 2, which never produced it.
    template = DriverTemplate.model_validate(
        {
            "first": {
                "ops": [_LIST, _state("Keep_r{{round}}", ["Live_r{{round}}"])],
                "export": ["Live_r{{round}}", "Keep_r{{round}}"],
            },
            "next": {
                "ops": [_state("Live_r{{round}}", ["Keep_r{{prev}}"])],
                "export": ["Live_r{{round}}"],
            },
            "stop_when_empty": "Live_r{{prev}}",
        }
    )
    with pytest.raises(DriverProtocolError, match="unknown node id 'Keep_r1'"):
        validate_template_spec(template, max_nodes_per_round=8)


def test_template_body_renders_every_string_it_carries() -> None:
    # Every string a body carries renders its placeholders, except a Lambda's
    # code, which is Python source, and single braces, which are a row template.
    body = {
        "ops": [
            _state("Source_r{{round}}", ["Symbols"]),
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
            _state("Echo_r{{round}}", ["Symbols"], code="# {{round}}\npass\n"),
        ],
        "export": ["Echo_r{{round}}"],
    }
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "Echo_r{{prev}}"}
    )
    turn = {o["id"]: o for o in template_plan_for_round(template, 2).ops}["Turn_r1"]
    assert turn["messages"][0]["content"] == "Source_r0"
    assert turn["rowwise_columns"][0]["node"] == "Source_r1"
    assert turn["rowwise_template"] == "{row}"
