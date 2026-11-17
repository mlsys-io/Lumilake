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


def _driver(template: dict, **extra: object) -> dict:
    return {
        "type": "dynamic",
        "goal": "g",
        "driver": {"model": "m", "template": template, **extra},
    }


def test_template_renders_the_round_suffix_and_the_stop_export() -> None:
    # Every string a body carries renders its placeholders, except a Lambda's
    # code, which is Python source, and single braces, which are a row template.
    turn = {
        "id": "Turn_r{{round}}",
        "op": "LLMChatOp",
        "inputs": ["Live_r{{round}}"],
        "messages": [{"role": "user", "content": "Live_r{{prev}}"}],
        "config": {"model": "m"},
        "rowwise_template": "{row}",
        "rowwise_columns": [{"label": "row", "node": "Live_r{{round}}", "path": "out"}],
    }
    template = DynamicSpec.model_validate(
        _driver(
            {
                "first": {"ops": [_LIST], "export": ["Live_r{{round}}"]},
                "next": {
                    "ops": [
                        dict(_LIST, inputs=["Live_r{{prev}}"]),
                        dict(_LIST, id="Echo_r{{round}}", code="# {{round}}\n"),
                        turn,
                    ],
                    "export": ["Live_r{{round}}"],
                },
                "stop_when_empty": "Live_r{{prev}}",
            }
        )
    ).driver.template
    assert template is not None
    # Round 2 on renders ``next``, whose previous round is round 1's.
    live, echo, rendered = template_plan_for_round(template, 2).ops
    assert (live["id"], live["inputs"]) == ("Live_r1", ["Live_r0"])
    assert echo["code"] == "# {{round}}\n"
    assert rendered["messages"][0]["content"] == "Live_r0"
    assert rendered["rowwise_columns"][0]["node"] == "Live_r1"
    assert rendered["rowwise_template"] == "{row}"
    assert template_stop_export(template, 2) == "Live_r0"
    # A template and an API planner are alternatives.
    with pytest.raises(ValidationError):
        DynamicSpec.model_validate(_driver({}, api={"url": "http://p.test"}))


_RETRIEVAL = {
    "id": "Live_r{{round}}",
    "op": "DataRetrievalOp",
    "inputs": ["Symbols"],
    "data_spec": {"type": "lumid", "mode": "sql", "template": "S"},
}
_TALLY = _state("T_r{{round}}", ["Symbols"])
_ROW_NEXT = dict(_LIST, mode="row", inputs=["Live_r{{prev}}"])
_MISSING = [{"ref": "Missing", "id": "Live_r{{round}}", "inputs": []}]
_OTHER = [_state("Other_r{{round}}", ["Symbols"])]
_KEEP = _state("Keep_r{{round}}", ["Live_r{{round}}"])


@pytest.mark.parametrize(
    ("first", "next_ops", "export", "message"),
    [
        # Only a list-mode export archives one item per row, so only it can read
        # as empty at the stop check; no other op stands as the stop export.
        ([dict(_LIST, mode="row")], [_LIST], None, "list-mode"),
        ([_LIST], [_ROW_NEXT], None, "not available"),
        ([_RETRIEVAL], [_LIST], None, "must be a LambdaOp"),
        # Every export names a node its body builds, and a library ref resolves.
        (_OTHER, [_LIST], None, "exported op id"),
        (_MISSING, [_LIST], None, "library"),
        # A next body may read a list-mode Lambda the earlier round ran, which is
        # how a tally carries across rounds; a row-wise one is misread. A node
        # only round 1 built is gone by round 3, which renders ``next`` again.
        ([_LIST, _TALLY], [_LIST, dict(_TALLY, inputs=["T_r{{prev}}"])], None, None),
        (
            [_LIST, _TALLY],
            [_LIST, dict(_TALLY, mode="row", inputs=["T_r{{prev}}"])],
            None,
            "'T_r1'",
        ),
        ([_LIST, _KEEP], [_LIST], ["Live_r{{round}}", "Keep_r{{round}}"], "Keep_r1"),
    ],
    ids=["row-1", "row-2", "retrieval", "no-export", "no-ref", "leaf", "rowleaf", "r3"],
)
def test_template_validator_checks_a_body(
    first: list[dict],
    next_ops: list[dict],
    export: list[str] | None,
    message: str | None,
) -> None:
    body = export or ["Live_r{{round}}"]
    template = DriverTemplate.model_validate(
        {
            "first": {"ops": first, "export": body},
            "next": {"ops": next_ops, "export": body},
            "stop_when_empty": "Live_r{{prev}}",
        }
    )
    library = {"Known": _state("Known", [])}
    if message is None:
        validate_template_spec(template, library, max_nodes_per_round=8)
        return
    with pytest.raises(DriverProtocolError, match=message):
        validate_template_spec(template, library, max_nodes_per_round=8)
