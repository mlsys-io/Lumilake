"""Tests for fixed-round dynamic templates and their data-driven stop."""

import json

import pytest
from pydantic import ValidationError

from lumilake_server.dynamic.driver import (
    DriverProtocolError,
    archived_export_has_rows,
    template_plan_for_round,
    template_stop_export,
    validate_template_spec,
)
from lumilake_server.dynamic.spec import DriverSettings, DynamicSpec


def _body(export_previous: bool) -> dict:
    source = "Input_r{{prev}}" if export_previous else "Input_r{{round}}"
    return {
        "ops": [
            {
                "id": "Input_r{{round}}",
                "op": "LambdaOp",
                "inputs": ["Symbols"],
                "mode": "list",
                "fn_name": "passthrough",
                "code": "def passthrough(columns):\n    return columns[0]\n",
            },
            {
                "id": "LiveState_r{{round}}",
                "op": "LambdaOp",
                "inputs": [source],
                "mode": "list",
                "fn_name": "passthrough",
                "code": "def passthrough(columns):\n    return columns[0]\n",
            },
        ],
        "export": ["LiveState_r{{round}}"],
    }


def _spec(template: dict, **extra: object) -> dict:
    return {
        "type": "dynamic",
        "goal": "one fixed shape",
        "driver": {"model": "Qwen/Qwen3-8B", "template": template, **extra},
    }


def test_template_and_api_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        DynamicSpec.model_validate(
            _spec(
                {
                    "first": _body(False),
                    "next": _body(True),
                    "stop_when_empty": "LiveState_r{{prev}}",
                },
                api={"url": "http://planner.test"},
            )
        )


@pytest.mark.parametrize("driver", [{}, {"api": None, "template": None}])
def test_template_and_api_may_both_be_absent(driver: dict) -> None:
    settings = DriverSettings(model="Qwen/Qwen3-8B", **driver)
    assert settings.api is None
    assert settings.template is None


def test_template_rejects_unknown_outer_and_nested_keys() -> None:
    template = {
        "first": _body(False),
        "next": _body(True),
        "stop_when_empty": "LiveState_r{{prev}}",
    }
    with pytest.raises(ValidationError, match="Extra inputs"):
        DynamicSpec.model_validate(_spec({**template, "unknown": True}))
    with pytest.raises(ValidationError, match="Extra inputs"):
        DynamicSpec.model_validate(
            _spec({**template, "first": {**_body(False), "unknown": True}})
        )


def test_template_render_rounds_one_two_and_twenty() -> None:
    spec = DynamicSpec.model_validate(
        _spec(
            {
                "first": _body(False),
                "next": {
                    "ops": [
                        {
                            "id": "CaseState_r{{round}}",
                            "op": "LambdaOp",
                            "inputs": ["LiveState_r{{prev}}", "Symbols"],
                            "mode": "list",
                            "fn_name": "state",
                            "code": "def state(columns):\n    return columns[0]\n",
                        },
                        {
                            "id": "LiveState_r{{round}}",
                            "op": "LambdaOp",
                            "inputs": ["CaseState_r{{round}}"],
                            "mode": "list",
                            "fn_name": "state",
                            "code": "def state(columns):\n    return columns[0]\n",
                        },
                    ],
                    "export": ["LiveState_r{{round}}"],
                },
                "stop_when_empty": "LiveState_r{{prev}}",
            }
        )
    )
    assert spec.driver.template is not None
    first = template_plan_for_round(spec.driver.template, 1)
    assert [op["id"] for op in first.ops] == ["Input_r0", "LiveState_r0"]
    second = template_plan_for_round(spec.driver.template, 2)
    assert [op["id"] for op in second.ops] == ["CaseState_r1", "LiveState_r1"]
    assert second.ops[0]["inputs"] == ["LiveState_r0", "Symbols"]
    twentieth = template_plan_for_round(spec.driver.template, 20)
    assert [op["id"] for op in twentieth.ops] == [
        "CaseState_r19",
        "LiveState_r19",
    ]
    assert twentieth.ops[0]["inputs"][0] == "LiveState_r18"
    assert template_stop_export(spec.driver.template, 20) == "LiveState_r18"


def _archive(rows: list[dict], *, content_wrapper: bool = False) -> list[str]:
    item = json.dumps({"task_type": "python", "value": {"items": rows}, "metrics": {}})
    if content_wrapper:
        return [json.dumps({"content": [item]})]
    return [item]


@pytest.mark.parametrize(
    ("archived", "expected"),
    [
        ([], False),
        (_archive([]), False),
        (_archive([], content_wrapper=True), False),
        (_archive([{"case_id": "0"}]), True),
        (_archive([{"case_id": "0"}], content_wrapper=True), True),
    ],
)
def test_stop_check_empty_and_non_empty_archives(
    archived: list[str], expected: bool
) -> None:
    assert (
        archived_export_has_rows(
            archived, export_id="LiveState_r0", op={"op": "LambdaOp"}
        )
        is expected
    )


def test_stop_check_rejects_non_empty_undecodable_value() -> None:
    with pytest.raises(DriverProtocolError, match="not valid JSON"):
        archived_export_has_rows(
            ["this is not an archived JSON item"],
            export_id="LiveState_r0",
            op={"op": "LambdaOp"},
        )


def test_template_validates_ref_only_bodies_with_overridden_inputs() -> None:
    library = {
        "LiveState": {
            "op": "LambdaOp",
            "inputs": ["Symbols"],
            "mode": "list",
            "fn_name": "state",
            "code": "def state(columns):\n    return columns[0]\n",
        },
        "CaseState": {
            "op": "LambdaOp",
            "inputs": ["Symbols"],
            "mode": "list",
            "fn_name": "state",
            "code": "def state(columns):\n    return columns[0]\n",
        },
    }
    template = DynamicSpec.model_validate(
        {
            "type": "dynamic",
            "goal": "ref-only templates",
            "driver": {
                "model": "Qwen/Qwen3-8B",
                "template": {
                    "first": {
                        "ops": [
                            {
                                "ref": "LiveState",
                                "id": "LiveState_r{{round}}",
                                "inputs": ["Symbols"],
                            }
                        ],
                        "export": ["LiveState_r{{round}}"],
                    },
                    "next": {
                        "ops": [
                            {
                                "ref": "CaseState",
                                "id": "CaseState_r{{round}}",
                                "inputs": ["LiveState_r{{prev}}"],
                            },
                            {
                                "ref": "LiveState",
                                "id": "LiveState_r{{round}}",
                                "inputs": ["CaseState_r{{round}}"],
                            },
                        ],
                        "export": ["LiveState_r{{round}}"],
                    },
                    "stop_when_empty": "LiveState_r{{prev}}",
                },
            },
            "library": library,
        }
    )
    assert template.driver.template is not None
    validate_template_spec(template.driver.template, template.library)


def test_template_validator_rejects_unknown_library_ref() -> None:
    template = DynamicSpec.model_validate(
        {
            "type": "dynamic",
            "goal": "bad ref",
            "driver": {
                "model": "Qwen/Qwen3-8B",
                "template": {
                    "first": {
                        "ops": [{"ref": "Missing", "id": "x_r{{round}}", "inputs": []}],
                        "export": ["x_r{{round}}"],
                    },
                    "next": {
                        "ops": [{"ref": "Missing", "id": "x_r{{round}}", "inputs": []}],
                        "export": ["x_r{{round}}"],
                    },
                    "stop_when_empty": "x_r{{prev}}",
                },
            },
            "library": {
                "Known": {
                    "op": "LambdaOp",
                    "inputs": [],
                    "mode": "list",
                    "fn_name": "state",
                    "code": "def state(columns):\n    return columns[0]\n",
                }
            },
        }
    )
    assert template.driver.template is not None
    with pytest.raises(DriverProtocolError, match="unknown library entry"):
        validate_template_spec(template.driver.template, template.library)


def test_template_validator_forwards_a_prior_round_leaf() -> None:
    library = {
        "Source": {
            "op": "LambdaOp",
            "inputs": ["Symbols"],
            "mode": "list",
            "fn_name": "state",
            "code": "def state(columns):\n    return columns[0]\n",
        },
        "Tally": {
            "op": "LambdaOp",
            "inputs": ["Symbols"],
            "mode": "list",
            "fn_name": "state",
            "code": "def state(columns):\n    return columns[0]\n",
        },
    }
    template = DynamicSpec.model_validate(
        {
            "type": "dynamic",
            "goal": "forward a leaf",
            "driver": {
                "model": "Qwen/Qwen3-8B",
                "template": {
                    "first": {
                        "ops": [
                            {
                                "ref": "Source",
                                "id": "State_r{{round}}",
                                "inputs": ["Symbols"],
                            },
                            {
                                "ref": "Tally",
                                "id": "Tally_r{{round}}",
                                "inputs": ["State_r{{round}}"],
                            },
                        ],
                        "export": ["State_r{{round}}"],
                    },
                    "next": {
                        "ops": [
                            {
                                "ref": "Tally",
                                "id": "Tally_r{{round}}",
                                "inputs": ["Tally_r{{prev}}"],
                            }
                        ],
                        "export": ["Tally_r{{round}}"],
                    },
                    "stop_when_empty": "State_r{{prev}}",
                },
            },
            "library": library,
        }
    )
    assert template.driver.template is not None
    validate_template_spec(template.driver.template, template.library)
