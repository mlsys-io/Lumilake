"""Tests for fixed-round dynamic templates and their data-driven stop."""

import json
from typing import Any

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
from lumilake_server.dynamic.spec import DriverSettings, DriverTemplate, DynamicSpec
from lumilake_server.routes.jobs import _decode_archived_items, _extract_leaf_outputs


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


def _collected_export(rows: list[dict]) -> list[Any]:
    outputs = {
        "round_1": {
            "leaf_lambda_round_1_LiveState_r0_7634": [
                json.dumps({"output": row}) for row in rows
            ]
        }
    }
    names = ["leaf_lambda_round_1_LiveState_r0_7634"]
    leaf_outputs = _extract_leaf_outputs(outputs, names)
    values = leaf_outputs[names[0][len("leaf_") :]]
    return _decode_archived_items(values)


_LIVE_CASE = {
    "case_id": "0",
    "turn": 1,
    "history": ["The draft still needs a second pass."],
    "task": "Revise the draft",
    "label": "example",
    "completed": False,
    "route": "open",
}


@pytest.mark.parametrize(("rows", "expected"), [([], False), ([_LIVE_CASE], True)])
def test_stop_check_uses_collected_all_results_shape(
    rows: list[dict], expected: bool
) -> None:
    archived = _collected_export(rows)
    assert archived == [{"output": row} for row in rows]
    assert (
        archived_export_has_rows(
            archived, export_id="LiveState_r0", op={"op": "LambdaOp"}
        )
        is expected
    )


def test_stop_check_counts_each_archived_item_as_one_row() -> None:
    assert archived_export_has_rows(
        [{"output": []}, {"output": ["kept"]}],
        export_id="LiveState_r0",
        op={"op": "LambdaOp"},
    )
    assert archived_export_has_rows(
        [{"output": []}],
        export_id="LiveState_r0",
        op={"op": "LambdaOp"},
    )


def test_stop_check_rejects_unexpected_value() -> None:
    with pytest.raises(DriverProtocolError, match="not a Lambda result item"):
        archived_export_has_rows(
            [{"value": {"items": [{"case_id": "0"}]}}],
            export_id="LiveState_r0",
            op={"op": "LambdaOp"},
        )


@pytest.mark.parametrize("mode", ["row", "aligned"])
def test_template_validator_rejects_a_stop_export_that_cannot_read_empty(
    mode: str,
) -> None:
    # The stop check counts one row per archived item, so only a list-mode
    # export can read as empty. A row- or aligned-mode export archives one item
    # per input row, which would keep the loop running to ``max_rounds``
    # instead of stopping, so the spec is refused at submission.
    body = {
        "ops": [
            {
                "id": "Live_r{{round}}",
                "op": "LambdaOp",
                "inputs": ["Symbols"],
                "mode": mode,
                "fn_name": "live",
                "code": "def live(columns):\n    return columns[0]\n",
            }
        ],
        "export": ["Live_r{{round}}"],
    }
    template = DriverTemplate.model_validate(
        {
            "first": body,
            "next": body,
            "stop_when_empty": "Live_r{{prev}}",
        }
    )
    with pytest.raises(DriverProtocolError, match="must be a list-mode LambdaOp"):
        validate_template_spec(template)


def test_template_validator_rejects_a_non_lambda_stop_export() -> None:
    body = {
        "ops": [
            {
                "id": "Live_r{{round}}",
                "op": "DataRetrievalOp",
                "inputs": ["Symbols"],
                "data_spec": {
                    "type": "lumid",
                    "mode": "sql",
                    "template": "SELECT 1",
                },
            }
        ],
        "export": ["Live_r{{round}}"],
    }
    template = DriverTemplate.model_validate(
        {
            "first": body,
            "next": body,
            "stop_when_empty": "Live_r{{prev}}",
        }
    )
    with pytest.raises(DriverProtocolError, match="must be a LambdaOp"):
        validate_template_spec(template)


def test_collected_export_forwards_live_case_rows_to_next_round() -> None:
    collected = _collected_export([_LIVE_CASE])
    ops, inputs = forward_refs(
        [{"id": "CaseState_r1", "op": "LambdaOp", "inputs": ["LiveState_r0"]}],
        {"LiveState_r0": collected},
        {"LiveState_r0": {"op": "LambdaOp"}},
    )
    assert ops[0]["inputs"] == ["LiveState_r0"]
    assert inputs == {"LiveState_r0": [json.dumps(_LIVE_CASE)]}


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


def _gated_body() -> dict:
    return {
        "ops": [
            {
                "id": "Source_r{{round}}",
                "op": "LambdaOp",
                "inputs": ["Symbols"],
                "mode": "list",
                "fn_name": "states",
                "code": "def states(columns):\n    return columns[0]\n",
            },
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
                "messages": [{"role": "user", "content": "Advance {row}"}],
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
            {
                "id": "Live_r{{round}}",
                "op": "LambdaOp",
                "inputs": ["Source_r{{round}}"],
                "mode": "list",
                "fn_name": "live",
                "code": (
                    "def live(columns):\n"
                    '    return [s for s in columns[0] if "DONE" not in s]\n'
                ),
            },
        ],
        "export": ["Live_r{{round}}"],
    }


def test_template_body_renders_the_node_refs_of_a_gated_op() -> None:
    # A gated body names the router it gates on and the node it takes a column
    # from by their templated ids, so those references render with the round
    # suffix and the rendered round validates.
    template = DriverTemplate.model_validate(
        {
            "first": _gated_body(),
            "next": _gated_body(),
            "stop_when_empty": "Live_r{{prev}}",
        }
    )
    validate_template_spec(template)
    plan = template_plan_for_round(template, 2)
    turn = next(op for op in plan.ops if op["id"] == "Turn_r1")
    assert turn["row_gate"]["node"] == "Route_r1"
    assert [column["node"] for column in turn["rowwise_columns"]] == ["Source_r1"]


def _gated_reader_body() -> dict:
    body = _gated_body()
    body["ops"].append(
        {
            "id": "Review_r{{round}}",
            "op": "LLMChatOp",
            "inputs": ["Turn_r{{round}}"],
            "messages": [{"role": "user", "content": "Turn_r{{round}}"}],
            "config": {"model": "Qwen/Qwen3-8B"},
        }
    )
    return body


def test_template_body_renders_message_content_node_refs() -> None:
    # A user message whose content is an op id reads that op's output, so the
    # content has to render with the round suffix before the body is parsed;
    # unrendered, the reader's declared input names no node its config uses.
    template = DriverTemplate.model_validate(
        {
            "first": _gated_reader_body(),
            "next": _gated_reader_body(),
            "stop_when_empty": "Live_r{{prev}}",
        }
    )
    validate_template_spec(template)
    plan = template_plan_for_round(template, 2)
    review = next(op for op in plan.ops if op["id"] == "Review_r1")
    assert [message["content"] for message in review["messages"]] == ["Turn_r1"]


def _list_source_op() -> dict:
    return {
        "id": "Src_r{{round}}",
        "op": "LambdaOp",
        "inputs": ["Symbols"],
        "mode": "list",
        "fn_name": "states",
        "code": "def states(columns):\n    return columns[0]\n",
    }


def _sink_op(source: str) -> dict:
    return {
        "id": "Sink_r{{round}}",
        "op": "LambdaOp",
        "inputs": [source],
        "mode": "row",
        "fn_name": "sink",
        "code": "def sink(columns):\n    return columns[0]\n",
    }


def _render_round_two(*extra_ops: dict) -> list[dict]:
    body = {"ops": [_list_source_op(), *extra_ops], "export": ["Src_r{{round}}"]}
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "Src_r{{prev}}"}
    )
    validate_template_spec(template)
    return template_plan_for_round(template, 2).ops


def _op_named(ops: list[dict], op_id: str) -> dict:
    return next(op for op in ops if op["id"] == op_id)


def test_template_body_renders_format_op_kwarg_node_refs() -> None:
    # A FormatOp's named arguments are node references the parser resolves, so
    # the round suffix has to render into them before the body is parsed.
    ops = _render_round_two(
        {
            "id": "Fmt_r{{round}}",
            "op": "FormatOp",
            "inputs": ["Src_r{{round}}"],
            "template": "{doc}",
            "format_kwargs": {"doc": "Src_r{{round}}"},
        },
        _sink_op("Fmt_r{{round}}"),
    )
    assert _op_named(ops, "Fmt_r1")["format_kwargs"] == {"doc": "Src_r1"}


def test_template_body_renders_format_op_arg_node_refs() -> None:
    ops = _render_round_two(
        {
            "id": "Fmt_r{{round}}",
            "op": "FormatOp",
            "inputs": ["Src_r{{round}}"],
            "template": "{}",
            "format_args": ["Src_r{{round}}"],
        },
        _sink_op("Fmt_r{{round}}"),
    )
    assert _op_named(ops, "Fmt_r1")["format_args"] == ["Src_r1"]


def test_template_body_renders_dict_prompt_kwarg_node_refs() -> None:
    # A dict prompt names its arguments by node id; unrendered, the body is
    # rejected with "references unknown id" instead of running.
    ops = _render_round_two(
        {
            "id": "Ask_r{{round}}",
            "op": "LLMChatOp",
            "inputs": ["Src_r{{round}}"],
            "prompt": {"template": "{row}", "format_kwargs": {"row": "Src_r{{round}}"}},
            "messages": [{"role": "user", "content": "advance"}],
            "config": {"model": "Qwen/Qwen3-8B"},
        }
    )
    ask = _op_named(ops, "Ask_r1")
    assert ask["prompt"]["format_kwargs"] == {"row": "Src_r1"}


def test_template_body_renders_retrieval_param_node_refs() -> None:
    ops = _render_round_two(
        {
            "id": "Lookup_r{{round}}",
            "op": "DataRetrievalOp",
            "inputs": ["Src_r{{round}}"],
            "data_spec": {
                "type": "s3",
                "connection_string": "s3://bucket",
                "template": "docs/{name}.json",
                "params": [
                    {
                        "label": "name",
                        "node": "Src_r{{round}}",
                        "path": "items.output",
                    }
                ],
            },
        }
    )
    lookup = _op_named(ops, "Lookup_r1")
    params = lookup["data_spec"]["params"]
    assert [param["node"] for param in params] == ["Src_r1"]


def test_template_body_renders_messages_ref_node_refs() -> None:
    ops = _render_round_two(
        {
            "id": "Msg_r{{round}}",
            "op": "MessageOp",
            "inputs": [],
            "messages": [{"role": "user", "content": "advance"}],
        },
        {
            "id": "Ask_r{{round}}",
            "op": "LLMChatOp",
            "inputs": ["Msg_r{{round}}"],
            "messages_ref": "Msg_r{{round}}",
            "config": {"model": "Qwen/Qwen3-8B"},
        },
    )
    assert _op_named(ops, "Ask_r1")["messages_ref"] == "Msg_r1"


def test_template_body_leaves_a_lambda_code_untouched() -> None:
    # A Lambda's code is Python source, not a node reference, so a placeholder
    # inside it stays literal even while the op's id renders.
    body = {
        "ops": [
            {
                "id": "Echo_r{{round}}",
                "op": "LambdaOp",
                "inputs": ["Symbols"],
                "mode": "list",
                "fn_name": "echo",
                "code": "# {{round}}\ndef echo(columns):\n    return columns[0]\n",
            }
        ],
        "export": ["Echo_r{{round}}"],
    }
    template = DriverTemplate.model_validate(
        {"first": body, "next": body, "stop_when_empty": "Echo_r{{prev}}"}
    )
    validate_template_spec(template)
    ops = template_plan_for_round(template, 2).ops
    echo = _op_named(ops, "Echo_r1")
    assert "# {{round}}" in echo["code"]
