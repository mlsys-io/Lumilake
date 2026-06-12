import json

import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import (
    _coerce_output_value,
    _walk_output_path,
)
from lumilake_server.runtime.sensitive import (
    REDACTED_TOKEN_PLACEHOLDER,
    redact_sensitive,
)


def test_walk_single_level_dict():
    assert _walk_output_path({"output": "hi"}, ("output",), "node-1") == "hi"


def test_walk_nested_dict():
    item = {"metadata": {"prompt": "hello"}}
    assert _walk_output_path(item, ("metadata", "prompt"), "node-2") == "hello"


def test_walk_into_json_serialized_dataframe():
    # Mirrors the shape from `mode: sql` retrievals where the DataFrame is
    # serialized as a JSON string before transport.
    df_json = json.dumps({"symbol": {"0": "NVDA"}, "close": {"0": 107.5}})
    item = {"table": df_json}
    walked = _walk_output_path(item, ("table", "symbol"), "node-sql")
    assert walked == {"0": "NVDA"}


def test_walk_raises_on_missing_key():
    with pytest.raises(RuntimeError, match=r"missing field at path 'items\.metadata'"):
        _walk_output_path({"output": "x"}, ("metadata",), "node-3")


def test_walk_raises_on_non_dict_descent():
    with pytest.raises(RuntimeError, match=r"missing field at path"):
        _walk_output_path({"output": 42}, ("output", "nested"), "node-4")


def test_walk_raises_on_undecodable_string():
    with pytest.raises(RuntimeError, match="non-JSON string"):
        _walk_output_path({"table": "not-json"}, ("table", "col"), "node-5")


def test_coerce_output_value_passes_strings_through():
    assert _coerce_output_value("hello") == "hello"


def test_coerce_output_value_serializes_containers():
    assert _coerce_output_value({"a": 1}) == json.dumps({"a": 1})
    assert _coerce_output_value([1, 2, 3]) == json.dumps([1, 2, 3])


def test_redact_strips_top_level_lumid_data_token():
    spec = {"type": "lumid", "lumid_data_token": "secret-bearer", "mode": "sql"}
    redacted = redact_sensitive(spec)
    assert redacted["lumid_data_token"] == REDACTED_TOKEN_PLACEHOLDER
    assert redacted["type"] == "lumid"
    assert redacted["mode"] == "sql"
    assert spec["lumid_data_token"] == "secret-bearer"


def test_redact_strips_nested_lumid_cfg_token():
    spec = {
        "type": "list",
        "lumid_cfg": {
            "lumid_data_url": "http://example",
            "lumid_data_token": "another-secret",
            "encoding": "utf-8",
        },
    }
    redacted = redact_sensitive(spec)
    assert redacted["lumid_cfg"]["lumid_data_token"] == REDACTED_TOKEN_PLACEHOLDER
    assert redacted["lumid_cfg"]["lumid_data_url"] == "http://example"


def test_redact_walks_lists_and_full_task_spec_shape():
    task_spec = {
        "spec": {
            "graph": {
                "nodes": [
                    {
                        "name": "n1",
                        "spec": {
                            "data": {
                                "type": "lumid",
                                "lumid_data_token": "tok-1",
                            },
                        },
                    },
                    {
                        "name": "n2",
                        "spec": {
                            "data": {
                                "type": "list",
                                "lumid_cfg": {"lumid_data_token": "tok-2"},
                            },
                        },
                    },
                ],
            },
        },
    }
    redacted = redact_sensitive(task_spec)
    nodes = redacted["spec"]["graph"]["nodes"]
    assert nodes[0]["spec"]["data"]["lumid_data_token"] == REDACTED_TOKEN_PLACEHOLDER
    assert (
        nodes[1]["spec"]["data"]["lumid_cfg"]["lumid_data_token"]
        == REDACTED_TOKEN_PLACEHOLDER
    )
    # Original is left untouched so the live submission keeps the token.
    assert (
        task_spec["spec"]["graph"]["nodes"][0]["spec"]["data"]["lumid_data_token"]
        == "tok-1"
    )


def test_redact_preserves_empty_token_field():
    # Empty/absent tokens are left as-is so we don't fabricate placeholders.
    spec = {"lumid_data_token": ""}
    assert redact_sensitive(spec) == {"lumid_data_token": ""}


def test_coerce_output_value_serializes_primitive_leaves():
    # Regression: a path that descends into a JSON-stringified DataFrame
    # column can yield a primitive leaf; the list[str] contract requires
    # these to be stringified before flowing into chat prompts.
    df_json = json.dumps({"symbol": {"0": 42}})
    walked = _walk_output_path({"table": df_json}, ("table", "symbol", "0"), "node-x")
    assert walked == 42
    coerced = _coerce_output_value(walked)
    assert coerced == "42"
    assert isinstance(coerced, str)
    assert _coerce_output_value(True) == "true"
    assert _coerce_output_value(None) == "null"
    assert _coerce_output_value(1.5) == "1.5"
