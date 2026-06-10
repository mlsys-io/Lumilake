import json

import pytest

from lumilake_server.runtime.runtime_manager.flowmesh import _walk_output_path


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
