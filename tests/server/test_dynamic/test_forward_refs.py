"""Unit tests for cross-round reference forwarding in dynamic rounds."""

import copy
import json
from typing import Any

import pytest

from lumilake_server.dynamic.blocks import INPUT_NODE_ID
from lumilake_server.dynamic.driver import (
    DriverProtocolError,
    build_round,
    compute_observation,
    forward_refs,
    resolve_subgraph,
    validate_emitted_subgraph,
    walk_archived_items,
)
from lumilake_server.graphs import Graph
from lumilake_server.parser.common import make_id
from lumilake_server.parser.yaml_parser import _op_id_prefix, parse_yaml_payload
from lumilake_server.runtime.runtime_graph import (
    RuntimeGraphBuilder,
)


def _internal_id(scope: str, op_type: str, user_id: str) -> str:
    return make_id(scope, _op_id_prefix(op_type), user_id)


def _llm_config(op_id: str, **overrides: Any) -> dict:
    config = {
        "id": op_id,
        "op": "LLMChatOp",
        "inputs": [],
        "prompt": {"template": "x", "format_kwargs": {}},
    }
    config.update(overrides)
    return config


def _sql_config(op_id: str) -> dict:
    return {
        "id": op_id,
        "op": "DataRetrievalOp",
        "inputs": [],
        "data_spec": {"type": "lumid", "mode": "sql", "template": "SELECT *"},
    }


def _chat_consumer(*columns: tuple[str, str, str], op_id: str = "q4") -> dict:
    return {
        "id": op_id,
        "op": "LLMChatOp",
        "inputs": [],
        "prompt": {"template": "x", "format_kwargs": {}},
        "config": {"model": "Qwen/Qwen3-8B"},
        "messages": [{"role": "user", "content": "x"}],
        "aggregate_table": [
            {"label": label, "node": node, "path": path}
            for label, node, path in columns
        ],
    }


def test_forward_implicit_llm_reference_binds_default_path():
    # A round-1 LLM op archived its whole item; round 2 references it through
    # ``inputs``. The bound input must equal walking the stored item with the
    # runtime's default path (``items.output``).
    llm_id = "s0"
    results = {llm_id: [{"output": "hello", "extra": 1}]}
    configs = {llm_id: _llm_config(llm_id)}
    subgraph = [
        {
            "id": "q1",
            "op": "FormatOp",
            "inputs": [llm_id],
            "template": "{ref0}",
            "format_kwargs": {"ref0": llm_id},
        }
    ]
    ops, inputs = forward_refs(subgraph, results, configs)
    # The implicit reference is left unchanged; the input is named after the
    # producer op id.
    assert ops[0]["inputs"] == [llm_id]
    assert ops[0]["format_kwargs"] == {"ref0": llm_id}
    assert inputs == {llm_id: ["hello"]}


def test_forward_explicit_sql_column_reference_binds_and_rewrites():
    # A round-1 SQL op archived its whole item; round 2 references a column
    # through ``data_spec.params`` with an explicit path. The reference is
    # rewritten to a per-(node, path) input with ``path: ""``.
    sql_id = "s1"
    results = {sql_id: [{"table": json.dumps({"symbol": {"0": "NVDA"}})}]}
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [
        {
            "id": "q2",
            "op": "DataRetrievalOp",
            "inputs": [],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "template": "SELECT * FROM t WHERE s = {p}",
                "params": [
                    {"label": "p", "node": sql_id, "path": "items.table.symbol"}
                ],
            },
        }
    ]
    ops, inputs = forward_refs(subgraph, results, configs)
    param = ops[0]["data_spec"]["params"][0]
    input_id = f"{sql_id} | items.table.symbol"
    assert param["node"] == input_id
    assert param["path"] == ""
    # Walking items.table.symbol yields the transposed column value.
    assert inputs[input_id] == [json.dumps({"0": "NVDA"})]


def test_forward_implicit_sql_reference_uses_retrieval_path():
    # An implicit ``inputs`` reference to a SQL op reads ``items.table``.
    sql_id = "s1"
    results = {sql_id: [{"table": json.dumps({"symbol": {"0": "NVDA"}})}]}
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [
        {
            "id": "q3",
            "op": "FormatOp",
            "inputs": [sql_id],
            "template": "{ref0}",
            "format_kwargs": {"ref0": sql_id},
        }
    ]
    ops, inputs = forward_refs(subgraph, results, configs)
    assert inputs[sql_id] == [json.dumps({"symbol": {"0": "NVDA"}})]


def test_forward_aggregate_table_column_reference():
    # ``aggregate_table`` columns with a ``node`` + ``path`` are explicit
    # references and are rewritten like params.
    sql_id = "s1"
    results = {sql_id: [{"table": json.dumps({"close": {"0": 107.5}})}]}
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [_chat_consumer(("close", sql_id, "items.table.close"))]
    ops, inputs = forward_refs(subgraph, results, configs)
    row = ops[0]["aggregate_table"][0]
    input_id = f"{sql_id} | items.table.close"
    assert row["node"] == input_id
    assert row["path"] == ""
    assert inputs[input_id] == [json.dumps({"0": 107.5})]


def test_forward_leaves_unknown_references_untouched():
    # A reference to an op not in ``results`` (e.g. another op in the same
    # round) is left as-is; only earlier-round references are forwarded.
    subgraph = [
        {
            "id": "a",
            "op": "DataOp",
            "inputs": [],
            "data": ["x"],
        },
        {
            "id": "b",
            "op": "FormatOp",
            "inputs": ["a"],
            "template": "{ref0}",
            "format_kwargs": {"ref0": "a"},
        },
    ]
    ops, inputs = forward_refs(subgraph, {}, {})
    assert ops == subgraph
    assert inputs == {}


def test_forward_does_not_mutate_input_subgraph():
    # forward_refs must deep-copy each op; rewriting an aggregate column or
    # param must not mutate the caller's plan / node_registry dicts.
    sql_id = "s1"
    results = {sql_id: [{"table": json.dumps({"close": {"0": 107.5}})}]}
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [_chat_consumer(("close", sql_id, "items.table.close"))]
    original = copy.deepcopy(subgraph)
    forward_refs(subgraph, results, configs)
    assert subgraph == original


def _referencing_op() -> dict[str, Any]:
    """A chat op that reads an earlier-round op's column."""
    op = _chat_consumer(("close", "s0", "items.table.close"), op_id="q1")
    op["inputs"] = ["s0"]
    return op


@pytest.mark.parametrize(
    ("results", "export", "message"),
    [
        # A reference to an earlier op whose result is not available (neither a
        # leaf nor exported) is rejected, never recomputed; ``export`` must name
        # ops of the same subgraph.
        ({}, [], "s0"),
        ({"s0": [{"table": "x"}]}, ["nope"], "exported op id 'nope'"),
    ],
    ids=["unavailable", "unknown-export"],
)
def test_validate_checks_the_earlier_ops_and_the_export(
    results: dict, export: list[str], message: str
) -> None:
    with pytest.raises(DriverProtocolError, match=message):
        validate_emitted_subgraph(
            [_referencing_op()],
            {"s0": _sql_config("s0")},
            max_nodes=4,
            results=results,
            export=export,
        )


def _llm_column_library() -> dict[str, dict[str, Any]]:
    return {
        "ReviewTurn": {
            "op": "LLMChatOp",
            "inputs": ["ReviewPrompt"],
            "messages": [{"role": "user", "content": "review"}],
            "rowwise_template": "{prompt}",
            "rowwise_columns": [
                {
                    "label": label,
                    "node": "ReviewPrompt",
                    "path": f"items.output.{label}",
                }
                for label in ("prompt", "review_system")
            ],
            "column_bindings": [
                {"label": label, "input": 0} for label in ("prompt", "review_system")
            ],
            "config": {"model": "m"},
        }
    }


def _llm_column_subgraph(prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{prefix}_prompt_1",
            "op": "LambdaOp",
            "inputs": ["Symbols"],
            "mode": "list",
            "fn_name": "review_prompt",
            "code": "def review_prompt(items):\n    return [{'prompt': 'x'}]\n",
        },
        {
            "ref": "ReviewTurn",
            "id": f"{prefix}_turn_1",
            "inputs": [f"{prefix}_prompt_1"],
        },
    ]


def test_column_bindings_remap_independent_library_instances() -> None:
    library = _llm_column_library()
    subgraph = _llm_column_subgraph("first") + _llm_column_subgraph("second")

    resolved = resolve_subgraph(subgraph, library)
    validate_emitted_subgraph(subgraph, {}, max_nodes=4, library=library)

    turns = [op for op in resolved if op["op"] == "LLMChatOp"]
    assert [turn["rowwise_columns"][0]["node"] for turn in turns] == [
        "first_prompt_1",
        "second_prompt_1",
    ]
    assert library["ReviewTurn"]["rowwise_columns"][0]["node"] == "ReviewPrompt"


@pytest.mark.parametrize(
    ("bindings", "inputs", "message"),
    [
        # The bindings must fill every column, name every input the round
        # passes, and stay inside the op's columns and inputs.
        (None, ["review_prompt_1", "unused_prompt"], "unused"),
        ([{"label": "prompt", "input": 0}], ["review_prompt_1"], "review_system"),
        ([{"label": "missing", "input": 0}], ["review_prompt_1"], "unknown label"),
        ([{"label": "prompt", "input": -1}], ["review_prompt_1"], "invalid input slot"),
    ],
    ids=["unused-input", "unbound-column", "unknown-label", "bad-slot"],
)
def test_column_bindings_reject_an_inconsistent_round(
    bindings: list[dict[str, Any]] | None, inputs: list[str], message: str
) -> None:
    library = _llm_column_library()
    if bindings is not None:
        library["ReviewTurn"]["column_bindings"] = bindings
    subgraph = _llm_column_subgraph("review")
    subgraph[1]["inputs"] = inputs

    with pytest.raises(DriverProtocolError, match=message):
        resolve_subgraph(subgraph, library)


def test_build_round_round2_graph_has_no_round1_op():
    # Round 1 emits a SQL op that round 2 references. Round 2's graph must
    # contain only round 2's ops (plus the observation/proposer wiring), never
    # the round-1 op — the reference is forwarded as a workflow input instead.
    sql_id = "s1"
    node_registry = {sql_id: _sql_config(sql_id)}
    results = {sql_id: [{"table": json.dumps({"symbol": {"0": "NVDA"}})}]}
    round2_subgraph = [
        {
            "id": "q2",
            "op": "DataRetrievalOp",
            "inputs": [sql_id],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "template": "SELECT * FROM t WHERE s = {p}",
                "params": [
                    {"label": "p", "node": sql_id, "path": "items.table.symbol"}
                ],
            },
        }
    ]
    round_build = build_round(
        round2_subgraph,
        node_registry=node_registry,
        round_index=1,
        goal="g",
        observations=["obs"],
        topology=[sql_id],
        preview_width=100,
        model="Qwen/Qwen3-8B",
        max_tokens=100,
        temperature=0.4,
        results=results,
    )
    graph = round_build.graph
    # The round-1 op must not appear anywhere in the round-2 graph.
    round1_internal = _internal_id("round_1", "DataRetrievalOp", sql_id)
    assert round1_internal not in graph
    # The round-2 op is present and its archived leaf output uses path: items.
    round2_internal = _internal_id("round_1", "DataRetrievalOp", "q2")
    assert round2_internal in graph
    leaf_output = graph[f"output_{round2_internal}"]
    assert leaf_output["path"] == "items"
    assert round_build.leaf_output_names == [f"leaf_{round2_internal}"]


def test_walk_archived_items_preserves_observation_for_llm_leaf():
    # Leaf outputs archive the whole result item (path: items). Walking each
    # stored item with the producer's default path recovers the same text value
    # the observation lambda saw before forwarding, so the planner's preview is
    # unchanged, and a row-wise producer's items each keep a row.
    llm_id = "s0"
    configs = {llm_id: _llm_config(llm_id)}
    walked = walk_archived_items({llm_id: [json.dumps({"output": "hello"})]}, configs)
    assert walked == {llm_id: ["hello"]}
    assert "hello" in compute_observation(walked, preview_width=100)
    rows = {llm_id: [json.dumps({"output": v}) for v in ("a", "b", "c")]}
    assert walk_archived_items(rows, configs) == {llm_id: ["a", "b", "c"]}
    # A Lambda leaf's default path keeps its archived output as one JSON item.
    lambda_op = {"id": "l0", "op": "LambdaOp", "mode": "list", "fn_name": "f"}
    lambda_op["code"] = "def f(items):\n    return items[0]\n"
    archived = {"l0": [json.dumps({"output": json.dumps([{"done": False}])})]}
    assert walk_archived_items(archived, {"l0": lambda_op}) == {
        "l0": [json.dumps([{"done": False}])]
    }


def test_walk_archived_items_rejects_non_json():
    # Archived values are always JSON items now; a non-JSON value is a protocol
    # violation, not a plain-text fallback.
    llm_id = "s0"
    configs = {llm_id: _llm_config(llm_id)}
    archived = {llm_id: ["not json"]}
    with pytest.raises(DriverProtocolError, match="not a JSON item"):
        walk_archived_items(archived, configs)


def _three_row_subgraph(ref: str) -> list[dict]:
    return [
        {
            "id": "q2",
            "op": "FormatOp",
            "inputs": [ref],
            "template": "{ref0}",
            "format_kwargs": {"ref0": ref},
        }
    ]


_ROW_LAMBDA_SOURCE: dict[str, Any] = {
    "id": "L",
    "op": "LambdaOp",
    "mode": "row",
    "inputs": [],
    "fn_name": "row",
    "code": "def row(columns):\n    return columns[0]\n",
}
_TABLE_ROWS = ["r1", "r2", "r3", "r4", "r5"]
_ONE_TABLE_ITEM = [json.dumps(_TABLE_ROWS)]


def _same_round_source_leaf(op_id: str, node: str, path: str) -> dict[str, Any]:
    return _llm_config(
        op_id,
        config={"model": "m"},
        messages=[{"role": "user", "content": "{c0}"}],
        rowwise_template="{c0}",
        rowwise_columns=[{"label": "c0", "node": node, "path": path}],
    )


def _s3_config(op_id: str) -> dict:
    return {
        "id": op_id,
        "op": "DataRetrievalOp",
        "inputs": [],
        "data_spec": {
            "type": "s3",
            "connection_string": "s3://bucket",
            "template": "k",
        },
    }


def _plain_chat(op_id: str, source: str) -> dict[str, Any]:
    return _llm_config(
        op_id,
        config={"model": "m"},
        messages=[{"role": "user", "content": "{c0}"}],
        prompt={"template": "x", "format_kwargs": {"c0": source}},
    )


@pytest.mark.parametrize(
    ("source", "rows", "node", "path", "expected", "source_in_round"),
    [
        # A row-mode Lambda's column yields one row per item; a retrieval column
        # projects into rows only when its path reaches into them, so a column
        # over the whole table is ungrouped and the consumer builds one row per
        # retrieval item instead. A chat op that is not row-wise yields one
        # scalar per item, so a leaf reading it is ungrouped even at one symbol,
        # where the fallback would bind its answers as one blob.
        (_ROW_LAMBDA_SOURCE, 1, "L", "items.output", _ONE_TABLE_ITEM, True),
        (_sql_config("t1"), 3, "t1", "items.table.symbol", _ONE_TABLE_ITEM, True),
        (_s3_config("t1"), 3, "t1", "items.content", _ONE_TABLE_ITEM, True),
        (_sql_config("t1"), 3, "t1", "items.table", _TABLE_ROWS, True),
        (_plain_chat("c0", INPUT_NODE_ID), 1, "c0", "items.output", _TABLE_ROWS, True),
        # A column reading an earlier round's op becomes a literal list input,
        # which FlowMesh reads ungrouped. The decision needs that round's
        # results: against an empty history the source is unknown, not literal.
        (_sql_config("c0"), 1, "c0", "items.output", _TABLE_ROWS, False),
        # A row-wise leaf reading another same-round row-wise leaf takes that
        # leaf's grouping, so a leaf chain over the input node stays ungrouped.
        (
            _same_round_source_leaf("L", INPUT_NODE_ID, ""),
            1,
            "L",
            "items.output",
            _TABLE_ROWS,
            True,
        ),
    ],
    ids=[
        "row-lambda-column",
        "retrieval-table-column",
        "retrieval-content-column",
        "whole-table",
        "scalar-chat",
        "earlier-round-source",
        "rowwise-leaf-chain",
    ],
)
def test_build_round_records_a_same_round_source_grouping(
    source: dict[str, Any],
    rows: int,
    node: str,
    path: str,
    expected: list[str],
    source_in_round: bool,
) -> None:
    # The producer's round records the grouping, because the consuming round
    # would otherwise decide it against a ``results`` already holding the
    # producer's own round.
    leaf = _same_round_source_leaf("s0", node, path)
    ops = [source, leaf] if source_in_round else [leaf]
    registry = {op["id"]: op for op in (ops if source_in_round else [leaf, source])}
    producing = _build(
        ops,
        round_index=0,
        node_registry=registry,
        results={} if source_in_round else {source["id"]: [{"output": _TABLE_ROWS}]},
    )
    archived = {"output": _TABLE_ROWS}
    consuming = _build(
        _three_row_subgraph("s0"),
        results={"s0": [archived], source["id"]: [archived]},
        rows=rows,
        whole_refs={"s0"} if rows > 1 else set(),
        grouped_sources=producing.grouped_sources,
        node_registry=registry,
    )
    assert consuming.forwarded_inputs["s0"] == expected


def _build(ops: list[dict[str, Any]], **overrides: Any) -> Any:
    overrides.setdefault("node_registry", {op["id"]: op for op in ops})
    overrides.setdefault("round_index", 1)
    return build_round(
        ops,
        goal="g",
        observations=[],
        topology=[],
        preview_width=100,
        model="m",
        max_tokens=10,
        temperature=0.0,
        include_proposer=False,
        **overrides,
    )


def _api_llm_config(op_id: str, **overrides: Any) -> dict:
    return _llm_config(
        op_id,
        config={
            "model": "qwen3.8-27b",
            "api": {"url": "https://lum.id/llm/v1", "timeout_sec": 300},
        },
        **overrides,
    )


def _api_record(content: str) -> dict:
    return {"json": {"choices": [{"message": {"content": content}}]}}


@pytest.mark.parametrize(
    ("config", "archived", "expected"),
    [
        # An API leaf archives the raw response record, read at
        # ``json.choices[0].message.content`` rather than ``output``.
        (
            _api_llm_config("s0"),
            [_api_record("first"), _api_record("second")],
            ["first", "second"],
        ),
        # A row-wise API task groups one response per input row under ``rows``;
        # the item itself stays whole, as for a local row-wise leaf.
        (
            _api_llm_config("s0", rowwise_template="{row}"),
            [{"rows": [_api_record("a"), _api_record("b")]}],
            [json.dumps(["a", "b"])],
        ),
        # Any LLMOp with ``config.api`` dispatches as an API task, so an
        # image-generation leaf reads the record too.
        (
            _llm_config(
                "s0", op="ImageGenerationOp", config={"model": "m", "api": {"url": "u"}}
            ),
            [_api_record("a picture")],
            ["a picture"],
        ),
        # A gated chat op archives the row-gate merge record, whose per-row
        # value stays under ``output`` even when the gated call itself runs
        # through the API executor.
        (
            _api_llm_config(
                "s0", rowwise_template="{row}", row_gate={"node": "r0", "branch": "x"}
            ),
            [{"output": json.dumps({"row": 0, "output": "done"}), "value": {"row": 0}}],
            [json.dumps({"row": 0, "output": "done"})],
        ),
    ],
    ids=["chat", "rowwise", "image_leaf", "gated"],
)
def test_walk_archived_items_reads_an_api_record(
    config: dict, archived: list[dict], expected: list[str]
) -> None:
    assert walk_archived_items(
        {"s0": [json.dumps(item) for item in archived]}, {"s0": config}
    ) == {"s0": expected}


def test_forward_per_row_implicit_reference_aligns_items_to_rows():
    llm_id = "s0"
    results = {llm_id: [{"output": "a"}, {"output": "b"}, {"output": "c"}]}
    configs = {llm_id: _llm_config(llm_id)}
    _, inputs = forward_refs(_three_row_subgraph(llm_id), results, configs, rows=3)
    assert inputs == {llm_id: ["a", "b", "c"]}


def test_forward_per_row_explicit_reference_aligns_items_to_rows():
    # A scalar per row binds one value per row and keeps the rewritten
    # ``path: ""``, with no materializer.
    sql_id = "s1"
    results = {
        sql_id: [{"table": json.dumps({"close": {"0": float(i)}})} for i in range(3)]
    }
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [_chat_consumer(("close", sql_id, "items.table.close"))]
    ops, inputs = forward_refs(subgraph, results, configs, rows=3)
    assert inputs[f"{sql_id} | items.table.close"] == [
        json.dumps({"0": float(i)}) for i in range(3)
    ]
    assert len(ops) == 1
    row = ops[0]["aggregate_table"][0]
    assert row["node"] == f"{sql_id} | items.table.close"
    assert row["path"] == ""


def test_forward_per_row_rejects_item_count_that_is_not_the_row_count():
    llm_id = "s0"
    configs = {llm_id: _llm_config(llm_id)}
    for items in (
        [{"output": "a"}],
        [{"output": "a"}, {"output": "b"}],
    ):
        with pytest.raises(DriverProtocolError, match="q2.*s0.*3-row run"):
            forward_refs(_three_row_subgraph(llm_id), {llm_id: items}, configs, rows=3)


def test_forward_per_row_rejects_variable_length_row_value():
    llm_id = "s0"
    results = {llm_id: [{"output": ["a"]}, {"output": ["b", "c"]}, {"output": ["d"]}]}
    configs = {llm_id: _llm_config(llm_id)}
    with pytest.raises(DriverProtocolError, match="q2.*s0.*some rows hold a list"):
        forward_refs(_three_row_subgraph(llm_id), results, configs, rows=3)


def test_forward_grouped_list_aggregate_column_materializes():
    # A per-row LIST (the symbol's news summaries) cannot ride a workflow input
    # literal (which is ungrouped on the worker), so an explicit aggregate
    # column whose walked per-row value is a list binds a JSON input per row
    # plus a list-mode LambdaOp materializer, and the consumer reads the
    # materializer's items.output.
    sql_id = "s1"
    results = {
        sql_id: [
            {"output": ["a", "b"]},
            {"output": ["c", "d"]},
            {"output": ["e", "f", "g"]},
        ]
    }
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [_chat_consumer(("summary", sql_id, "items.output"))]
    ops, inputs = forward_refs(subgraph, results, configs, rows=3)
    input_id = f"{sql_id} | items.output"
    materializer_id = f"{input_id} | rows"
    # The JSON input holds one dumps per row.
    assert inputs[input_id] == [
        json.dumps(["a", "b"]),
        json.dumps(["c", "d"]),
        json.dumps(["e", "f", "g"]),
    ]
    # The consumer reads the materializer's items.output.
    row = ops[0]["aggregate_table"][0]
    assert row["node"] == materializer_id
    assert row["path"] == "items.output"
    # The materializer is a row-aligned LambdaOp decoding each row's JSON.
    materializer = ops[1]
    assert materializer["id"] == materializer_id
    assert materializer["op"] == "LambdaOp"
    assert materializer["inputs"] == [input_id]
    assert materializer["mode"] == "aligned"
    assert "json.loads" in materializer["code"]
    # The rewritten ops plus the materializer parse as a valid workflow.
    assert parse_yaml_payload(
        {"name": "round_1", "inputs": inputs, "ops": ops, "outputs": []}
    )


def test_forward_rowwise_columns_reference():
    # ``rowwise_columns`` entries with a node + path are explicit references
    # and are forwarded like aggregate_table columns.
    sql_id = "s1"
    results = {sql_id: [{"table": json.dumps({"title": {"0": "a"}})}]}
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [
        {
            "id": "q5",
            "op": "LLMVisionOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "rowwise_columns": [
                {"label": "title", "node": sql_id, "path": "items.table.title"}
            ],
        }
    ]
    ops, inputs = forward_refs(subgraph, results, configs)
    column = ops[0]["rowwise_columns"][0]
    input_id = f"{sql_id} | items.table.title"
    assert column["node"] == input_id
    assert column["path"] == ""
    assert inputs[input_id] == [json.dumps({"0": "a"})]


def test_forward_grouped_rowwise_columns_materializes():
    # A grouped rowwise_columns ref across a cut is forwarded through a
    # materializer, one per (ref, path).
    sql_id = "s1"
    results = {
        sql_id: [
            {"output": ["a", "b"]},
            {"output": ["c", "d"]},
        ]
    }
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [
        {
            "id": "q5",
            "op": "LLMVisionOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "rowwise_columns": [
                {"label": "summary", "node": sql_id, "path": "items.output"}
            ],
        }
    ]
    ops, inputs = forward_refs(subgraph, results, configs, rows=2)
    input_id = f"{sql_id} | items.output"
    materializer_id = f"{input_id} | rows"
    column = ops[0]["rowwise_columns"][0]
    assert column["node"] == materializer_id
    assert column["path"] == "items.output"
    assert ops[1]["id"] == materializer_id
    assert ops[1]["mode"] == "aligned"


def test_forward_grouped_implicit_reference_materializes():
    # An implicit ``inputs`` reference whose per-row value is a list (News
    # Report's inputs naming News Per-row Summary) is rewritten to the
    # materializer id instead of raising the one-scalar-per-row error.
    llm_id = "s0"
    results = {
        llm_id: [{"output": ["a", "b"]}, {"output": ["c", "d"]}],
    }
    configs = {llm_id: _llm_config(llm_id)}
    subgraph = [
        {
            "id": "q2",
            "op": "FormatOp",
            "inputs": [llm_id],
            "template": "{ref0}",
            "format_kwargs": {"ref0": llm_id},
        }
    ]
    ops, inputs = forward_refs(subgraph, results, configs, rows=2)
    materializer_id = f"{llm_id} | items.output | rows"
    # The inputs entry and the format_kwargs value are rewritten to the
    # materializer id.
    assert ops[0]["inputs"] == [materializer_id]
    assert ops[0]["format_kwargs"] == {"ref0": materializer_id}
    assert ops[1]["id"] == materializer_id
    assert ops[1]["mode"] == "aligned"
    assert inputs[f"{llm_id} | items.output"] == [
        json.dumps(["a", "b"]),
        json.dumps(["c", "d"]),
    ]


def test_forward_grouped_image_source_rejected():
    # An image_source whose per-row value is a list cannot be JSON-forwarded;
    # it is rejected naming the op.
    llm_id = "s0"
    results = {
        llm_id: [{"output": ["a", "b"]}, {"output": ["c", "d"]}],
    }
    configs = {llm_id: _llm_config(llm_id)}
    subgraph = [
        {
            "id": "q6",
            "op": "LLMVisionOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "image_source": llm_id,
        }
    ]
    with pytest.raises(DriverProtocolError, match="q6.*image_source.*s0"):
        forward_refs(subgraph, results, configs, rows=2)


def _df_item(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a stored retrieval item in the real serialized-DataFrame shape."""
    columns: dict[str, dict[str, Any]] = {}
    for row_index, record in enumerate(rows):
        for col, value in record.items():
            columns.setdefault(col, {})[str(row_index)] = value
    return {"table": {"df": json.dumps(columns)}}


def test_forward_serialized_dataframe_columns_materialize():
    # A 2-symbol News Query -> News Report cut forwards table columns
    # (title/publisheddate/category) through materializers: each walked per-row
    # value is a list, so each column binds a JSON input per row plus a
    # list-mode LambdaOp materializer.
    sql_id = "s1"
    results = {
        sql_id: [
            _df_item(
                [
                    {"title": "a0", "publisheddate": "2024-01-01", "category": "x"},
                    {"title": "a1", "publisheddate": "2024-01-02", "category": "y"},
                ]
            ),
            _df_item(
                [
                    {"title": "b0", "publisheddate": "2024-02-01", "category": "p"},
                    {"title": "b1", "publisheddate": "2024-02-02", "category": "q"},
                ]
            ),
        ]
    }
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [
        _chat_consumer(
            ("title", sql_id, "items.table.title"),
            ("publisheddate", sql_id, "items.table.publisheddate"),
            ("category", sql_id, "items.table.category"),
        )
    ]
    ops, inputs = forward_refs(subgraph, results, configs, rows=2)
    # One materializer per (ref, path), plus the consumer op.
    assert len(ops) == 4
    for label, values in [
        ("title", [["a0", "a1"], ["b0", "b1"]]),
        ("publisheddate", [["2024-01-01", "2024-01-02"], ["2024-02-01", "2024-02-02"]]),
        ("category", [["x", "y"], ["p", "q"]]),
    ]:
        input_id = f"{sql_id} | items.table.{label}"
        materializer_id = f"{input_id} | rows"
        assert inputs[input_id] == [json.dumps(v) for v in values]
        column = next(c for c in ops[0]["aggregate_table"] if c["label"] == label)
        assert column["node"] == materializer_id
        assert column["path"] == "items.output"
        materializer = next(o for o in ops if o["id"] == materializer_id)
        assert materializer["op"] == "LambdaOp"
        assert materializer["mode"] == "aligned"
        assert materializer["inputs"] == [input_id]


def test_forward_aligned_materializers_are_row_cardinality():
    # A forwarded round whose report reads two materializers compiles as a
    # multi-row group: the materializer runtime nodes are aligned (not list),
    # so neither they nor the report are whole-list-per-run, and the report's
    # columns still read the materializers at value.items.output.
    sql_id = "s1"
    sql_id2 = "s2"
    results = {
        sql_id: [
            {"output": ["a", "b"]},
            {"output": ["c", "d"]},
            {"output": ["e", "f"]},
        ],
        sql_id2: [
            {"output": ["g", "h"]},
            {"output": ["i", "j"]},
            {"output": ["k", "l"]},
        ],
    }
    configs = {sql_id: _sql_config(sql_id), sql_id2: _sql_config(sql_id2)}
    subgraph = [
        _chat_consumer(
            ("title", sql_id, "items.output"),
            ("summary", sql_id2, "items.output"),
        )
    ]
    ops, inputs = forward_refs(subgraph, results, configs, rows=3)
    inputs["Symbols"] = ["AAA", "BBB", "CCC"]
    workflow = {
        "name": "r1",
        "inputs": inputs,
        "ops": ops,
        "outputs": [{"name": "o", "ref": "q4"}],
    }
    spec = next(iter(parse_yaml_payload(workflow).values()))
    graph = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    # The two materializers compile to aligned python steps.
    aligned = [
        node
        for node in graph.nodes.values()
        if node.task_type == "python" and node.data_spec.get("mode") == "aligned"
    ]
    assert len(aligned) == 2
    # Neither the materializers nor News Report are whole-list-per-run.
    cardinality = graph.list_lambda_cardinality_nodes()
    assert not any(node.node_id in cardinality for node in aligned)
    report = next(
        node
        for node in graph.nodes.values()
        if node.task_type == "inference" and "q4" in node.node_id
    )
    assert report.node_id not in cardinality
    # News Report's columns still read the materializers at value.items.output.
    df_columns = report.data_spec["template"]["columns"][0]["data"]["columns"]
    assert [c["label"] for c in df_columns] == ["title", "summary"]
    for column in df_columns:
        assert column["path"] == "value.items.output"


def _rowwise_llm_config(op_id: str, **overrides: Any) -> dict:
    return _llm_config(
        op_id,
        rowwise_template="{row}",
        rowwise_columns=[{"label": "row", "node": INPUT_NODE_ID}],
        **overrides,
    )


@pytest.mark.parametrize("leaf_op", ["LLMChatOp", "LLMVisionOp"])
def test_forwarded_literal_input_binds_as_a_data_column(leaf_op: str) -> None:
    # A forwarded reference becomes a workflow input that only a rowwise column
    # names, so no op takes an edge from it. The runtime graph must still
    # resolve the id to the graph's input op and bind its values as a literal
    # data column, on a row-wise chat and on a row-wise VLM alike.
    values = ["v0", "v1", "v2"]
    leaf = _llm_config(
        "C_r1",
        op=leaf_op,
        config={"model": "m"},
        messages=[{"role": "user", "content": "grade {row}"}],
        rowwise_template="{row}",
        rowwise_columns=[{"label": "row", "node": "C_r0", "path": "items.output"}],
    )
    if leaf_op == "LLMVisionOp":
        leaf["image_source"] = "C_r0"
    ops, inputs = forward_refs(
        [leaf],
        {"C_r0": [{"output": values}]},
        {"C_r0": _rowwise_llm_config("C_r0")},
        rows=3,
        whole_refs={"C_r0"},
    )
    inputs["Symbols"] = ["NVDA", "AAPL", "MSFT"]
    spec = next(
        iter(
            parse_yaml_payload(
                {"name": "r2", "inputs": inputs, "ops": ops, "outputs": []}
            ).values()
        )
    )
    graph = RuntimeGraphBuilder().build(
        Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    )
    node = next(
        n
        for n in graph.nodes.values()
        if "C_r1" in n.node_id and n.data_spec.get("type") != "list"
    )
    columns = (
        node.data_spec["template"]["columns"]
        if leaf_op == "LLMVisionOp"
        else node.data_spec["columns"]
    )
    assert columns[0] == {"label": "row", "data": {"type": "list", "items": values}}
