"""Unit tests for cross-round reference forwarding in dynamic rounds.

``forward_refs`` rewrites a round's subgraph so references to earlier-round
ops bind as workflow inputs carrying exactly the value the in-graph reference
would have read, instead of pulling the earlier op back into the round graph
(which would recompute it).
"""

import copy
import json
from typing import Any

import pytest

from lumilake_server.dynamic.driver import (
    DriverProtocolError,
    SubgraphPlan,
    build_round,
    compute_observation,
    forward_refs,
    validate_emitted_subgraph,
    walk_archived_items,
)
from lumilake_server.parser.common import make_id
from lumilake_server.parser.yaml_parser import _op_id_prefix, parse_yaml_payload
from lumilake_server.routes.jobs import _admit_subgraph


def _internal_id(scope: str, op_type: str, user_id: str) -> str:
    return make_id(scope, _op_id_prefix(op_type), user_id)


def _llm_config(op_id: str) -> dict:
    return {
        "id": op_id,
        "op": "LLMChatOp",
        "inputs": [],
        "prompt": {"template": "x", "format_kwargs": {}},
    }


def _sql_config(op_id: str) -> dict:
    return {
        "id": op_id,
        "op": "DataRetrievalOp",
        "inputs": [],
        "data_spec": {"type": "lumid", "mode": "sql", "template": "SELECT *"},
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
    subgraph = [
        {
            "id": "q4",
            "op": "LLMChatOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "aggregate_table": [
                {"label": "close", "node": sql_id, "path": "items.table.close"}
            ],
        }
    ]
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
    subgraph = [
        {
            "id": "q4",
            "op": "LLMChatOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "aggregate_table": [
                {"label": "close", "node": sql_id, "path": "items.table.close"}
            ],
        }
    ]
    original = copy.deepcopy(subgraph)
    forward_refs(subgraph, results, configs)
    assert subgraph == original


def test_forward_rewritten_ops_parse_through_yaml_parser():
    # The rewritten op (with a per-(node, path) input id and path "") must
    # still parse through the YAML parser as a valid workflow, with the input
    # id resolvable from the workflow inputs block.
    sql_id = "s1"
    results = {sql_id: [{"table": json.dumps({"close": {"0": 107.5}})}]}
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [
        {
            "id": "q4",
            "op": "DataRetrievalOp",
            "inputs": [],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "template": "SELECT * FROM t WHERE c = {p}",
                "params": [{"label": "p", "node": sql_id, "path": "items.table.close"}],
            },
        }
    ]
    ops, inputs = forward_refs(subgraph, results, configs)
    input_id = f"{sql_id} | items.table.close"
    workflow = {
        "name": "round_1",
        "inputs": {input_id: inputs[input_id]},
        "ops": ops,
        "outputs": [],
    }
    parsed = parse_yaml_payload(workflow)
    assert parsed


def test_validate_rejects_reference_to_unavailable_earlier_op():
    # A reference to an earlier op whose result is not available (neither a
    # leaf nor exported) must be rejected, never recomputed.
    node_registry = {"s0": _sql_config("s0")}
    subgraph = [
        {
            "id": "q1",
            "op": "DataRetrievalOp",
            "inputs": ["s0"],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "template": "SELECT * FROM t WHERE id = {p}",
                "params": [{"label": "p", "node": "s0", "path": "items.table.symbol"}],
            },
        }
    ]
    with pytest.raises(DriverProtocolError, match="s0"):
        validate_emitted_subgraph(subgraph, node_registry, max_nodes=4, results={})


def test_validate_accepts_reference_to_available_earlier_op():
    # A reference to an earlier op whose result IS available (a leaf or
    # exported) is accepted.
    node_registry = {"s0": _sql_config("s0")}
    subgraph = [
        {
            "id": "q1",
            "op": "DataRetrievalOp",
            "inputs": ["s0"],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "template": "SELECT * FROM t WHERE id = {p}",
                "params": [{"label": "p", "node": "s0", "path": "items.table.symbol"}],
            },
        }
    ]
    validate_emitted_subgraph(
        subgraph, node_registry, max_nodes=4, results={"s0": [{"table": "x"}]}
    )


def test_validate_rejects_export_not_in_subgraph():
    # ``export`` must name ops of the same subgraph.
    subgraph = [
        {
            "id": "q1",
            "op": "DataRetrievalOp",
            "inputs": [],
            "data_spec": {"type": "lumid", "mode": "sql", "template": "SELECT *"},
        }
    ]
    with pytest.raises(DriverProtocolError, match="exported op id 'nope'"):
        validate_emitted_subgraph(subgraph, {}, max_nodes=4, export=["nope"])


def test_validate_accepts_export_in_subgraph():
    subgraph = [
        {
            "id": "q1",
            "op": "DataRetrievalOp",
            "inputs": [],
            "data_spec": {"type": "lumid", "mode": "sql", "template": "SELECT *"},
        }
    ]
    validate_emitted_subgraph(subgraph, {}, max_nodes=4, export=["q1"])


def test_admit_subgraph_forwards_export_to_validation():
    # ``_admit_subgraph`` must pass the plan's ``export`` through to
    # ``validate_emitted_subgraph``; a plan naming an op that is not in its own
    # subgraph is rejected, and a valid export is admitted and registered.
    op = {
        "id": "q1",
        "op": "DataRetrievalOp",
        "inputs": [],
        "data_spec": {"type": "lumid", "mode": "sql", "template": "SELECT *"},
    }
    with pytest.raises(DriverProtocolError, match="exported op id 'nope'"):
        _admit_subgraph(SubgraphPlan(ops=[op], export=["nope"]), {}, 4, None, {})
    resolved = _admit_subgraph(SubgraphPlan(ops=[op], export=["q1"]), {}, 4, None, {})
    assert [r["id"] for r in resolved] == ["q1"]


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
    # Leaf outputs now archive the whole result item (path: items). Walking each
    # stored item with the producer's default path must recover the same text
    # value the observation lambda saw before forwarding, so the observation the
    # planner sees is unchanged.
    llm_id = "s0"
    configs = {llm_id: _llm_config(llm_id)}
    # The archived whole item for an LLM leaf; its default path is items.output.
    archived = {llm_id: [json.dumps({"output": "hello", "extra": 1})]}
    walked = walk_archived_items(archived, configs)
    assert walked == {llm_id: ["hello"]}
    # The observation computed from the walked leaves matches the pre-forwarding
    # text value (a single string row).
    obs = compute_observation(walked, preview_width=100)
    assert "rows=1" in obs
    assert "hello" in obs


def test_walk_archived_items_handles_multiple_rows():
    # A row-wise producer emits one archived item per row (path: items). Walking
    # must keep every item and the observation must count all rows.
    llm_id = "s0"
    configs = {llm_id: _llm_config(llm_id)}
    archived = {
        llm_id: [
            json.dumps({"output": "a"}),
            json.dumps({"output": "b"}),
            json.dumps({"output": "c"}),
        ]
    }
    walked = walk_archived_items(archived, configs)
    assert walked == {llm_id: ["a", "b", "c"]}
    obs = compute_observation(walked, preview_width=100)
    assert "rows=3" in obs


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


def test_forward_per_row_implicit_reference_aligns_items_to_rows():
    llm_id = "s0"
    results = {llm_id: [{"output": "a"}, {"output": "b"}, {"output": "c"}]}
    configs = {llm_id: _llm_config(llm_id)}
    _, inputs = forward_refs(_three_row_subgraph(llm_id), results, configs, rows=3)
    assert inputs == {llm_id: ["a", "b", "c"]}


def test_forward_per_row_explicit_reference_aligns_items_to_rows():
    sql_id = "s1"
    results = {
        sql_id: [{"table": json.dumps({"close": {"0": float(i)}})} for i in range(3)]
    }
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [
        {
            "id": "q4",
            "op": "LLMChatOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "aggregate_table": [
                {"label": "close", "node": sql_id, "path": "items.table.close"}
            ],
        }
    ]
    _, inputs = forward_refs(subgraph, results, configs, rows=3)
    assert inputs[f"{sql_id} | items.table.close"] == [
        json.dumps({"0": float(i)}) for i in range(3)
    ]


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
    subgraph = [
        {
            "id": "q4",
            "op": "LLMChatOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "aggregate_table": [
                {"label": "summary", "node": sql_id, "path": "items.output"}
            ],
        }
    ]
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


def test_forward_grouped_list_materializer_parses_through_yaml_parser():
    # The rewritten op plus the appended materializer must parse as a valid
    # workflow, with the materializer's input resolvable from the inputs block.
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
            "id": "q4",
            "op": "LLMChatOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "config": {"model": "Qwen/Qwen3-8B"},
            "messages": [{"role": "user", "content": "x"}],
            "aggregate_table": [
                {"label": "summary", "node": sql_id, "path": "items.output"}
            ],
        }
    ]
    ops, inputs = forward_refs(subgraph, results, configs, rows=2)
    workflow = {
        "name": "round_1",
        "inputs": inputs,
        "ops": ops,
        "outputs": [],
    }
    parsed = parse_yaml_payload(workflow)
    assert parsed


def test_forward_scalar_per_row_keeps_today_path():
    # A scalar per row keeps today's path ("" after rewrite) and no materializer.
    sql_id = "s1"
    results = {
        sql_id: [{"table": json.dumps({"close": {"0": float(i)}})} for i in range(3)]
    }
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [
        {
            "id": "q4",
            "op": "LLMChatOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "aggregate_table": [
                {"label": "close", "node": sql_id, "path": "items.table.close"}
            ],
        }
    ]
    ops, inputs = forward_refs(subgraph, results, configs, rows=3)
    assert len(ops) == 1
    row = ops[0]["aggregate_table"][0]
    assert row["node"] == f"{sql_id} | items.table.close"
    assert row["path"] == ""


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
    """Build a stored retrieval item in the real serialized-DataFrame shape:
    ``{"df": "<pandas to_json column-orient>"}`` = ``{col: {row_index: value}}``."""
    columns: dict[str, dict[str, Any]] = {}
    for row_index, record in enumerate(rows):
        for col, value in record.items():
            columns.setdefault(col, {})[str(row_index)] = value
    return {"table": {"df": json.dumps(columns)}}


def test_walk_serialized_dataframe_column_returns_list():
    # Walking ``table.<col>`` on a real stored item (a serialized DataFrame)
    # returns the column as a list in row order, not a dict or an error.
    from lumilake_server.runtime.runtime_manager.flowmesh import _walk_output_path

    item = _df_item(
        [
            {"title": "a0", "publisheddate": "2024-01-01", "category": "x"},
            {"title": "a1", "publisheddate": "2024-01-02", "category": "y"},
            {"title": "a2", "publisheddate": "2024-01-03", "category": "z"},
        ]
    )
    assert _walk_output_path(item, ("table", "title"), "nq") == ["a0", "a1", "a2"]
    assert _walk_output_path(item, ("table", "publisheddate"), "nq") == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert _walk_output_path(item, ("table", "category"), "nq") == ["x", "y", "z"]


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
        {
            "id": "q4",
            "op": "LLMChatOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "aggregate_table": [
                {"label": "title", "node": sql_id, "path": "items.table.title"},
                {
                    "label": "publisheddate",
                    "node": sql_id,
                    "path": "items.table.publisheddate",
                },
                {"label": "category", "node": sql_id, "path": "items.table.category"},
            ],
        }
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
    # A forwarded round where News Report reads two materializers compiles as a
    # multi-row group: the materializer runtime nodes are aligned (not list),
    # so neither they nor News Report are whole-list-per-run, and News Report's
    # columns still read the materializers at value.items.output.
    from lumilake_server.graphs import Graph
    from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

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
        {
            "id": "q4",
            "op": "LLMChatOp",
            "inputs": [],
            "prompt": {"template": "x", "format_kwargs": {}},
            "config": {"model": "Qwen/Qwen3-8B"},
            "messages": [{"role": "user", "content": "x"}],
            "aggregate_table": [
                {"label": "title", "node": sql_id, "path": "items.output"},
                {"label": "summary", "node": sql_id2, "path": "items.output"},
            ],
        }
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
