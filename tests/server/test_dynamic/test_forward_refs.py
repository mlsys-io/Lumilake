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
    has_live_rows,
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


def test_has_live_rows_ignores_completely_terminal_sets() -> None:
    # A row-gated leaf archives the merge step's record, which wraps each row as
    # {"output": <row json>, "value": <row>}.
    def merged(done: bool) -> str:
        row = {"selected": True, "done": done, "branch": "finish", "row": 0}
        return json.dumps({"output": json.dumps(row), "value": row})

    assert has_live_rows({"leaf": [merged(True)]}) is False
    assert has_live_rows({"leaf": [merged(True), merged(False)]}) is True
    assert has_live_rows({"state": ["ordinary-output"]}) is True
    assert has_live_rows({}) is True
    assert has_live_rows({"state": []}) is True


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


def test_runtime_rejects_format_op_leaf_source() -> None:
    graph = {
        "input_graph_Symbols": {
            "_id": "input_graph_Symbols",
            "_op": "InputOp",
            "_inputs": [],
            "_max_iter": None,
            "name": "Symbols",
        },
        "format_graph_leaf": {
            "_id": "format_graph_leaf",
            "_op": "FormatOp",
            "_inputs": ["input_graph_Symbols"],
            "_max_iter": None,
            "template": "{ref0}",
            "format_args": ["input_graph_Symbols"],
            "format_kwargs": {},
        },
        "output_graph_leaf": {
            "_id": "output_graph_leaf",
            "_op": "OutputOp",
            "_inputs": ["format_graph_leaf"],
            "_max_iter": None,
            "name": "leaf",
            "path": "items",
        },
    }
    compiled = Graph.from_json(graph).compile(Symbols=["a"])
    with pytest.raises(ValueError, match="must be an LLMOp"):
        RuntimeGraphBuilder().build(compiled)


def _llm_column_library() -> dict[str, dict[str, Any]]:
    return {
        "ReviewTurn": {
            "op": "LLMChatOp",
            "inputs": ["ReviewPrompt"],
            "messages": [{"role": "user", "content": "review"}],
            "rowwise_template": "{prompt}",
            "rowwise_columns": [
                {
                    "label": "prompt",
                    "node": "ReviewPrompt",
                    "path": "items.output.prompt",
                },
                {
                    "label": "review_system",
                    "node": "ReviewPrompt",
                    "path": "items.output.review_system",
                },
            ],
            "column_bindings": [
                {"label": "prompt", "input": 0},
                {"label": "review_system", "input": 0},
            ],
            "config": {
                "model": "Qwen/Qwen3.8-27B-FP8",
                "temperature": 0.05,
                "max_tokens": 200,
            },
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
    assert [
        {column["node"] for column in turn["rowwise_columns"]} for turn in turns
    ] == [
        {"first_prompt_1"},
        {"second_prompt_1"},
    ]
    assert [turn["rowwise_columns"][0]["node"] for turn in turns] == [
        "first_prompt_1",
        "second_prompt_1",
    ]
    assert {column["node"] for column in library["ReviewTurn"]["rowwise_columns"]} == {
        "ReviewPrompt"
    }


@pytest.mark.parametrize(
    ("bindings", "extra_input", "message"),
    [
        # A library input the round passes but the bindings never name.
        (None, "unused_prompt", "unused"),
        # A column the bindings never fill.
        ([{"label": "prompt", "input": 0}], None, "review_system"),
        # A binding naming a column the op does not declare.
        ([{"label": "missing", "input": 0}], None, "unknown label 'missing'"),
        # A binding naming an input slot the round does not pass.
        ([{"label": "prompt", "input": -1}], None, "invalid input slot -1"),
    ],
    ids=["unused-input", "unbound-column", "unknown-label", "bad-slot"],
)
def test_column_bindings_reject_an_inconsistent_round(
    bindings: list[dict[str, Any]] | None, extra_input: str | None, message: str
) -> None:
    library = _llm_column_library()
    if bindings is not None:
        library["ReviewTurn"]["column_bindings"] = bindings
    subgraph = _llm_column_subgraph("review")
    if extra_input is not None:
        subgraph[1]["inputs"].append(extra_input)

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
    # Leaf outputs now archive the whole result item (path: items). Walking each
    # stored item with the producer's default path must recover the same text
    # value the observation lambda saw before forwarding, so the observation the
    # planner sees is unchanged, and a row-wise producer's items each keep a row.
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

    rows = {
        llm_id: [
            json.dumps({"output": "a"}),
            json.dumps({"output": "b"}),
            json.dumps({"output": "c"}),
        ]
    }
    assert walk_archived_items(rows, configs) == {llm_id: ["a", "b", "c"]}


def test_walk_archived_items_uses_output_for_lambda_leaf() -> None:
    lambda_id = "l0"
    configs = {
        lambda_id: {
            "id": lambda_id,
            "op": "LambdaOp",
            "mode": "list",
            "fn_name": "states",
            "code": "def states(items):\n    return items[0]\n",
        }
    }
    archived = {lambda_id: [json.dumps({"output": json.dumps([{"done": False}])})]}
    walked = walk_archived_items(archived, configs)
    assert walked == {lambda_id: [json.dumps([{"done": False}])]}


def test_walk_archived_items_rejects_non_json():
    # Archived values are always JSON items now; a non-JSON value is a protocol
    # violation, not a plain-text fallback.
    llm_id = "s0"
    configs = {llm_id: _llm_config(llm_id)}
    archived = {llm_id: ["not json"]}
    with pytest.raises(DriverProtocolError, match="not a JSON item"):
        walk_archived_items(archived, configs)


def _sourced_rowwise_config(op_id: str, *sources: str) -> dict:
    config = _llm_config(op_id)
    config["rowwise_template"] = "{c0}"
    config["rowwise_columns"] = [
        {
            "label": f"c{index}",
            "node": source,
            **({} if source == INPUT_NODE_ID else {"path": "items.output"}),
        }
        for index, source in enumerate(sources)
    ]
    return config


_ROW_COUNT_SOURCE_CASES: list[Any] = [
    pytest.param(
        {"s0": _sourced_rowwise_config("s0", INPUT_NODE_ID)},
        {"s0": [{"output": ["a0", "a1", "a2"]}]},
        1,
        {"s0": ["a0", "a1", "a2"]},
        id="input-node-one-symbol",
    ),
    pytest.param(
        {"c0": _llm_config("c0"), "s0": _sourced_rowwise_config("s0", "c0")},
        {"s0": [{"output": ["a0", "a1", "a2"]}]},
        3,
        {"s0": ["a0", "a1", "a2"]},
        id="same-round-chat-three-symbols",
    ),
    pytest.param(
        {"p": _llm_config("p"), "s0": _sourced_rowwise_config("s0", "p")},
        {"p": [{"output": ["v0", "v1", "v2"]}], "s0": [{"output": ["a0", "a1", "a2"]}]},
        1,
        {"s0": ["a0", "a1", "a2"]},
        id="earlier-round-source-one-symbol",
    ),
    pytest.param(
        {"t1": _sql_config("t1"), "s0": _sourced_rowwise_config("s0", "t1")},
        {"s0": [{"output": ["r1", "r2", "r3", "r4", "r5"]}]},
        1,
        {"s0": [json.dumps(["r1", "r2", "r3", "r4", "r5"])]},
        id="inner-table-single-symbol",
    ),
    pytest.param(
        {
            "L": {"id": "L", "op": "LambdaOp", "mode": "list", "inputs": []},
            "S": _sourced_rowwise_config("S", "L"),
        },
        {"S": [{"output": ["a1", "a2"]}, {"output": ["b1", "b2"]}]},
        3,
        {"S": [json.dumps(["a1", "a2"]), json.dumps(["b1", "b2"])]},
        id="list-lambda-list-elements",
    ),
]


@pytest.mark.parametrize(
    ("configs", "results", "rows", "expected"),
    _ROW_COUNT_SOURCE_CASES,
)
def test_forward_whole_rowwise_leaf_groups_by_its_source(
    configs: dict[str, dict], results: dict[str, list[dict]], rows: int, expected: dict
) -> None:
    # A row-wise leaf whose column reads the input node or an earlier-round op
    # reads a literal list input, which FlowMesh does not group: the leaf's one
    # table holds its own rows, so the whole binding expands it whatever the
    # symbol count. A column reading a same-round op leaves the grouping
    # unknown, and the binding falls back to expanding over more than one
    # symbol.
    leaf_id = next(
        op_id for op_id, config in configs.items() if config.get("rowwise_columns")
    )
    _, inputs = forward_refs(
        _three_row_subgraph(leaf_id),
        results,
        configs,
        rows=rows,
        whole_refs={leaf_id},
    )
    assert inputs == expected


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


def _exporting_reader(source: str) -> dict[str, Any]:
    return {
        "id": "R",
        "op": "LambdaOp",
        "mode": "list",
        "inputs": [source],
        "fn_name": "reader",
        "code": "def reader(columns):\n    return columns[0]\n",
    }


def _same_round_source_leaf(
    op_id: str, *columns: tuple[str, str | None]
) -> dict[str, Any]:
    config = _llm_config(op_id)
    config["config"] = {"model": "m"}
    config["messages"] = [{"role": "user", "content": "{c0}"}]
    config["rowwise_template"] = "{c0}"
    config["rowwise_columns"] = [
        {"label": f"c{index}", "node": node, **({} if path is None else {"path": path})}
        for index, (node, path) in enumerate(columns)
    ]
    return config


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


_SAME_ROUND_SOURCE_CASES: list[Any] = [
    # A row-mode Lambda's column yields one row per item.
    pytest.param(
        _ROW_LAMBDA_SOURCE,
        False,
        1,
        [("L", "items.output")],
        _ONE_TABLE_ITEM,
        id="row-lambda-column",
    ),
    # A retrieval column inside the table projects into rows.
    pytest.param(
        _sql_config("t1"),
        False,
        3,
        [("t1", "items.table.symbol")],
        _ONE_TABLE_ITEM,
        id="retrieval-table-column",
    ),
    pytest.param(
        _sql_config("t1"),
        True,
        3,
        [("t1", "items.table.symbol")],
        _ONE_TABLE_ITEM,
        id="retrieval-table-column-exported",
    ),
    pytest.param(
        _s3_config("t1"),
        False,
        3,
        [("t1", "items.content")],
        _ONE_TABLE_ITEM,
        id="retrieval-content-column",
    ),
    # A column over the whole table does not project into rows.
    pytest.param(
        _sql_config("t1"),
        False,
        3,
        [("t1", "items.table")],
        _TABLE_ROWS,
        id="retrieval-whole-table",
    ),
]


@pytest.mark.parametrize(
    ("source", "exported", "rows", "columns", "expected"), _SAME_ROUND_SOURCE_CASES
)
def test_build_round_records_a_same_round_source_grouping(
    source: dict[str, Any],
    exported: bool,
    rows: int,
    columns: list[tuple[str, str | None]],
    expected: list[str],
) -> None:
    # The producer's round records the grouping, because the consuming round
    # would otherwise decide it against a ``results`` already holding the
    # producer's own round. A column over the whole ``items.table`` is not
    # grouped, so the consumer builds one row per retrieval item instead.
    ops = [source, _same_round_source_leaf("s0", *columns)]
    export: tuple[str, ...] = ()
    if exported:
        ops.append(_exporting_reader(source["id"]))
        export = (source["id"],)
    registry = {op["id"]: op for op in ops}
    producing = _build(ops, round_index=0, export=export, registry=registry)
    archived = {"output": _TABLE_ROWS}
    consuming = _build(
        _three_row_subgraph("s0"),
        results={"s0": [archived], source["id"]: [archived]},
        rows=rows,
        whole_refs={"s0"} if rows > 1 else set(),
        grouped_sources=producing.grouped_sources,
        registry=registry,
    )
    assert consuming.forwarded_inputs["s0"] == expected


def _build(
    ops: list[dict[str, Any]],
    *,
    results: dict[str, list[dict]] | None = None,
    rows: int = 1,
    round_index: int = 1,
    whole_refs: set[str] | None = None,
    export: tuple[str, ...] = (),
    grouped_sources: dict[str, bool | None] | None = None,
    registry: dict[str, dict[str, Any]] | None = None,
) -> Any:
    node_registry = registry or {op["id"]: op for op in ops}
    return build_round(
        ops,
        node_registry=node_registry,
        round_index=round_index,
        goal="g",
        observations=[],
        topology=[],
        preview_width=100,
        model="m",
        max_tokens=10,
        temperature=0.0,
        include_proposer=False,
        results=results,
        rows=rows,
        whole_refs=whole_refs,
        grouped_sources=grouped_sources,
        export=export,
    )


def _plain_chat(op_id: str, source: str) -> dict[str, Any]:
    config = _llm_config(op_id)
    config["config"] = {"model": "m"}
    config["messages"] = [{"role": "user", "content": "{c0}"}]
    config["prompt"]["format_kwargs"] = {"c0": source}
    return config


@pytest.mark.parametrize("kind", ["same_round_chat", "earlier_round_op"])
def test_build_round_decides_an_ungrouped_source(kind: str) -> None:
    if kind == "same_round_chat":
        # A chat op that is not row-wise yields one scalar per item, so a
        # row-wise leaf reading it builds one table of its own rows whatever the
        # symbol count: the leaf's answers expand even at one symbol, where the
        # fallback would bind them as one blob.
        ops = [
            _plain_chat("c0", INPUT_NODE_ID),
            _same_round_source_leaf("s0", ("c0", None)),
        ]
        producing = _build(ops, round_index=0)
        assert producing.grouped_sources == {"s0": False}
        consuming = _build(
            _three_row_subgraph("s0"),
            results={"s0": [{"output": _TABLE_ROWS}]},
            whole_refs={"s0"},
            grouped_sources=producing.grouped_sources,
            registry={op["id"]: op for op in ops},
        )
        assert consuming.forwarded_inputs["s0"] == _TABLE_ROWS
        return
    # A column reading an earlier round's op becomes a literal list input, which
    # FlowMesh reads ungrouped. The decision needs that op's archived results:
    # against an empty history the source is unknown rather than a literal.
    ops = [_same_round_source_leaf("s0", ("c0", None))]
    build = _build(ops, results={"c0": [{"output": _TABLE_ROWS}], "s0": []}, rows=3)
    assert build.grouped_sources == {"s0": False}


def _api_llm_config(op_id: str, **overrides: Any) -> dict:
    config = _llm_config(op_id)
    config["config"] = {
        "model": "qwen3.8-27b",
        "api": {"url": "https://lum.id/llm/v1", "timeout_sec": 300},
    }
    config.update(overrides)
    return config


def _api_record(content: str, *, index: int = 0) -> dict:
    return {
        "index": index,
        "url": "https://lum.id/llm/v1/chat/completions",
        "status_code": 200,
        "truncated": False,
        "json": {"choices": [{"message": {"content": content}}]},
        "usage": {"total_tokens": 3},
        "text": content,
        "prompt": "[]",
    }


@pytest.mark.parametrize(
    ("config", "archived", "expected"),
    [
        # An API chat op archives the raw response record, so its text is read
        # at ``json.choices[0].message.content`` rather than ``output``.
        (
            _api_llm_config("s0"),
            [json.dumps(_api_record("first")), json.dumps(_api_record("second"))],
            ["first", "second"],
        ),
        # A row-wise API task groups one response per input row under ``rows``,
        # so the walk reaches each row's content; the item itself stays whole,
        # as for a local row-wise leaf.
        (
            _api_llm_config("s0", rowwise_template="{row}"),
            [
                json.dumps(
                    {
                        "index": 0,
                        "rows": [_api_record("a", index=0), _api_record("b", index=1)],
                    }
                )
            ],
            [json.dumps(["a", "b"])],
        ),
        # The runtime dispatches any non-vision LLMOp with ``config.api`` as an
        # API task, not only a chat op, so an image-generation leaf archives the
        # raw response record too.
        (
            {
                "id": "s0",
                "op": "ImageGenerationOp",
                "inputs": [],
                "config": {
                    "model": "qwen3.8-27b",
                    "api": {"url": "https://example.com/v1"},
                },
            },
            [json.dumps(_api_record("a picture"))],
            ["a picture"],
        ),
    ],
    ids=["chat", "rowwise", "image_leaf"],
)
def test_walk_archived_items_reads_an_api_record(
    config: dict, archived: list[str], expected: list[str]
) -> None:
    assert walk_archived_items({"s0": archived}, {"s0": config}) == {"s0": expected}


def test_forward_implicit_gated_api_reference_binds_output():
    # A gated chat op archives the row-gate merge record, whose per-row value
    # stays under ``output`` even when the gated call itself runs through the
    # API executor.
    llm_id = "s0"
    row = {"selected": True, "done": True, "branch": "finish", "row": 0}
    item = {"output": json.dumps({**row, "output": "done text"}), "value": row}
    results = {llm_id: [item]}
    configs = {
        llm_id: _api_llm_config(
            llm_id,
            rowwise_template="{row}",
            row_gate={"node": "r0", "branch": "finish"},
        )
    }
    ops, inputs = forward_refs(_three_row_subgraph(llm_id), results, configs)
    assert inputs == {llm_id: [json.dumps({**row, "output": "done text"})]}


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
    subgraph = [_chat_consumer(("close", sql_id, "items.table.close"))]
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


def test_forward_scalar_per_row_keeps_today_path():
    # A scalar per row keeps today's path ("" after rewrite) and no materializer.
    sql_id = "s1"
    results = {
        sql_id: [{"table": json.dumps({"close": {"0": float(i)}})} for i in range(3)]
    }
    configs = {sql_id: _sql_config(sql_id)}
    subgraph = [_chat_consumer(("close", sql_id, "items.table.close"))]
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
    """Build a stored retrieval item in the real serialized-DataFrame shape."""
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
    config = _llm_config(op_id)
    config["rowwise_template"] = "{row}"
    config["rowwise_columns"] = [{"label": "row", "node": INPUT_NODE_ID}]
    config.update(overrides)
    return config


def test_forward_whole_rowwise_leaf_binds_one_value_per_row():
    # A row-wise LLM leaf archives one grouped item per input table, holding
    # every row's value. In a one-slice round that is one item for the whole
    # round, so a later round reading it whole must still get one value per
    # row: the same values a per-symbol round binds, so a consumer reading it
    # alongside ``Symbols`` lines up with its own rows.
    llm_id = "s0"
    symbols = ["A", "B", "C"]
    configs = {llm_id: _rowwise_llm_config(llm_id)}

    per_symbol = {llm_id: [{"output": [symbol]} for symbol in symbols]}
    _, per_symbol_inputs = forward_refs(
        _three_row_subgraph(llm_id), per_symbol, configs, rows=3
    )
    assert per_symbol_inputs == {llm_id: ["A", "B", "C"]}

    one_slice = {llm_id: [{"output": list(symbols)}]}
    ops, whole_inputs = forward_refs(
        _three_row_subgraph(llm_id), one_slice, configs, rows=3, whole_refs={llm_id}
    )
    assert ops[0]["inputs"] == [llm_id]
    assert whole_inputs == per_symbol_inputs

    # A row-gated leaf archives one merge record per row, so its whole result is
    # already one value per row and is bound without flattening.
    rows = [
        {"selected": True, "done": False, "branch": "pending", "row": index}
        for index in range(3)
    ]
    gated = {
        llm_id: [
            {"output": json.dumps({**row, "output": f"text-{index}"}), "value": row}
            for index, row in enumerate(rows)
        ]
    }
    _, gated_inputs = forward_refs(
        _three_row_subgraph(llm_id),
        gated,
        {
            llm_id: _rowwise_llm_config(
                llm_id, row_gate={"node": "r0", "branch": "finish"}
            )
        },
        rows=3,
        whole_refs={llm_id},
    )
    assert gated_inputs[llm_id] == [
        json.dumps({**row, "output": f"text-{i}"}) for i, row in enumerate(rows)
    ]


def test_forwarded_literal_input_resolves_for_a_single_node_reference_guard():
    # An embedding op reading a row-wise leaf by a single node reference can
    # only carry one row, so the graph refuses it. The leaf's row count comes
    # from its forwarded column, which resolves to the graph's input op even
    # though the op dict omits it.
    prior = "C_r0"
    values = ["v0", "v1", "v2"]
    round_ops = [
        {
            "id": "C_r1",
            "op": "LLMChatOp",
            "inputs": [],
            "messages": [{"role": "user", "content": "grade {row}"}],
            "rowwise_template": "{row}",
            "rowwise_columns": [
                {"label": "row", "node": prior, "path": "items.output"}
            ],
            "config": {"model": "m"},
        },
        {
            "id": "E1",
            "op": "EmbeddingOp",
            "inputs": ["C_r1"],
            "input": "C_r1",
            "config": {"model": "bge-m3"},
        },
    ]
    ops, inputs = forward_refs(
        round_ops,
        {prior: [{"output": list(values)}]},
        {prior: _rowwise_llm_config(prior)},
        rows=3,
        whole_refs={prior},
    )
    inputs["Symbols"] = ["NVDA", "AAPL", "MSFT"]
    workflow = {
        "name": "r2",
        "inputs": inputs,
        "ops": ops,
        "outputs": [{"name": "o", "ref": "E1"}],
    }
    spec = next(iter(parse_yaml_payload(workflow).values()))
    with pytest.raises(ValueError, match="produces multiple rows"):
        RuntimeGraphBuilder().build(
            Graph.from_json(spec["graph"]).compile(**spec["inputs"])
        )


@pytest.mark.parametrize("leaf_kind", ["chat", "vision"])
def test_forwarded_literal_input_binds_as_a_data_column(leaf_kind):
    # A forwarded reference becomes a workflow input that a rowwise column
    # names by id, so the input op carries no consumer edge and the sorted op
    # dict omits it. The runtime graph still resolves the id to the graph's
    # input op and binds its values as a literal data column, on a row-wise
    # chat and on a row-wise VLM alike.
    prior = "C_r0"
    values = ["v0", "v1", "v2"]
    leaf: dict[str, Any] = {
        "id": "C_r1",
        "op": "LLMVisionOp" if leaf_kind == "vision" else "LLMChatOp",
        "inputs": [],
        "messages": [{"role": "user", "content": "grade {row}"}],
        "rowwise_template": "{row}",
        "rowwise_columns": [{"label": "row", "node": prior, "path": "items.output"}],
        "config": {"model": "m"},
    }
    if leaf_kind == "vision":
        leaf["image_source"] = prior
    ops, inputs = forward_refs(
        [leaf],
        {prior: [{"output": list(values)}]},
        {prior: _rowwise_llm_config(prior)},
        rows=3,
        whole_refs={prior},
    )
    inputs["Symbols"] = ["NVDA", "AAPL", "MSFT"]
    workflow = {
        "name": "r2",
        "inputs": inputs,
        "ops": ops,
        "outputs": [{"name": "o", "ref": "C_r1"}],
    }
    spec = next(iter(parse_yaml_payload(workflow).values()))
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
        if leaf_kind == "vision"
        else node.data_spec["columns"]
    )
    assert columns[0] == {"label": "row", "data": {"type": "list", "items": values}}
