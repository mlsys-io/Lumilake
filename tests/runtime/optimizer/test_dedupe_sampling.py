"""Structural dedupe keeps independent random draws apart."""

import copy
from typing import Any

import pytest

from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.optimizer.halo import HaloOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder


def _payload(chat_op_ids: list[str], config: dict[str, Any]) -> dict[str, Any]:
    """A YAML workflow whose chat ops feed a Python step, so none of them is an
    output node and each is a dedupe candidate."""
    chat_config: dict[str, Any] = {"model": "m"}
    chat_config.update(config)
    chat_ops = [
        {
            "id": op_id,
            "op": "LLMChatOp",
            "inputs": ["Topic"],
            "messages": [{"role": "user", "content": "Explain {ref0}"}],
            "config": chat_config,
        }
        for op_id in chat_op_ids
    ]
    return {
        "name": "chat-twins",
        "inputs": {"Topic": ["a database index"]},
        "ops": [
            *chat_ops,
            {
                "id": "Report",
                "op": "LambdaOp",
                "inputs": list(chat_op_ids),
                "fn_name": "report",
                "mode": "row",
                "code": "def report(rows):\n    return rows[0]",
            },
        ],
        "outputs": [{"name": "report", "ref": "Report"}],
    }


@pytest.mark.parametrize(
    ("chat_op_ids", "config", "slice_prefixes", "expected_chat_nodes"),
    [
        (["AskA", "AskB"], {"temperature": 0.7}, ["wf"], 2),
        (["AskA", "AskB"], {"temperature": 0.0}, ["wf"], 1),
        (["AskA", "AskB"], {"temperature": 0.7, "seed": 7}, ["wf"], 2),
        (["AskA"], {"temperature": 0.7, "seed": 7}, ["slice_a", "slice_b"], 1),
    ],
    ids=[
        "unseeded-draws-stay-apart",
        "greedy-twins-merge",
        "seeded-ops-of-different-logical-ops-stay-apart",
        "seeded-op-of-one-slice-merges-with-its-coalesced-twin",
    ],
)
def test_dedupe_respects_random_draws(
    chat_op_ids: list[str],
    config: dict[str, Any],
    slice_prefixes: list[str],
    expected_chat_nodes: int,
) -> None:
    spec = next(
        iter(parse_yaml_payload(copy.deepcopy(_payload(chat_op_ids, config))).values())
    )
    compiled = Graph.from_json(spec["graph"]).compile(**spec["inputs"])
    graphs = {
        prefix: RuntimeGraphBuilder().build(compiled, node_prefix=prefix)
        for prefix in slice_prefixes
    }

    merged, _, _ = HaloOptimizer().optimize_graphs(graphs)

    chat_nodes = {
        node_id
        for runtime_ids in merged.dsl_to_runtime.values()
        for node_id in runtime_ids
        if merged.nodes[node_id].task_type == "inference"
    }
    assert len(chat_nodes) == expected_chat_nodes
