import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from lumilake import envs

from lumilake_server.parser import parse_n8n_payload, parse_yaml_payload
from lumilake_server.parser.n8n import N8N_CHAT_TRIGGER

TEMPLATE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "templates"
N8N_DIR = TEMPLATE_ROOT / "n8n"
YAML_DIR = TEMPLATE_ROOT / "yaml"


def _ensure_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _collect_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if not N8N_DIR.is_dir() or not YAML_DIR.is_dir():
        return pairs
    for n8n_path in sorted(N8N_DIR.glob("*.json")):
        yaml_path = YAML_DIR / f"{n8n_path.stem}.yaml"
        if yaml_path.exists():
            pairs.append((n8n_path, yaml_path))
    return pairs


def _build_inputs_from_n8n(workflow: dict[str, Any]) -> dict[str, list[str]]:
    inputs: dict[str, list[str]] = {}
    for node in workflow.get("nodes", []):
        if node.get("type") != N8N_CHAT_TRIGGER:
            continue
        name = node.get("name")
        if isinstance(name, str) and name:
            inputs[name] = []
    return inputs


def _input_op_names(graph: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {
        oid: str(op.get("name", ""))
        for oid, op in graph.items()
        if op.get("_op") == "InputOp"
    }


def _canon_entry(
    entry: dict[str, Any],
    input_ops: dict[str, str],
    inputs: dict[str, list[str]],
) -> str:
    if "node" in entry:
        node = entry["node"]
        if node in input_ops:
            input_name = input_ops[node]
            items = inputs.get(input_name, [])
            return f"INPUT:{json.dumps(items, sort_keys=True)}"
        path = entry.get("path", "")
        return f"NODE:{node}:{path}"
    if "data" in entry and isinstance(entry["data"], dict):
        data = entry["data"]
        if data.get("type") == "list":
            return f"INPUT:{json.dumps(data.get('items', []), sort_keys=True)}"
    return f"OPAQUE:{json.dumps(entry, sort_keys=True)}"


def _has_rowwise_consumer(graph: dict[str, dict[str, Any]], msg_id: str) -> bool:
    for downstream in graph.values():
        if downstream.get("messages") == msg_id and downstream.get("rowwise_template"):
            return True
    return False


def _canonicalize_op(
    op: dict[str, Any],
    graph: dict[str, dict[str, Any]],
    inputs: dict[str, list[str]],
) -> dict[str, Any]:
    op = copy.deepcopy(op)
    input_ops = _input_op_names(graph)
    label_map: dict[str, str] = {}

    if op.get("_op") == "MessageOp" and _has_rowwise_consumer(graph, op.get("_id", "")):
        for msg in op.get("messages", []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = ""

    for container_key, field in (
        ("data_spec", "params"),
        (None, "rowwise_columns"),
        (None, "aggregate_table"),
    ):
        bucket = op.get(container_key) if container_key else op
        if not isinstance(bucket, dict):
            continue
        items = bucket.get(field)
        if not isinstance(items, list):
            continue
        deduped: dict[str, dict[str, Any]] = {}
        for entry in items:
            if not isinstance(entry, dict) or "label" not in entry:
                continue
            canon = _canon_entry(entry, input_ops, inputs)
            label_map[entry["label"]] = canon
            entry["label"] = canon
            if "node" in entry and entry["node"] in input_ops:
                input_name = input_ops[entry["node"]]
                entry.pop("node")
                entry.pop("path", None)
                entry["data"] = {
                    "type": "list",
                    "items": list(inputs.get(input_name, [])),
                }
            deduped[canon] = entry
        bucket[field] = sorted(deduped.values(), key=lambda e: e["label"])

    if op.get("_op") == "FormatOp":
        kwargs = op.get("format_kwargs")
        if isinstance(kwargs, dict):
            canon_kwargs: dict[str, str] = {}
            for key, val in kwargs.items():
                if isinstance(val, str) and val in input_ops:
                    items = inputs.get(input_ops[val], [])
                    canon_kwargs[key] = f"INPUT:{json.dumps(items, sort_keys=True)}"
                else:
                    canon_kwargs[key] = val
            op["format_kwargs"] = canon_kwargs

    for container_key, field in (
        ("data_spec", "template"),
        (None, "rowwise_template"),
        ("data_spec", "description"),
    ):
        bucket = op.get(container_key) if container_key else op
        if not isinstance(bucket, dict):
            continue
        text = bucket.get(field)
        if not isinstance(text, str) or not label_map:
            continue
        bucket[field] = _replace_placeholders(text, label_map)

    return op


def _replace_placeholders(text: str, label_map: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        return "{" + label_map.get(name, name) + "}"

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, text)


def _non_input_fields(op_dict: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in op_dict.items() if k != "_inputs"}


_PAIRS = _collect_pairs()


@pytest.mark.parametrize(
    ("n8n_path", "yaml_path"),
    _PAIRS,
    ids=[n.stem for n, _ in _PAIRS],
)
def test_n8n_and_yaml_parsers_produce_equivalent_graphs(
    n8n_path: Path, yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For each (n8n, yaml) pair, assert the compiled graphs are semantically
    equivalent.

    Label naming (``ref_<auto>`` vs author-chosen ``symbol``) and InputOp
    encoding (inline ``data: {items: ...}`` vs ``node:`` ref) are
    canonicalized to a common form before comparison. All other fields —
    op set, wiring, model config, structural_outputs, system messages,
    template text — must match exactly.
    """
    _ensure_envs(monkeypatch)

    workflow = json.loads(n8n_path.read_text())
    inputs = _build_inputs_from_n8n(workflow)
    n8n_specs = parse_n8n_payload(
        {"graphs": [{"name": n8n_path.stem, "workflow": workflow, "inputs": inputs}]}
    )
    yaml_specs = parse_yaml_payload(yaml_path.read_text())

    assert set(n8n_specs) == set(
        yaml_specs
    ), f"graph names differ: n8n={sorted(n8n_specs)} yaml={sorted(yaml_specs)}"
    (graph_name,) = n8n_specs.keys()

    n8n_graph = n8n_specs[graph_name]["graph"]
    yaml_graph = yaml_specs[graph_name]["graph"]

    assert set(n8n_graph) == set(yaml_graph), (
        f"_id sets differ for {graph_name}:\n"
        f"  only in n8n:  {sorted(set(n8n_graph) - set(yaml_graph))}\n"
        f"  only in yaml: {sorted(set(yaml_graph) - set(n8n_graph))}"
    )

    yaml_inputs = yaml_specs[graph_name]["inputs"]
    n8n_inputs = n8n_specs[graph_name]["inputs"]
    for op_id in sorted(n8n_graph):
        n8n_op = _canonicalize_op(n8n_graph[op_id], n8n_graph, n8n_inputs)
        yaml_op = _canonicalize_op(yaml_graph[op_id], yaml_graph, yaml_inputs)
        assert _non_input_fields(n8n_op) == _non_input_fields(
            yaml_op
        ), f"op {op_id!r} non-input fields differ:\n  n8n:  {n8n_op}\n  yaml: {yaml_op}"
        assert set(n8n_op["_inputs"]) == set(yaml_op["_inputs"]), (
            f"op {op_id!r} _inputs sets differ:\n"
            f"  n8n:  {n8n_op['_inputs']}\n"
            f"  yaml: {yaml_op['_inputs']}"
        )


def test_equivalence_pair_discovery_found_templates() -> None:
    assert _PAIRS, (
        f"no (n8n, yaml) template pairs found under {TEMPLATE_ROOT}; "
        "add at least one matching pair before relying on the equivalence suite"
    )
