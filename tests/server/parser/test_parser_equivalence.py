import json
from pathlib import Path
from typing import Any

import pytest

from lumilake import envs
from lumilake.server.parser import parse_n8n_payload, parse_yaml_payload
from lumilake.server.parser.n8n import N8N_CHAT_TRIGGER

TEMPLATE_ROOT = Path(__file__).resolve().parents[3] / "templates"
N8N_DIR = TEMPLATE_ROOT / "n8n"
YAML_DIR = TEMPLATE_ROOT / "yaml"


def _ensure_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "DATABASE_URL", "sqlite://")
    monkeypatch.setattr(envs, "S3_URL", "s3://dummy")
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "")


def _collect_pairs() -> list[tuple[Path, Path]]:
    """Discover (n8n.json, yaml.yaml) pairs by matching file stems."""
    pairs: list[tuple[Path, Path]] = []
    if not N8N_DIR.is_dir() or not YAML_DIR.is_dir():
        return pairs
    for n8n_path in sorted(N8N_DIR.glob("*.json")):
        yaml_path = YAML_DIR / f"{n8n_path.stem}.yaml"
        if yaml_path.exists():
            pairs.append((n8n_path, yaml_path))
    return pairs


def _build_inputs_from_n8n(workflow: dict[str, Any]) -> dict[str, list[str]]:
    """Build a minimal inputs dict: each ChatTrigger node name -> empty list.

    The equivalence test compares graph structure; concrete input values are
    irrelevant. Empty lists keep the payload validation path happy without
    constraining the YAML author to match any particular sample data.
    """
    inputs: dict[str, list[str]] = {}
    for node in workflow.get("nodes", []):
        if node.get("type") != N8N_CHAT_TRIGGER:
            continue
        name = node.get("name")
        if isinstance(name, str) and name:
            inputs[name] = []
    return inputs


def _non_input_fields(op_dict: dict[str, Any]) -> dict[str, Any]:
    """Return all op fields *except* `_inputs` (which is compared set-wise)."""
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
    """For each (n8n, yaml) pair, assert the compiled graph_specs match.

    Design decisions (fixed, see PR #30):
      * `_id` values MUST match between parsers for the same logical node.
        Both parsers derive ids via `_make_id(scope, prefix, name)`, so equal
        user-facing node names + equal graph scope + equal op types ==
        identical internal `_id`s within a single interpreter process.
      * `_inputs` is a list whose *order* is implementation-dependent; we
        compare as a set so rearrangements do not cause spurious failures.
      * All other op fields (including nested `messages`, `config`,
        `data_spec`, `format_kwargs`, etc.) must be exactly equal.
    """
    _ensure_envs(monkeypatch)

    workflow = json.loads(n8n_path.read_text())
    inputs = _build_inputs_from_n8n(workflow)
    n8n_specs = parse_n8n_payload(
        {"graphs": [{"name": n8n_path.stem, "workflow": workflow, "inputs": inputs}]}
    )
    yaml_specs = parse_yaml_payload(yaml_path.read_text())

    # Exactly one graph per template.
    assert set(n8n_specs) == set(
        yaml_specs
    ), f"graph names differ: n8n={sorted(n8n_specs)} yaml={sorted(yaml_specs)}"
    (graph_name,) = n8n_specs.keys()

    n8n_graph = n8n_specs[graph_name]["graph"]
    yaml_graph = yaml_specs[graph_name]["graph"]

    # 1) Set of `_id`s must match exactly.
    assert set(n8n_graph) == set(yaml_graph), (
        f"_id sets differ for {graph_name}:\n"
        f"  only in n8n:  {sorted(set(n8n_graph) - set(yaml_graph))}\n"
        f"  only in yaml: {sorted(set(yaml_graph) - set(n8n_graph))}"
    )

    # 2) Per-op comparison.
    for op_id in sorted(n8n_graph):
        n8n_op = n8n_graph[op_id]
        yaml_op = yaml_graph[op_id]
        assert _non_input_fields(n8n_op) == _non_input_fields(
            yaml_op
        ), f"op {op_id!r} non-input fields differ:\n  n8n:  {n8n_op}\n  yaml: {yaml_op}"
        assert set(n8n_op["_inputs"]) == set(yaml_op["_inputs"]), (
            f"op {op_id!r} _inputs sets differ:\n"
            f"  n8n:  {n8n_op['_inputs']}\n"
            f"  yaml: {yaml_op['_inputs']}"
        )


def test_equivalence_pair_discovery_found_templates() -> None:
    """Guard against silently passing when no pairs exist."""
    assert _PAIRS, (
        f"no (n8n, yaml) template pairs found under {TEMPLATE_ROOT}; "
        "add at least one matching pair before relying on the equivalence suite"
    )
