"""The typed ``data_spec`` each task type carries, and its wire shape."""

import json
from pathlib import Path

import pytest

import lumilake_server.ops.ops as ops_module
from lumilake import envs
from lumilake_server.data_specs import parse_data_spec
from tests.support import golden_graphs

_GOLDEN = Path(__file__).resolve().parent / "fixtures" / "data_spec_golden.json"
_LUMID_URL = "http://lumid-data"
_LUMID_TOKEN = "test-token"


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", _LUMID_URL)
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", _LUMID_TOKEN)
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "test-token")
    monkeypatch.setattr(ops_module, "unique_id", golden_graphs.FixedIds.next)


@pytest.mark.parametrize("graph_name", sorted(golden_graphs.GRAPHS))
def test_built_data_specs_match_the_golden_wire(graph_name: str) -> None:
    """Every builder's spec matches the wire the original builders produced,
    and a python step's generated code matches byte for byte."""
    golden = json.loads(_GOLDEN.read_text())[graph_name]
    built = golden_graphs.build(graph_name)

    assert json.dumps(built, sort_keys=True) == json.dumps(golden, sort_keys=True)


def test_unknown_task_type_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown task type"):
        parse_data_spec("query", {"type": "list", "items": []})


def test_unknown_data_spec_key_is_refused() -> None:
    with pytest.raises(ValueError, match="invalid data_spec"):
        parse_data_spec("python", {"plan": [], "bogus": 1})
