"""A retrieval's declared record key: YAML -> graph -> runtime check."""

import json
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.utils.retrieval_key import key_problem

_WORKFLOW = """\
name: k
ops:
  - id: Cases
    op: DataRetrievalOp
    data_spec:
      type: lumid
      mode: {mode}
      template: {template}
{key}
outputs:
  - {{name: cases, ref: Cases}}
"""

_KEY = "      key: {column: %r}"


def _df(columns: dict[str, Any]) -> dict[str, Any]:
    """One retrieval item carrying a column table in its envelope."""
    return {"table": {"df": json.dumps(columns)}}


_TABLE = _df({"case_id": {"0": "a", "1": "b"}})


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "test-token")


def _spec(key: str | None = None, mode: str = "sql") -> dict[str, Any]:
    """One retrieval workflow's runtime data_spec, through parser and builder."""
    template = "SELECT case_id FROM cases" if mode == "sql" else "cases/{id}.txt"
    text = _WORKFLOW.format(
        mode=mode, template=template, key="" if key is None else _KEY % key
    )
    parsed = parse_yaml_payload(text)["k"]
    runtime = RuntimeGraphBuilder().build(
        Graph.from_json(parsed["graph"]).compile(**parsed["inputs"])
    )
    (node,) = runtime.nodes.values()
    return node.data_spec


@pytest.mark.parametrize(
    ("key", "spec_key"),
    [(None, None), ("case_id", {"column": "case_id"})],
)
def test_a_declared_key_reaches_the_retrieval_node(
    key: str | None, spec_key: dict[str, str] | None
) -> None:
    assert _spec(key).get("key") == spec_key


@pytest.mark.parametrize(
    ("items", "problem"),
    [
        ([_TABLE, _df({"case_id": {"0": "c"}})], None),
        ([_TABLE, _df({"case_id": {"0": "a"}})], "repeated at row 0"),
        ([{"table": {"case_id": {"0": None}}}], "null key"),
        ([{"table": {"other": {"0": "a"}}}], "no such column"),
    ],
)
def test_a_keyed_result_is_checked_when_it_arrives(
    items: list[dict[str, Any]], problem: str | None
) -> None:
    found = key_problem("Cases", items, "case_id")
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


@pytest.mark.parametrize(
    ("key", "mode", "message"),
    [("case_id", "s3", "mode 's3'"), ("", "sql", "non-empty string")],
)
def test_a_key_the_spec_cannot_carry_is_refused_at_submit(
    key: str, mode: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _spec(key, mode)
