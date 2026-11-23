"""A retrieval's declared record key: YAML -> graph -> runtime check."""

import json
from typing import Any

import pytest

from lumilake import envs
from lumilake_server.data_specs import RetrievalData, RetrievalKey
from lumilake_server.graphs import Graph
from lumilake_server.parser import parse_yaml_payload
from lumilake_server.runtime.retrieval_key_check import key_problem
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder

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


def _df(columns: dict[str, Any]) -> dict[str, Any]:
    return {"table": {"df": json.dumps(columns)}}


@pytest.fixture(autouse=True)
def _lumid_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(envs, "LUMID_DATA_TOKEN", "t")


def _spec(key: str | None, mode: str = "sql") -> RetrievalData:
    template = "SELECT case_id FROM cases" if mode == "sql" else "cases/{id}.txt"
    line = "" if key is None else f"      key: {{column: {key!r}}}"
    text = _WORKFLOW.format(mode=mode, template=template, key=line)
    parsed = parse_yaml_payload(text)["k"]
    runtime = RuntimeGraphBuilder().build(
        Graph.from_json(parsed["graph"]).compile(**parsed["inputs"])
    )
    (node,) = runtime.nodes.values()
    spec = node.data_spec
    assert isinstance(spec, RetrievalData)
    return spec


def test_a_declared_key_reaches_the_retrieval_node() -> None:
    assert _spec(None).key is None
    assert _spec("case_id").key == RetrievalKey(column="case_id")


@pytest.mark.parametrize(
    ("items", "problem"),
    [
        ([_df({"case_id": {"0": "a"}}), _df({"case_id": {"0": "c"}})], None),
        (
            [_df({"case_id": {"0": "a", "1": "a"}})],
            "repeated at row 0 of item 0 and row 1 of item 0 of this batch",
        ),
        (
            [_df({"case_id": {"0": "a"}}), _df({"case_id": {"0": "a"}})],
            "repeated at row 0 of item 0 and row 0 of item 1 of this batch",
        ),
        ([_df({"case_id": {"0": None}})], "null key"),
        ([_df({"other": {"0": "a"}})], "no such column"),
        ([{"table": "not a table"}], "could not be read as a column table"),
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
