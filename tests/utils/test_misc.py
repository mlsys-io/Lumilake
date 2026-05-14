"""Tests for `filter_models_by_queries`."""

from collections.abc import Mapping

from pydantic import BaseModel
from starlette.datastructures import QueryParams

from lumilake_server.utils.misc import filter_models_by_queries


class Item(BaseModel):
    id: str
    status: str
    tags: list[str] = []
    last_seen_at: str | None = None
    enabled: bool = True
    metadata: dict[str, str] = {}


def _models() -> list[Item]:
    return [
        Item(
            id="k1",
            status="active",
            tags=["alpha"],
            enabled=True,
            last_seen_at="2026-04-26T00:00:00Z",
        ),
        Item(
            id="k2",
            status="archived",
            tags=["beta"],
            enabled=False,
            last_seen_at="2026-04-25T00:00:00Z",
        ),
        Item(
            id="k3",
            status="active",
            tags=["gamma"],
            last_seen_at=None,
            metadata={"region": "us-east-1"},
        ),
    ]


def test_no_query_returns_all() -> None:
    out = filter_models_by_queries(_models(), {})
    assert {m.id for m in out} == {"k1", "k2", "k3"}


def test_exact_match() -> None:
    out = filter_models_by_queries(_models(), {"status": "active"})
    assert {m.id for m in out} == {"k1", "k3"}


def test_repeated_keys_are_or() -> None:
    queries = QueryParams("status=active&status=archived")
    out = filter_models_by_queries(_models(), queries)
    assert {m.id for m in out} == {"k1", "k2", "k3"}


def test_list_field_membership() -> None:
    out = filter_models_by_queries(_models(), {"tags": "beta"})
    assert {m.id for m in out} == {"k2"}


def test_null_matching() -> None:
    out = filter_models_by_queries(_models(), {"last_seen_at": "null"})
    assert {m.id for m in out} == {"k3"}


def test_boolean_spellings() -> None:
    out = filter_models_by_queries(_models(), {"enabled": "false"})
    assert {m.id for m in out} == {"k2"}
    out = filter_models_by_queries(_models(), {"enabled": "1"})
    assert {m.id for m in out} == {"k1", "k3"}


def test_dot_notation_into_dict_field() -> None:
    # Models whose `metadata.region` traversal misses (k1, k2 -> empty dict)
    # have the key skipped, so they pass through. Only k3 has the field and
    # is exact-matched.
    out = filter_models_by_queries(_models(), {"metadata.region": "us-east-1"})
    assert {m.id for m in out} == {"k1", "k2", "k3"}
    out = filter_models_by_queries(_models(), {"metadata.region": "elsewhere"})
    assert {m.id for m in out} == {"k1", "k2"}


def test_unknown_key_is_ignored() -> None:
    out = filter_models_by_queries(_models(), {"no_such_field": "x"})
    assert {m.id for m in out} == {"k1", "k2", "k3"}


def test_works_with_plain_mapping() -> None:
    queries: Mapping = {"status": "archived"}
    out = filter_models_by_queries(_models(), queries)
    assert [m.id for m in out] == ["k2"]
