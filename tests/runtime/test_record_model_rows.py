"""Per-record serving of a model op: its rows cut to the missed ones."""

import pytest

from lumilake_server.runtime.record_model_rows import cut_rows
from lumilake_server.runtime.record_store import (
    op_spec_digest,
)
from tests.support.data_specs import runtime_op

_CHAT = {"type": "dataframe", "messages": [{"role": "user", "content": "{q}"}]}
_COLUMN = {"label": "q", "data": {"type": "list", "items": ["a", "b", "c"]}}


@pytest.mark.parametrize(
    ("spec", "missed", "cut"),
    [
        ({**_CHAT, "columns": [_COLUMN], "seeds": [7, 8, 9]}, [0, 2], True),
        ({"type": "list", "items": ["only"]}, [1, 2], False),
        ({**_CHAT, "columns": [{"label": "q", "node": "N", "path": "x"}]}, [1], False),
    ],
)
def test_cut_rows_keeps_a_literal_list_at_the_missed_positions(
    spec: dict, missed: list[int], cut: bool
) -> None:
    """A column's rows and its seeds are cut to the missed positions; a one-value list
    and a node reference stay whole."""
    out = cut_rows(spec, missed)
    if not cut:
        assert out == spec
        return
    assert out["columns"][0]["data"]["items"] == ["a", "c"]
    assert out["seeds"] == [7, 9]
    task = "python" if "plan" in spec else "inference"
    assert op_spec_digest(runtime_op("n", task, "vllm", "m", out, {}, {}), {}) == (
        op_spec_digest(runtime_op("n", task, "vllm", "m", spec, {}, {}), {})
    )
