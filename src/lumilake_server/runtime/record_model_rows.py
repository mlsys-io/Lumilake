"""A model op's rows, cut to the missed records for a partial dispatch.

Every model family reads its rows from a data spec, never a python plan, so a partial
run narrows those rows in place: a literal list keeps the missed positions and a
one-value list stays to broadcast. Node references stay whole; a pass-through step
narrows those instead.
"""

from collections.abc import Mapping, Sequence
from typing import Any

_LIST_TYPE = "list"

MERGE_SUFFIX = "/merge"
PASS_SUFFIX = "/pass"


def merge_id(node_id: str) -> str:
    """The id of the merge step a partial run generates for ``node_id``."""
    return f"{node_id}{MERGE_SUFFIX}"


def pass_id(node_id: str, source: str) -> str:
    """The id of the pass-through step that narrows ``source`` for ``node_id``."""
    return f"{node_id}{PASS_SUFFIX}/{source}"


def cut_rows(spec: Mapping[str, Any], missed: Sequence[int]) -> dict[str, Any]:
    """A model op's data spec over the missed rows only.

    A literal ``type: list`` payload keeps the missed positions, and a literal
    top-level ``seeds`` list is cut beside its rows. A one-value list stays
    whole, so it broadcasts over the rows this run computes.
    """
    cut = _cut_side(dict(spec), missed)
    seeds = cut.get("seeds")
    if isinstance(seeds, list):
        cut["seeds"] = _cut_list(seeds, missed)
    return cut


def _cut_side(value: Any, missed: Sequence[int]) -> Any:
    """One data-spec side with each literal rows list cut to ``missed``."""
    if isinstance(value, Mapping):
        items = value.get("items")
        if value.get("type") == _LIST_TYPE and isinstance(items, list):
            return {**value, "items": _cut_list(items, missed)}
        return {key: _cut_side(item, missed) for key, item in value.items()}
    if isinstance(value, list):
        return [_cut_side(item, missed) for item in value]
    return value


def _cut_list(values: list[Any], missed: Sequence[int]) -> list[Any]:
    """A literal rows list at the missed positions; a one-value list stays."""
    if len(values) == 1:
        return values
    return [values[position] for position in missed]
