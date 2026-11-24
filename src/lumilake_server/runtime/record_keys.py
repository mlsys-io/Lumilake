"""Record keys: how one op's rows are addressed.

An expand child is ``(parent key, child index)``; a reduce group, a join row
and a composite read their key through `key_text`.
"""

import json
from collections.abc import Sequence
from typing import Any

from lumilake_server.data_specs import JoinDescription
from lumilake_server.runtime.record_store import is_marker, key_text

CHILD_KEY_SEPARATOR = "\x1f"
"""Separates a parent record key from its child index in an expand key."""


def value_at(row: Any, path: str) -> Any:
    """The value a dotted field path names in one row, or ``None`` when the
    row lacks it or the path reaches a marker.

    A row that is JSON text is decoded first."""
    current = row
    for part in [part for part in path.split(".") if part]:
        if is_marker(current):
            return None
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except ValueError:
                return None
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return None if is_marker(current) else current


def composite_key(row: Any, paths: Sequence[str]) -> str | None:
    """The key a row carries at a composite of paths, or ``None`` when any of
    them does not resolve: one path's value, or the list of their values."""
    values = [value_at(row, path) for path in paths]
    if not values or any(found is None for found in values):
        return None
    return key_text(values[0] if len(values) == 1 else values)


def child_key(parent_key: str, child: int) -> str:
    """An expand child's record key: its parent's key and its index among that
    parent's children."""
    return f"{parent_key}{CHILD_KEY_SEPARATOR}{child}"


def join_key_paths(join: JoinDescription) -> list[str]:
    """The paths a join's rows are keyed by: its ``by``, one path or a composite."""
    return list(join.by)


def join_row_key_paths(join: JoinDescription) -> list[str]:
    """The paths that keep a broadcast join's rows apart, or nothing without one."""
    return list(join.row_key or [])
