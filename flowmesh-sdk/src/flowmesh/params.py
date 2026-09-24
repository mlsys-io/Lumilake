"""Helpers for encoding repeated query parameters."""

from collections.abc import Iterable

type QueryValue = str | bool | int


def append_param(
    params: list[tuple[str, str]],
    key: str,
    value: QueryValue | None,
) -> None:
    """Append a single scalar query parameter if present."""
    if value is None:
        return
    if isinstance(value, bool):
        params.append((key, str(value).lower()))
        return
    params.append((key, str(value)))


def extend_params(
    params: list[tuple[str, str]],
    key: str,
    values: QueryValue | Iterable[QueryValue] | None,
) -> None:
    """Append one or more query parameters."""
    if values is None:
        return
    if isinstance(values, (str, bool, int)):
        append_param(params, key, values)
        return
    for value in values:
        append_param(params, key, value)
