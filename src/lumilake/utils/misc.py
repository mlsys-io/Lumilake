"""HTTP-style query-param filtering for Pydantic model lists."""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

_MISSING = object()


def _query_items(queries: Mapping) -> list[tuple[str, str]]:
    multi_items = getattr(queries, "multi_items", None)
    if callable(multi_items):
        return list(multi_items())
    return list(queries.items())


def _get_nested_value(data: Any, key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _matches_query(model_value: Any, key: str, query_values: list[str]) -> bool:
    if model_value is None:
        return any(v in ("", "null", "None") for v in query_values)

    if isinstance(model_value, bool):
        normalized_model = "true" if model_value else "false"
        normalized_queries = {v.strip().lower() for v in query_values}
        truthy = {"1", "true", "yes", "on"}
        falsy = {"0", "false", "no", "off"}
        if normalized_model == "true":
            return bool(normalized_queries & truthy)
        return bool(normalized_queries & falsy)

    if isinstance(model_value, (list, tuple, set)):
        value_set = {str(v) for v in model_value}
        return any(v in value_set for v in query_values)

    if key == "tags" and isinstance(model_value, str):
        tag_set = {t.strip() for t in model_value.split(",") if t.strip()}
        return any(v in tag_set for v in query_values)

    return any(str(model_value) == v for v in query_values)


def filter_models_by_queries[T: BaseModel](
    models: list[T], queries: Mapping
) -> list[T]:
    """Filter Pydantic models by HTTP-style query parameters.

    Supported behaviors:

    - **Exact match (default)**: ``?status=IDLE`` matches when
      ``model.status == "IDLE"``.
    - **Repeated keys (OR semantics)**: ``?status=IDLE&status=BUSY`` matches when
      the field equals **any** provided value.
    - **Nested keys via dot-notation**: ``?env.region=us-east-1`` traverses
      dict-like fields. Traversal failure → key ignored for that model.
    - **List/set membership**: if the model field is a list/tuple/set, then a
      match occurs when any query value equals any element (stringified).
    - **Tag membership**: if the field is named ``tags`` and is a comma-
      separated string, ``?tags=gpu`` matches if ``"gpu"`` is one of the tags.
    - **Null-ish matching**: ``None`` matches query values ``""``, ``"null"``,
      or ``"None"``.
    - **Booleans**: query accepts ``true/false``, ``1/0``, ``yes/no``, ``on/off``.

    Unknown keys are ignored. Partial/substring/regex/numeric/case-insensitive
    matching is not supported.
    """
    query_map: dict[str, list[str]] = defaultdict(list)
    for key, value in _query_items(queries):
        query_map[str(key)].append(str(value))

    filtered: list[T] = []
    for model in models:
        model_dict = model.model_dump()
        match = True
        for key, values in query_map.items():
            model_value = _get_nested_value(model_dict, key)
            if model_value is _MISSING:
                continue
            if not _matches_query(model_value, key, values):
                match = False
                break
        if match:
            filtered.append(model)
    return filtered
