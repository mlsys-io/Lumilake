"""n8n wire-format constants and the helpers several node families share.

Anything specific to one family lives in that family's module; this module
holds only what more than one of them needs.
"""

import json
import re
from typing import Any

from . import references
from .constants import (
    QUERY_EXPR_PATTERN,
    REF_DETAIL_PATTERN,
    REF_PATTERN,
    REF_TOKEN_PREFIX,
)


def parse_notes(notes: Any) -> dict[str, Any]:
    if notes is None or notes == "":
        return {}
    if isinstance(notes, dict):
        return notes
    if not isinstance(notes, str):
        raise ValueError("n8n notes must be a JSON string or object")
    raw = notes.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in n8n notes: {exc}") from exc


def extract_refs(text: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for match in REF_PATTERN.finditer(text):
        full = match.group(0)
        name = match.group(1) or match.group(2)
        if name:
            refs.append((full, name))
    return refs


def extract_ref_details(text: str) -> list[tuple[str, str | None]]:
    refs: list[tuple[str, str | None]] = []
    for match in REF_DETAIL_PATTERN.finditer(text):
        name = match.group(1)
        path = match.group(2)
        if name:
            refs.append((name, path))
    return refs


def extract_ref_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, str):
        names.extend(name for _, name in extract_refs(value))
    elif isinstance(value, list):
        for item in value:
            names.extend(extract_ref_names(item))
    elif isinstance(value, dict):
        for item in value.values():
            names.extend(extract_ref_names(item))
    return names


def strip_expr(value: str) -> str:
    if isinstance(value, str) and value.startswith("="):
        return value[1:]
    return value


def is_op_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(
        (
            "input_",
            "llm_",
            "llmdata_",
            "imagegen_",
            "llmvision_",
            "format_",
            "data_",
            "lambda_",
            "retrieval_",
        )
    )


def encode_ref_token(node_id: str, path: str) -> str:
    return f"{REF_TOKEN_PREFIX}{node_id}:{path}"


def decode_ref_token(token: str) -> tuple[str, str]:
    payload = token[len(REF_TOKEN_PREFIX) :]
    node_id, _, path = payload.partition(":")
    if not node_id or not path:
        raise ValueError(f"Invalid reference token: {token}")
    return node_id, path


def resolve_value(value: Any, op_ids: dict[str, str]) -> Any:
    if isinstance(value, str):
        refs = extract_ref_details(strip_expr(value))
        if refs:
            name, path = refs[0]
            if name in op_ids:
                op_id = op_ids[name]
                runtime_path = references.to_runtime_output_path(path)
                if runtime_path:
                    return encode_ref_token(op_id, runtime_path)
                return op_id
            raise ValueError(f"Unknown reference '{name}' in filters")
    return value


def ensure_unique_label(label: str, seen: set[str]) -> str:
    if label not in seen:
        seen.add(label)
        return label
    idx = 2
    while f"{label}_{idx}" in seen:
        idx += 1
    unique = f"{label}_{idx}"
    seen.add(unique)
    return unique


def resolve_input_key(node_name: str, inputs: dict[str, list[str]]) -> str:
    """Return the matching key in *inputs* for *node_name* (case-insensitive)."""
    if node_name in inputs:
        return node_name
    lower = node_name.lower()
    for key in inputs:
        if key.lower() == lower:
            return key
    raise ValueError(f"Missing input for '{node_name}'")


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def quote_identifier(ident: str) -> str:
    stripped = ident.strip()
    if not stripped:
        return stripped
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped
    if any(ch.isupper() for ch in stripped):
        return f'"{stripped}"'
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", stripped):
        return stripped
    return f'"{stripped}"'


def quote_table_ref(table: str) -> str:
    parts = [part.strip() for part in table.split(".")]
    return ".".join(quote_identifier(part) for part in parts if part)


def quote_column_ref(column: str) -> str:
    parts = [part.strip() for part in column.split(".")]
    return ".".join(quote_identifier(part) for part in parts if part)


def sql_param_label(column: str, operator: str) -> str:
    clean_column = re.sub(r"[^A-Za-z0-9_]", "", column) or "value"
    op = operator.strip()
    if op in {">", ">="}:
        return f"{clean_column}Min"
    if op in {"<", "<="}:
        return f"{clean_column}Max"
    return clean_column


def next_sql_label(base: str, seen_labels: set[str]) -> str:
    label = re.sub(r"[^A-Za-z0-9_]", "_", base).strip("_") or "param"
    if label[0].isdigit():
        label = f"p_{label}"
    if label not in seen_labels:
        return label
    idx = 2
    while f"{label}_{idx}" in seen_labels:
        idx += 1
    return f"{label}_{idx}"


def build_execute_query_template_and_params(
    query: str,
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
    node_map: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    template = strip_expr(query).strip()
    params: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    dependencies: list[str] = []
    input_names_by_op_id = {
        op_id: name for name, op_id in op_ids.items() if name in inputs
    }
    op_type_by_id = {
        op_id: (node_map.get(name) or {}).get("type") for name, op_id in op_ids.items()
    }

    def add_param(label: str, payload: dict[str, Any]) -> None:
        if label in seen_labels:
            return
        seen_labels.add(label)
        params.append({"label": label, **payload})

    def replacement(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if not expr:
            raise ValueError("n8n postgres executeQuery contains empty expression")
        resolved = resolve_value(expr, op_ids)
        if isinstance(resolved, str) and resolved.startswith(REF_TOKEN_PREFIX):
            node_id, path = decode_ref_token(resolved)
            label = next_sql_label(f"ref_{node_id}", seen_labels)
            add_param(label, {"node": node_id, "path": path})
            dependencies.append(node_id)
            return f"{{{label}}}"
        if isinstance(resolved, str) and is_op_id(resolved):
            label = next_sql_label(f"ref_{resolved}", seen_labels)
            input_name = input_names_by_op_id.get(resolved)
            if input_name:
                add_param(
                    label, {"data": {"type": "list", "items": inputs[input_name]}}
                )
            else:
                add_param(
                    label,
                    {
                        "node": resolved,
                        "path": references.items_path(
                            op_type_by_id.get(resolved), None
                        ),
                    },
                )
                dependencies.append(resolved)
            return f"{{{label}}}"
        return sql_literal(resolved)

    templated = QUERY_EXPR_PATTERN.sub(replacement, template)
    return templated, params, list(dict.fromkeys(dependencies))
