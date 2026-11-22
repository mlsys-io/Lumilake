"""n8n Postgres/S3 retrieval nodes: table, query-template and S3-key helpers
plus the three retrieval translators.
"""

import re
from typing import Any

from ..common import make_id as _make_id
from . import references, shared
from .entries import NodeDict, OpContext

_BINARY_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".pdf",
)


def _get_param_value(value: Any, field: str, node_name: str) -> str:
    if isinstance(value, dict) and "value" in value:
        value = value.get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"n8n postgres node '{node_name}' missing {field}")
    return value.strip()


def _normalize_limit(
    value: Any, node_name: str, op_ids: dict[str, str]
) -> tuple[int | str | None, list[str]]:
    if value is None:
        return None, []
    if isinstance(value, bool):
        raise ValueError(f"n8n postgres node '{node_name}' limit must be an integer")
    if isinstance(value, int):
        return value, []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, []
        resolved = shared.resolve_value(stripped, op_ids)
        if isinstance(resolved, str) and resolved.startswith(shared.REF_TOKEN_PREFIX):
            ref_node_id, _ = shared.decode_ref_token(resolved)
            return resolved, [ref_node_id]
        if isinstance(resolved, str) and resolved.isdigit():
            return int(resolved), []
        if stripped.isdigit():
            return int(stripped), []
    raise ValueError(f"n8n postgres node '{node_name}' limit must be an integer")


def _parse_postgres_filters(
    where: Any, op_ids: dict[str, str], node_name: str
) -> tuple[list[list[list[Any]]], list[str]]:
    if where in (None, {}):
        return _normalize_filters([], op_ids, 1)
    if not isinstance(where, dict):
        raise ValueError(f"n8n postgres node '{node_name}' where must be an object")
    values = where.get("values") or []
    if not isinstance(values, list):
        raise ValueError(f"n8n postgres node '{node_name}' where.values must be a list")
    raw_filters: list[list[Any]] = []
    for entry in values:
        if not isinstance(entry, dict):
            raise ValueError(
                f"n8n postgres node '{node_name}' where.values entries must be objects"
            )
        column = entry.get("column")
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                f"n8n postgres node '{node_name}' where.values missing column"
            )
        op = (
            entry.get("operation")
            or entry.get("operator")
            or entry.get("condition")
            or "="
        )
        value = entry.get("value")
        if isinstance(value, str):
            value = shared.strip_expr(value)
        raw_filters.append([column, op, value])
    return _normalize_filters(raw_filters, op_ids, 1)


def _normalize_filters(
    filters_raw: Any, op_ids: dict[str, str], table_count: int
) -> tuple[list[list[list[Any]]], list[str]]:
    if filters_raw is None or filters_raw == []:
        return [[] for _ in range(table_count or 1)], []

    if not isinstance(filters_raw, list):
        raise ValueError("n8n filters must be a list")

    filters = filters_raw
    if filters and isinstance(filters[0], (list, tuple)) and len(filters[0]) == 3:
        filters = [filters]

    normalized: list[list[list[Any]]] = []
    extra_inputs: list[str] = []
    for table_filters in filters:
        if not isinstance(table_filters, list):
            raise ValueError("n8n filters must be list of lists")
        table_list: list[list[Any]] = []
        for item in table_filters:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("n8n filter entries must be [column, op, value]")
            col, op, val = item
            resolved_val = shared.resolve_value(val, op_ids)
            if shared.is_op_id(resolved_val):
                extra_inputs.append(resolved_val)
            elif isinstance(resolved_val, str) and resolved_val.startswith(
                shared.REF_TOKEN_PREFIX
            ):
                ref_node_id, _ = shared.decode_ref_token(resolved_val)
                extra_inputs.append(ref_node_id)
            table_list.append([col, op, resolved_val])
        normalized.append(table_list)

    if table_count and len(normalized) != table_count:
        raise ValueError("filters length must match tables length")

    return normalized, extra_inputs


def _build_sql_template_and_params(
    table: str,
    filters: list[list[Any]],
    locked_columns: list[str],
    limit: int | str | None,
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
    node_map: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    select_cols = (
        ", ".join(shared.quote_column_ref(col) for col in locked_columns)
        if locked_columns
        else "*"
    )
    query = f"SELECT {select_cols} FROM {shared.quote_table_ref(table)}"
    query_params: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
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
        query_params.append({"label": label, **payload})

    where_clauses: list[str] = []
    for raw_filter in filters:
        if not isinstance(raw_filter, (list, tuple)) or len(raw_filter) != 3:
            continue
        col, op, val = raw_filter
        clause_label = shared.sql_param_label(str(col), str(op))
        placeholder = f"{{{clause_label}}}"
        resolved_val = val

        if isinstance(resolved_val, str) and resolved_val.startswith(
            shared.REF_TOKEN_PREFIX
        ):
            node_id, path = shared.decode_ref_token(resolved_val)
            add_param(clause_label, {"node": node_id, "path": path})
            where_clauses.append(
                f"{shared.quote_column_ref(str(col))} {op} '{placeholder}'"
            )
            continue
        if isinstance(resolved_val, str) and shared.is_op_id(resolved_val):
            input_name = input_names_by_op_id.get(resolved_val)
            if input_name:
                add_param(
                    clause_label,
                    {"data": {"type": "list", "items": inputs[input_name]}},
                )
            else:
                add_param(
                    clause_label,
                    {
                        "node": resolved_val,
                        "path": references.items_path(
                            op_type_by_id.get(resolved_val), None
                        ),
                    },
                )
            where_clauses.append(
                f"{shared.quote_column_ref(str(col))} {op} '{placeholder}'"
            )
            continue
        if isinstance(resolved_val, (list, tuple)):
            values = ", ".join(shared.sql_literal(item) for item in resolved_val)
            where_clauses.append(f"{shared.quote_column_ref(str(col))} {op} ({values})")
            continue

        where_clauses.append(
            f"{shared.quote_column_ref(str(col))} {op}"
            f" {shared.sql_literal(resolved_val)}"
        )

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    if isinstance(limit, int):
        query += f" LIMIT {limit}"
    elif isinstance(limit, str) and limit.startswith(shared.REF_TOKEN_PREFIX):
        node_id, path = shared.decode_ref_token(limit)
        add_param("limit", {"node": node_id, "path": path})
        query += " LIMIT {limit}"
    elif isinstance(limit, str) and shared.is_op_id(limit):
        input_name = input_names_by_op_id.get(limit)
        if input_name:
            add_param("limit", {"data": {"type": "list", "items": inputs[input_name]}})
        else:
            add_param(
                "limit",
                {
                    "node": limit,
                    "path": references.items_path(op_type_by_id.get(limit), None),
                },
            )
        query += " LIMIT {limit}"
    elif limit is not None:
        query += f" LIMIT {shared.sql_literal(limit)}"

    return query, query_params


def _sql_data_spec(
    template: str,
    params: list[dict[str, Any]],
    notes: dict[str, Any],
    table: str | None = None,
) -> dict[str, Any]:
    """The retrieval spec for an n8n postgres node, with its ``rows``
    declaration when the notes set one."""
    data_spec: dict[str, Any] = {
        "type": "lumid",
        "mode": "sql",
        "template": template,
        "params": params,
    }
    if table is not None:
        data_spec["table"] = table
    if notes.get("rows") is not None:
        data_spec["rows"] = notes["rows"]
    return data_spec


def make_postgres(
    node: NodeDict, ctx: OpContext, node_name: str
) -> tuple[list[NodeDict], str]:
    params = node.get("parameters", {})
    if not params or (len(params) == 0):
        raise ValueError(
            f"n8n postgres node '{node_name}' has empty parameters; "
            "schema, table, and operation are required"
        )
    operation = params.get("operation")
    main_inputs = [
        ctx.op_ids[dep]
        for dep in references.main_incoming(node_name, ctx.incoming, ctx.node_map)
        if dep in ctx.op_ids
    ]
    notes = shared.parse_notes(node.get("notes", ""))

    if operation == "executeQuery":
        query = _get_param_value(params.get("query"), "query", node_name)
        sql_template, sql_params, extra_inputs = (
            shared.build_execute_query_template_and_params(
                query=query,
                op_ids=ctx.op_ids,
                inputs=ctx.inputs,
                node_map=ctx.node_map,
            )
        )
        dep_inputs = [
            param["node"]
            for param in sql_params
            if isinstance(param, dict) and isinstance(param.get("node"), str)
        ]
        query_op: dict[str, Any] = {
            "_id": _make_id(ctx.scope, "retrieval", node_name),
            "_op": "DataRetrievalOp",
            "_max_iter": None,
            "_inputs": list(dict.fromkeys(extra_inputs + dep_inputs + main_inputs)),
            "data_spec": _sql_data_spec(sql_template, sql_params, notes),
        }
        return [query_op], query_op["_id"]

    schema = _get_param_value(params.get("schema"), "schema", node_name)
    table = _get_param_value(params.get("table"), "table", node_name)
    full_table = f"{schema}.{table}"

    limit, limit_inputs = _normalize_limit(params.get("limit"), node_name, ctx.op_ids)
    filters, extra_inputs = _parse_postgres_filters(
        params.get("where"), ctx.op_ids, node_name
    )

    options = params.get("options") or {}
    output_columns = options.get("outputColumns")
    if output_columns and not isinstance(output_columns, list):
        raise ValueError(
            f"n8n postgres node '{node_name}' options.outputColumns must be a list"
        )
    if output_columns and not all(isinstance(col, str) for col in output_columns):
        raise ValueError(
            f"n8n postgres node '{node_name}' options.outputColumns must be strings"
        )
    locked_columns = list(output_columns) if isinstance(output_columns, list) else []
    filter_columns = [str(f[0]) for f in (filters[0] if filters else []) if f]
    for col in filter_columns:
        if col not in locked_columns:
            locked_columns.append(col)
    sql_template, sql_params = _build_sql_template_and_params(
        table=full_table,
        filters=filters[0] if filters else [],
        locked_columns=locked_columns,
        limit=limit,
        op_ids=ctx.op_ids,
        inputs=ctx.inputs,
        node_map=ctx.node_map,
    )

    dep_inputs = [
        param["node"]
        for param in sql_params
        if isinstance(param, dict) and isinstance(param.get("node"), str)
    ]
    op: dict[str, Any] = {
        "_id": _make_id(ctx.scope, "retrieval", node_name),
        "_op": "DataRetrievalOp",
        "_max_iter": None,
        "_inputs": list(
            dict.fromkeys(extra_inputs + limit_inputs + dep_inputs + main_inputs)
        ),
        "data_spec": _sql_data_spec(sql_template, sql_params, notes, full_table),
    }
    return [op], op["_id"]


def is_lake_retrieval_node(node: dict[str, Any]) -> bool:
    return bool(node.get("parameters", {}).get("lakeRetrieval", False))


def make_s3(
    node: NodeDict, ctx: OpContext, node_name: str
) -> tuple[list[NodeDict], str]:
    if is_lake_retrieval_node(node):
        return make_lake_retrieval(node, ctx, node_name)
    params = node.get("parameters", {})
    if not params:
        raise ValueError(
            f"n8n s3 node '{node_name}' has empty parameters; "
            "bucketName and operation are required"
        )
    upstream_main = None
    for edge in ctx.incoming.get(node_name, []):
        if edge.get("type") == "main" and edge.get("node"):
            upstream_main = edge["node"]
            break
    if not upstream_main or upstream_main not in ctx.op_ids:
        raise ValueError(
            f"n8n s3 node '{node_name}' missing upstream SQL retrieval dependency"
        )
    s3_template, s3_params, param_deps = _build_s3_template_and_params(
        node=node,
        node_name=node_name,
        node_map=ctx.node_map,
        op_ids=ctx.op_ids,
        inputs=ctx.inputs,
    )
    data_spec: dict[str, Any] = {
        "type": "lumid",
        "mode": "s3",
        "template": s3_template,
        "params": s3_params,
        "encoding": _encoding_for(s3_template),
    }
    deps = [ctx.op_ids[upstream_main], *param_deps]
    op: dict[str, Any] = {
        "_id": _make_id(ctx.scope, "retrieval", node_name),
        "_op": "DataRetrievalOp",
        "_max_iter": None,
        "_inputs": list(dict.fromkeys(deps)),
        "data_spec": data_spec,
    }
    return [op], op["_id"]


def make_lake_retrieval(
    node: NodeDict, ctx: OpContext, node_name: str
) -> tuple[list[NodeDict], str]:
    params = node.get("parameters", {})
    options = params.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError(
            f"n8n lake retrieval node for '{node_name}' options must be an object"
        )
    upstream_main = None
    for edge in ctx.incoming.get(node_name, []):
        if edge.get("type") == "main" and edge.get("node"):
            upstream_main = edge["node"]
            break
    if not upstream_main or upstream_main not in ctx.op_ids:
        raise ValueError(
            f"n8n lake retrieval node '{node_name}' missing upstream SQL retrieval"
            " dependency"
        )
    s3_template, s3_params, param_deps = _build_s3_template_and_params(
        node=node,
        node_name=node_name,
        node_map=ctx.node_map,
        op_ids=ctx.op_ids,
        inputs=ctx.inputs,
    )
    data_spec: dict[str, Any] = {
        "type": "lumid",
        "mode": "s3",
        "template": s3_template,
        "params": s3_params,
        "encoding": _encoding_for(s3_template),
    }
    deps = [ctx.op_ids[upstream_main], *param_deps]

    op: dict[str, Any] = {
        "_id": _make_id(ctx.scope, "retrieval", node_name),
        "_op": "DataRetrievalOp",
        "_max_iter": None,
        "_inputs": list(dict.fromkeys(deps)),
        "data_spec": data_spec,
    }
    return [op], op["_id"]


def _encoding_for(s3_template: str) -> str:
    s3_lower = s3_template.lower()
    binary = any(
        s3_lower.endswith(ext) or ext + "}" in s3_lower for ext in _BINARY_EXTENSIONS
    )
    return "binary" if binary else "utf-8"


def _build_s3_template_and_params(
    node: dict[str, Any],
    node_name: str,
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    params = node.get("parameters", {})
    prefix = params.get("bucketName")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"n8n s3 node for '{node_name}' missing bucketName")
    options = params.get("options") or {}
    if not isinstance(options, dict):
        raise ValueError(f"n8n s3 node for '{node_name}' options must be an object")
    folder_key = options.get("folderKey")
    if not isinstance(folder_key, str) or not folder_key.strip():
        raise ValueError(f"n8n s3 node for '{node_name}' missing options.folderKey")

    key_template, key_params, param_deps = _parse_s3_folder_key(
        folder_key=folder_key,
        node_name=node_name,
        node_map=node_map,
        op_ids=op_ids,
        inputs=inputs,
    )
    normalized_prefix = prefix.strip().strip("/")
    normalized_key = key_template.strip().lstrip("/")
    full_template = (
        f"{normalized_prefix}/{normalized_key}" if normalized_key else normalized_prefix
    )
    return full_template, key_params, param_deps


def _parse_s3_folder_key(
    folder_key: str,
    node_name: str,
    node_map: dict[str, dict[str, Any]],
    op_ids: dict[str, str],
    inputs: dict[str, list[str]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    stripped = shared.strip_expr(folder_key).strip()
    stripped = re.sub(r"\$\s*\{\{", "{{", stripped)
    if not stripped:
        raise ValueError(f"n8n s3 node for '{node_name}' folderKey cannot be empty")

    refs = shared.extract_refs(stripped)
    if not refs:
        return stripped.lstrip("/"), [], []

    input_names_by_op_id = {
        op_id: name for name, op_id in op_ids.items() if name in inputs
    }
    template = stripped
    key_params: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    dependencies: list[str] = []

    for idx, (full_literal, ref_name) in enumerate(refs):
        if ref_name not in op_ids:
            raise ValueError(
                f"n8n s3 node for '{node_name}' folderKey references unknown node"
                f" '{ref_name}'"
            )
        detail_matches = shared.extract_ref_details(full_literal)
        ref_path = detail_matches[0][1] if detail_matches else None
        base_label = references.path_to_label(ref_path) or f"value{idx}"
        label = shared.ensure_unique_label(base_label, seen_labels)
        template = template.replace(full_literal, f"{{{label}}}", 1)

        node_id = op_ids[ref_name]
        input_name = input_names_by_op_id.get(node_id)
        if input_name:
            key_params.append(
                {
                    "label": label,
                    "data": {"type": "list", "items": inputs[input_name]},
                }
            )
            continue

        node_type = (node_map.get(ref_name) or {}).get("type")
        path = references.items_path(node_type, ref_path)
        key_params.append({"label": label, "node": node_id, "path": path})
        dependencies.append(node_id)

    return template.lstrip("/"), key_params, dependencies


def postgres_dependencies(node_name: str, ctx: OpContext) -> list[str]:
    node = ctx.node_map[node_name]
    deps: list[str] = []
    params = node.get("parameters", {})
    deps.extend(shared.extract_ref_names(params.get("limit")))
    where = params.get("where", {})
    if isinstance(where, dict):
        deps.extend(shared.extract_ref_names(where))
    deps.extend(shared.extract_ref_names(params.get("query")))
    upstream_main = references.find_main_upstream(node_name, ctx.incoming, ctx.node_map)
    if upstream_main:
        deps.append(upstream_main)
    return [dep for dep in deps if isinstance(dep, str)]


def s3_dependencies(node_name: str, ctx: OpContext) -> list[str]:
    node = ctx.node_map[node_name]
    deps: list[str] = []
    params = node.get("parameters", {})
    options = params.get("options")
    if isinstance(options, dict):
        deps.extend(shared.extract_ref_names(options.get("folderKey")))
    for edge in ctx.incoming.get(node_name, []):
        if edge.get("type") == "main" and edge.get("node"):
            deps.append(edge["node"])
    return [dep for dep in deps if isinstance(dep, str)]
