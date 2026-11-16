"""YAML emitter for the HTTP fetch op."""

from typing import Any

from lumilake_server.ops.http_fetch_ops import (
    HttpFetchConfig,
    validate_fetch_request,
)

from .common import EmitContext, validate_op_config


def emit_http_fetch_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    user_id_to_internal = ctx.user_id_to_internal
    url = entry.fields.get("url")
    if not isinstance(url, str):
        raise ValueError(f"HttpFetchOp '{entry.id}' requires a non-empty 'url'")
    query_ref = entry.fields.get("query")
    if not isinstance(query_ref, str) or query_ref not in user_id_to_internal:
        raise ValueError(
            f"HttpFetchOp '{entry.id}' requires a 'query' reference to an op or input"
        )
    config = entry.fields.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError(f"HttpFetchOp '{entry.id}' config must be a mapping")
    method = str(entry.fields.get("method", "GET")).upper()
    params = entry.fields.get("params")
    body = entry.fields.get("body")
    validate_fetch_request(
        url.strip(), method, params, body, where=f"HttpFetchOp '{entry.id}'"
    )
    validate_op_config(HttpFetchConfig, config, entry.id, "HttpFetchOp")
    op_dict["url"] = url
    op_dict["method"] = method
    op_dict["query"] = user_id_to_internal[query_ref]
    op_dict["config"] = config
    if params is not None:
        op_dict["params"] = params
    if body is not None:
        op_dict["body"] = body
    op_dict["_inputs"] = [user_id_to_internal[query_ref]]
