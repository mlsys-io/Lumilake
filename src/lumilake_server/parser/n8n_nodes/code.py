"""n8n code nodes and the ChatTrigger input node."""

import re
from typing import Any

from lumilake_server.ops.util_ops import build_lambda_spec

from ..common import make_id as _make_id
from . import references, shared
from .entries import NodeDict, OpContext


def make_code(
    node: NodeDict, ctx: OpContext, node_name: str
) -> tuple[list[NodeDict], str]:
    notes = shared.parse_notes(node.get("notes", ""))

    input_names = notes.get("inputs")
    if input_names is None or input_names == []:
        input_names = references.main_incoming(node_name, ctx.incoming, ctx.node_map)
    if not isinstance(input_names, list) or not input_names:
        raise ValueError(f"Code node '{node_name}' missing inputs")

    params = node.get("parameters", {})
    code = params.get("pythonCode")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"Code node '{node_name}' missing pythonCode")

    input_ids: list[str] = []
    for name in input_names:
        if name not in ctx.op_ids:
            raise ValueError(f"Code node '{node_name}' unknown input '{name}'")
        input_ids.append(ctx.op_ids[name])

    fn_name = _safe_fn_name(node_name)
    fn_code = _wrap_code(fn_name, code)
    spec = build_lambda_spec(
        {
            "fn_name": fn_name,
            "code": fn_code,
            "mode": notes.get("mode", "row"),
            "by": notes.get("by"),
            "image": notes.get("image"),
            "timeout_s": notes.get("timeout-s"),
            "memory_mb": notes.get("memory-mb"),
            "expected_runtime_s": notes.get("expected-runtime-s"),
            "cache": notes.get("cache"),
        },
        f"Code node '{node_name}'",
        names={"timeout_s": "timeout-s", "expected_runtime_s": "expected-runtime-s"},
    )
    op: dict[str, Any] = {
        "_id": _make_id(ctx.scope, "lambda", node_name),
        "_op": "LambdaOp",
        "_max_iter": None,
        "_inputs": input_ids,
    }
    op.update(spec.model_dump(by_alias=True, exclude_none=True))
    is_output = notes.get("is-output")
    if is_output is not None and not isinstance(is_output, bool):
        raise ValueError(f"Code node '{node_name}' notes 'is-output' must be a bool")
    output_field = notes.get("output-field")
    if output_field is not None and (
        not isinstance(output_field, str) or not output_field.strip()
    ):
        raise ValueError(
            f"Code node '{node_name}' notes 'output-field' must be a non-empty string"
        )
    sample_value = notes.get("sample-value")
    if sample_value is None:
        if "sample-value" in notes:
            raise ValueError(
                f"Code node '{node_name}' notes 'sample-value' must be a "
                "non-null JSON value"
            )
    else:
        op["sample_value"] = sample_value
    return [op], op["_id"]


def make_trigger(
    node: NodeDict, ctx: OpContext, node_name: str
) -> tuple[list[NodeDict], str]:
    matched_key = shared.resolve_input_key(node_name, ctx.inputs)
    op: dict[str, Any] = {
        "_id": _make_id(ctx.scope, "input", node_name),
        "_op": "InputOp",
        "_max_iter": None,
        "_inputs": [],
        "name": matched_key,
    }
    return [op], op["_id"]


def code_dependencies(node_name: str, ctx: OpContext) -> list[str]:
    node = ctx.node_map[node_name]
    notes = shared.parse_notes(node.get("notes", ""))
    deps = notes.get("inputs")
    if deps is None or deps == []:
        deps = references.main_incoming(node_name, ctx.incoming, ctx.node_map)
    return [dep for dep in deps if isinstance(dep, str)]


def trigger_dependencies(node_name: str, ctx: OpContext) -> list[str]:
    return []


def _safe_fn_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "lambda_fn"
    if slug[0].isdigit():
        slug = f"fn_{slug}"
    return slug


def _wrap_code(fn_name: str, code: str) -> str:
    stripped = code.strip()
    if stripped.startswith(("def ", "lambda ")):
        return stripped
    lines = stripped.splitlines() or [""]
    indented = "\n".join("    " + line for line in lines)
    return f"def {fn_name}(args):\n{indented}"
