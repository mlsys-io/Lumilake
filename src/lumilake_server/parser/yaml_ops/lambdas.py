"""YAML emitter for the Lambda op."""

from typing import Any

from lumilake_server.ops.util_ops import CLEARS_KEY, build_lambda_spec

from .common import EmitContext


def emit_lambda_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    spec = build_lambda_spec(
        {
            "fn_name": entry.fields.get("fn_name"),
            "code": entry.fields.get("code") or entry.fields.get("_code"),
            "mode": entry.fields.get("mode", "row"),
            "by": entry.fields.get("by"),
            "timeout_s": entry.fields.get("timeout_s"),
            "memory_mb": entry.fields.get("memory_mb"),
            "image": entry.fields.get("image"),
            "expected_runtime_s": entry.fields.get("expected_runtime_s"),
            "cache": entry.fields.get("cache"),
            "on_fail": entry.fields.get("on_fail"),
            "on_empty": entry.fields.get("on_empty"),
            # Set on the branches a router's expansion generated, never a
            # workflow field.
            CLEARS_KEY: True if ctx.clears_markers else None,
        },
        f"LambdaOp '{entry.id}'",
    )
    op_dict.update(spec.model_dump(by_alias=True, exclude_none=True))
