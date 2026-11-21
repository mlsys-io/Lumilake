"""YAML emitter for the Lambda op."""

from typing import Any

from lumilake_server.ops.util_ops import LambdaOp

from .common import EmitContext


def emit_lambda_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    fn_name = entry.fields.get("fn_name")
    code = entry.fields.get("code") or entry.fields.get("_code")
    if not isinstance(fn_name, str):
        raise ValueError(f"LambdaOp '{entry.id}' requires 'fn_name: str'")
    if not isinstance(code, str):
        raise ValueError(f"LambdaOp '{entry.id}' requires 'code: str'")
    mode = entry.fields.get("mode", "row")
    if mode not in LambdaOp.LAMBDA_MODES:
        raise ValueError(
            f"LambdaOp '{entry.id}' mode must be one of {LambdaOp.LAMBDA_MODES} "
            f"(got {mode!r})"
        )
    op_dict["fn_name"] = fn_name
    op_dict["_code"] = code
    op_dict["mode"] = mode
    # Set by a router's expansion, which the parser marks on the branches it
    # generated; never a workflow field. The leading underscore keeps it
    # parser-owned through serialization.
    if ctx.clears_markers:
        op_dict["_clears_markers"] = True
    # Optional limits for the FlowMesh python task the op compiles to; bounds
    # are checked when the op is built.
    for key, kind in (("timeout_s", (int, float)), ("memory_mb", int)):
        value = entry.fields.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, kind):
            raise ValueError(f"LambdaOp '{entry.id}' '{key}' must be a number")
        op_dict[key] = value
    image = entry.fields.get("image")
    if image is not None:
        if not isinstance(image, str) or not image.strip():
            raise ValueError(
                f"LambdaOp '{entry.id}' 'image' must be a non-empty string"
            )
        op_dict["image"] = image
    expected_runtime_s = entry.fields.get("expected_runtime_s")
    if expected_runtime_s is not None:
        if isinstance(expected_runtime_s, bool) or not isinstance(
            expected_runtime_s, (int, float)
        ):
            raise ValueError(
                f"LambdaOp '{entry.id}' 'expected_runtime_s' must be a number"
            )
        op_dict["expected_runtime_s"] = expected_runtime_s
    cache = entry.fields.get("cache")
    if cache is not None:
        if not isinstance(cache, bool):
            raise ValueError(f"LambdaOp '{entry.id}' 'cache' must be a bool")
        op_dict["cache"] = cache
    by = entry.fields.get("by")
    if by is not None:
        op_dict["by"] = by
