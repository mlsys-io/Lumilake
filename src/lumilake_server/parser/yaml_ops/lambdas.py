"""YAML emitter for the Lambda op."""

from typing import Any

from pydantic import ValidationError

from lumilake_server.data_specs import GpuRequest
from lumilake_server.ops.util_ops import LambdaMode, build_lambda_spec, check_seed_mode
from lumilake_server.parser.keyed_seed import read_key_seed

from .common import EmitContext, emit_keyed_seed


def emit_lambda_op(op_dict: dict[str, Any], ctx: EmitContext) -> None:
    entry = ctx.entry
    seed = entry.fields.get("seed")
    key_seed = (
        None if seed is None else read_key_seed(entry.id, "LambdaOp", "seed", seed)
    )
    gpu = entry.fields.get("gpu")
    if gpu is not None:
        try:
            gpu = GpuRequest.model_validate(gpu).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise ValueError(f"LambdaOp '{entry.id}' 'gpu' is invalid: {exc}") from exc
    spec = build_lambda_spec(
        {
            "fn_name": entry.fields.get("fn_name"),
            "code": entry.fields.get("code") or entry.fields.get("_code"),
            "mode": entry.fields.get("mode", LambdaMode.ROW),
            "by": entry.fields.get("by"),
            "timeout_s": entry.fields.get("timeout_s"),
            "memory_mb": entry.fields.get("memory_mb"),
            "image": entry.fields.get("image"),
            "gpu": gpu,
            "expected_runtime_s": entry.fields.get("expected_runtime_s"),
            "cache": entry.fields.get("cache"),
            "on_fail": entry.fields.get("on_fail"),
            "on_empty": entry.fields.get("on_empty"),
        },
        f"LambdaOp '{entry.id}'",
    )
    if seed is not None:
        check_seed_mode(spec.mode, f"LambdaOp '{entry.id}'")
    op_dict.update(spec.model_dump(by_alias=True, exclude_none=True))
    if key_seed is not None:
        emit_keyed_seed(op_dict, ctx, key_seed, "seed", "LambdaOp")
