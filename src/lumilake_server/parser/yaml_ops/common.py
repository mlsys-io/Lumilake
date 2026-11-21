"""Shared types for the YAML op emitters, which fill in one entry's op-specific
fields and may queue implicit upstream ops."""

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

from lumilake_server.common import GenerationConfig
from lumilake_server.parser import keyed_seed

_GENERATION_CONFIG_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(GenerationConfig)
)


@dataclass
class OpEntry:
    id: str
    op: str
    inputs: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    max_iter: int | None = None


@dataclass
class EmitContext:
    entry: OpEntry
    user_id_to_internal: dict[str, str]
    scope: str
    graph_ops: dict[str, dict[str, Any]]
    job_seed: int = 0
    round_number: int | None = None
    template_ids: dict[str, str] = field(default_factory=dict)


Emitter = Callable[[dict[str, Any], EmitContext], None]


@dataclass(frozen=True)
class OpEmitter:
    prefix: str
    emit: Emitter


def validate_op_config(
    config_cls: type[Any], config: dict[str, Any], entry_id: str, op_kind: str
) -> None:
    """Run the op's own config validation here, where the user-facing op id is
    known: the op constructor raises the same errors without an id."""
    unknown = set(config) - {f.name for f in fields(config_cls)}
    if unknown:
        raise ValueError(
            f"{op_kind} '{entry_id}' config has unknown fields: {sorted(unknown)}"
        )
    try:
        config_cls(**config)
    except ValueError as exc:
        detail = str(exc).removeprefix(f"{op_kind} ")
        raise ValueError(f"{op_kind} '{entry_id}' {detail}") from exc


def build_generation_config(
    config: dict[str, Any], entry_id: str, op_kind: str = "LLMChatOp"
) -> dict[str, Any]:
    unknown = set(config) - _GENERATION_CONFIG_FIELDS
    if unknown:
        raise ValueError(
            f"{op_kind} '{entry_id}' config has unknown fields: {sorted(unknown)}. "
            "Use 'extra_sampling_params' for keys not on GenerationConfig."
        )
    return dict(config)


def seed_input_id(op_id: str, node_ref: str, ctx: EmitContext, op_kind: str) -> str:
    """The internal id of a node a keyed seed reads, after checking the
    producer is built first."""
    if node_ref not in ctx.user_id_to_internal:
        raise ValueError(f"{op_kind} '{op_id}' seed references unknown id '{node_ref}'")
    node_id = ctx.user_id_to_internal[node_ref]
    producer = ctx.graph_ops.get(node_id)
    is_workflow_input = producer is not None and producer.get("_op") == "InputOp"
    if node_ref not in ctx.entry.inputs and not is_workflow_input:
        raise ValueError(
            f"{op_kind} '{op_id}' seed reads '{node_ref}', which is"
            " not listed in the op's inputs; add it there so the seed's"
            " producer is built before it"
        )
    return node_id


def emit_keyed_seed(
    op_dict: dict[str, Any],
    ctx: EmitContext,
    seed: dict[str, Any],
    field: str,
    op_kind: str,
) -> None:
    """Derive each row's seed from its key: queue a row function over the
    key column and bind the op that draws with it to its output."""
    entry = ctx.entry
    node_ref, path = keyed_seed.key_target(
        entry.id,
        seed,
        op_kind,
        entry.fields.get("rowwise_template"),
        entry.fields.get("config", {}),
    )
    node_id = seed_input_id(entry.id, node_ref, ctx, op_kind)
    seed_dict = keyed_seed.seed_op(
        entry.id,
        node_id,
        path,
        ctx.job_seed,
        ctx.round_number,
        op_kind,
        template_id=ctx.template_ids.get(entry.id, entry.id),
        scope=ctx.scope,
        pairing=keyed_seed.key_pairing(seed, entry.id, op_kind),
    )
    op_dict.setdefault("__implicit_ops__", []).append(seed_dict)
    op_dict[field] = {"node": seed_dict["_id"], "path": "items.output"}
    if seed_dict["_id"] not in op_dict["_inputs"]:
        op_dict["_inputs"].append(seed_dict["_id"])


def keyed_seeds(
    spec: dict[str, Any], inputs: dict[str, list[str]]
) -> list[tuple[str, str, str, str, bool]]:
    """Every key form a workflow declares whose key column the workflow
    supplies, as ``(op id, op kind, node, path, shared)``, so the submission
    can check them for duplicate keys over the whole input set before it splits
    that set into slices. A column is the workflow's when it is an input or a
    literal data op, which the caller resolves against ``inputs``; any other
    producer's rows are not known until it runs."""
    declared: list[tuple[str, str, str, str, bool]] = []
    ops = spec.get("ops")
    if not isinstance(ops, list):
        return declared
    for op in ops:
        if not isinstance(op, dict):
            continue
        op_kind = op.get("op")
        if op_kind not in ("LLMChatOp", "LambdaOp"):
            continue
        field = "seed" if op_kind == "LambdaOp" else "rowwise_seed"
        seed = op.get(field)
        if not isinstance(seed, dict):
            continue
        key = seed.get("key")
        node = key.get("node") if isinstance(key, dict) else None
        path = key.get("path") if isinstance(key, dict) else None
        if not isinstance(node, str) or not isinstance(path, str):
            continue
        shared = seed.get("pairing") == "shared"
        declared.append((str(op.get("id")), op_kind, node, path, shared))
    return declared
