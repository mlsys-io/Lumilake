"""Shared types for the YAML op emitters, which fill in one entry's op-specific
fields and may queue implicit upstream ops."""

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

from lumilake_server.common import GenerationConfig

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
    key: dict[str, str] | None = None
    """A declared record key: the node carrying row keys and the path into it."""


@dataclass
class EmitContext:
    entry: OpEntry
    user_id_to_internal: dict[str, str]
    scope: str
    graph_ops: dict[str, dict[str, Any]]


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
