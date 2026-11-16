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


def require_str(spec: dict[str, Any], key: str, where: str) -> str:
    """A field's value, typed for the op whose constructor validates it."""
    value = spec.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{where}: requires the {key!r} field as a string")
    return value
