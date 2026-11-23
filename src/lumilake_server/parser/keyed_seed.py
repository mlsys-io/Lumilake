"""Deriving a per-row seed from a row key, through a generated row LambdaOp."""

import json
from typing import Any

from lumilake_server.common import GenerationConfig
from lumilake_server.ops.llm_ops import RowwiseSeed, check_rowwise_seed
from lumilake_server.ops.seed_spec import (
    PAIRINGS,
    ColumnSeed,
    KeySeed,
    Pairing,
)
from lumilake_server.ops.spec_validation import validate_fields
from lumilake_server.parser.common import make_id

SEED_SUFFIX = "__seed"
SEED_FN_NAME = "seed_for_row"
_OUTPUT = "output"
SEED_MODULUS = 2**31


def read_key_seed(op_id: str, op_kind: str, field: str, entry: Any) -> KeySeed:
    """One op entry's key form, naming the op and field in the error."""
    return validate_fields(KeySeed, entry, f"{op_kind} '{op_id}' {field}")


def read_seed_spec(
    op_id: str, op_kind: str, field: str, entry: Any
) -> ColumnSeed | KeySeed:
    """One op entry's seed in either form, naming the op and field in the error."""
    if isinstance(entry, dict) and "key" in entry:
        return read_key_seed(op_id, op_kind, field, entry)
    return validate_fields(ColumnSeed, entry, f"{op_kind} '{op_id}' {field}")


_SEED_FN = """\
import hashlib
import json

_SEEN = {{}}


def {fn_name}(row):
    value = row[0]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = value
        if isinstance(decoded, (dict, list)):
            value = decoded
    for part in {parts!r}:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError({missing!r} % (row[0],))
    if value is None:
        raise ValueError({null!r} % (row[0],))
    if {unique!r}:
        identity = json.dumps(value, sort_keys=True, separators=(",", ":"))
        seen = _SEEN.get(identity)
        if seen is not None:
            raise ValueError({duplicate!r} % (seen, len(_SEEN)))
        _SEEN[identity] = len(_SEEN)
    payload = json.dumps(
        [{job_seed!r}, {template_id!r}, {round_number!r}, value],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % {modulus}
"""


def key_parts(path: str) -> list[str]:
    """The field names a key path drills below ``items``."""
    if not path:
        return []
    if not path.startswith("items."):
        raise ValueError(f"key path must start with 'items.', got '{path}'")
    parts = [part for part in path[len("items.") :].split(".") if part]
    if parts[:1] == [_OUTPUT]:
        parts = parts[1:]
    return parts


def key_cells(values: list[str], parts: list[str]) -> list[Any]:
    """The key each row of a literal or forwarded key column carries."""
    return [_decode_cell(value, parts) for value in values]


def _decode_cell(value: Any, parts: list[str]) -> Any:
    """One row's key out of its raw cell, or ``None`` with no key to compare."""
    for index, part in enumerate(parts):
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except ValueError:
                return None
            value = decoded if isinstance(decoded, (dict, list)) else value
            if index and isinstance(value, str):
                return None
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def check_unique(op_id: str, op_kind: str, keys: list[Any], pairing: str) -> None:
    """Refuse a repeated key unless the form declares shared pairing."""
    if pairing == "shared":
        return
    seen: dict[str, int] = {}
    for index, key in enumerate(keys):
        if key is None:
            continue
        identity = json.dumps(key, sort_keys=True, separators=(",", ":"))
        first = seen.get(identity)
        if first is not None:
            raise ValueError(
                f"{op_kind} '{op_id}' derives one seed per key, but rows"
                f" {first} and {index} share the key {identity}; declare"
                " 'pairing: shared' if the same key should draw the same seed"
            )
        seen[identity] = index


def seed_op_id(chat_op_id: str, scope: str | None = None) -> str:
    """The id of the row function generated for one op, held by no user op."""
    if scope is None:
        return f"{chat_op_id}{SEED_SUFFIX}"
    return make_id(scope, "seed", chat_op_id)


def seed_source(
    fn_name: str,
    job_seed: int,
    template_id: str,
    round_number: int | None,
    parts: list[str],
    op_kind: str,
    shared: bool,
) -> str:
    """The source of a keyed-seed row function, which returns one row's seed."""
    named = f"{op_kind} '{template_id}'"
    columns = ".".join(parts) or "<the value itself>"
    return _SEED_FN.format(
        fn_name=fn_name,
        parts=parts,
        missing=_seed_message(
            f"{named} has no row key in the value with path {columns}:"
        ),
        null=_seed_message(f"{named} has a null row key in {columns}:"),
        duplicate=_duplicate_message(named),
        unique=not shared,
        job_seed=job_seed,
        template_id=template_id,
        round_number=round_number,
        modulus=SEED_MODULUS,
    )


def _seed_message(message: str) -> str:
    """One runtime message with its value placeholder appended."""
    return message.replace("%", "%%") + " %s"


def _duplicate_message(named: str) -> str:
    """The refusal for a repeated key inside one task, naming both rows."""
    return (
        named.replace("%", "%%")
        + " derives one seed per key, but rows %r and %r of this task share a"
        " key; declare pairing: shared if the same key should draw the same seed"
    )


def key_target(
    op_id: str,
    spec: KeySeed,
    op_kind: str,
    rowwise_template: str | None,
    config: Any,
) -> tuple[str, str, Pairing]:
    """The ``(node, path)`` a key form reads and its pairing, checked for this op."""
    column = spec.key
    if op_kind != "LambdaOp":
        if not isinstance(config, dict):
            raise ValueError(f"{op_kind} '{op_id}' config must be a mapping")
        try:
            generation_config = GenerationConfig(**config)
        except TypeError as exc:
            raise ValueError(f"{op_kind} '{op_id}' config is not valid: {exc}") from exc
        check_rowwise_seed(
            RowwiseSeed(column.node, column.path),
            rowwise_template,
            generation_config,
            op_id,
        )
    return column.node, column.path, spec.pairing


def seed_op(
    op_id: str,
    key_node: str,
    key_path: str,
    job_seed: int,
    round_number: int | None,
    op_kind: str,
    template_id: str | None = None,
    scope: str | None = None,
    pairing: Pairing = PAIRINGS[0],
) -> dict[str, Any]:
    """The row-mode ``LambdaOp`` that returns each row's seed from its key."""
    return {
        "_id": seed_op_id(op_id, scope),
        "_op": "LambdaOp",
        "_max_iter": None,
        "_inputs": [key_node],
        "fn_name": SEED_FN_NAME,
        "_code": seed_source(
            SEED_FN_NAME,
            job_seed,
            op_id if template_id is None else template_id,
            round_number,
            key_parts(key_path),
            op_kind,
            pairing == "shared",
        ),
        "mode": "row",
    }


def rewrite_native_keys(
    graph: dict[str, dict[str, Any]], *, job_seed: int
) -> dict[str, dict[str, Any]]:
    """Rewrite every key-form seed of a native graph in place."""
    rewritten: dict[str, dict[str, Any]] = {}
    for op_id, op in graph.items():
        # Native payloads reach here unvalidated, so an op may name no kind.
        op_kind = op.get("_op")
        field = "seed" if op_kind == "LambdaOp" else "rowwise_seed"
        seed = op.get(field)
        if (
            op_kind not in ("LLMChatOp", "LambdaOp")
            or not isinstance(seed, dict)
            or "key" not in seed
        ):
            rewritten[op_id] = op
            continue
        spec = read_key_seed(op_id, op_kind, field, seed)
        node_ref, path, pairing = key_target(
            op_id,
            spec,
            op_kind,
            op.get("rowwise_template"),
            op.get("config", {}),
        )
        if node_ref not in op.get("_inputs", []):
            raise ValueError(
                f"{op_kind} '{op_id}' seed reads '{node_ref}', which is not"
                " listed in the op's inputs; add it there so the seed's"
                " producer is built before it"
            )
        generated = seed_op(
            op_id,
            node_ref,
            path,
            job_seed,
            None,
            op_kind,
            pairing=pairing,
        )
        if generated["_id"] in graph:
            raise ValueError(
                f"{op_kind} '{op_id}' keyed seed generates the id"
                f" '{generated['_id']}', which the graph already holds; rename"
                " that op so the generated row function has its own id"
            )
        rewritten[generated["_id"]] = generated
        rewritten[op_id] = {
            **op,
            field: {"node": generated["_id"], "path": "items.output"},
            "_inputs": [*op.get("_inputs", []), generated["_id"]],
        }
    return rewritten
