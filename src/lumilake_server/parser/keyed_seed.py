"""Deriving a per-row seed from a row key, through a generated row LambdaOp."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from lumilake_server.common import GenerationConfig
from lumilake_server.ops.llm_ops import RowwiseSeed, check_rowwise_seed
from lumilake_server.ops.seed_spec import (
    PAIRINGS,
    ColumnSeed,
    KeySeed,
    Pairing,
)
from lumilake_server.ops.spec_validation import validate_fields
from lumilake_server.ops.util_ops import UNDECODED_COLUMN_KEY
from lumilake_server.parser.common import make_id

SEED_SUFFIX = "__seed"
SEED_FN_NAME = "seed_for_row"
SAMPLE_SUFFIX = "__sample"
SAMPLE_FN_NAME = "samples_for_row"
SAMPLE_SEED_LABEL = "__seed"
_OUTPUT = "output"
SEED_MODULUS = 2**31


def keyed_seed(
    job_seed: int, template_id: str, round_number: int | None, key: Any
) -> int:
    """The seed the generated row function draws for one ``(op, round, key)``."""
    payload = json.dumps(
        [job_seed, template_id, round_number, key],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % SEED_MODULUS


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


class NodeColumn(BaseModel):
    """A rowwise column entry read from a node's output at an optional path."""

    model_config = ConfigDict(extra="allow")

    label: str
    node: str
    path: str | None = None


_SAMPLE_COLUMNS: TypeAdapter[list[NodeColumn]] = TypeAdapter(list[NodeColumn])


def read_sample_columns(op_id: str, columns: Any) -> list[NodeColumn]:
    """The fan-out's node columns, refusing a literal column or the seed label."""
    if not columns:
        raise ValueError(f"LLMChatOp '{op_id}' config.n > 1 needs rowwise_columns")
    try:
        validated = _SAMPLE_COLUMNS.validate_python(columns)
    except ValidationError as exc:
        error = exc.errors(include_url=False)[0]
        where = "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error["loc"]
        )
        message = error["msg"].removeprefix("Value error, ")
        raise ValueError(
            f"LLMChatOp '{op_id}' config.n > 1 needs node columns;"
            f" rowwise_columns{where}: {message}"
        ) from exc
    for column in validated:
        if column.label == SAMPLE_SEED_LABEL:
            raise ValueError(
                f"LLMChatOp '{op_id}' config.n > 1 labels a rowwise column"
                f" '{SAMPLE_SEED_LABEL}', which the fan-out writes its sample seed to;"
                " rename that column"
            )
    return validated


_PRELUDE = """\
import hashlib
import json

_SEEN = {{}}
"""


_HELPERS = """

def _drill(cell, parts, missing):
    value = cell
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = value
        if isinstance(decoded, (dict, list)):
            value = decoded
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError(missing % (cell,))
    return value


def _cell(cell, parts, missing):
    if not parts:
        return cell
    return _drill(cell, parts, missing)


def _unique(value, duplicate):
    identity = json.dumps(value, sort_keys=True, separators=(",", ":"))
    seen = _SEEN.get(identity)
    if seen is not None:
        raise ValueError(duplicate % (seen, len(_SEEN)))
    _SEEN[identity] = len(_SEEN)


def _prf(job_seed, template_id, round_number, value, modulus):
    payload = json.dumps(
        [job_seed, template_id, round_number, value],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus
"""


_SEED_FN = _PRELUDE + """\
_PARTS = {parts!r}
_MISSING = {missing!r}
_NULL = {null!r}
_DUPLICATE = {duplicate!r}
_UNIQUE = {unique!r}
_JOB_SEED = {job_seed!r}
_TEMPLATE_ID = {template_id!r}
_ROUND = {round_number!r}
_MODULUS = {modulus}


def {fn_name}(row):
    value = _drill(row[0], _PARTS, _MISSING)
    if value is None:
        raise ValueError(_NULL % (row[0],))
    if _UNIQUE:
        _unique(value, _DUPLICATE)
    return _prf(_JOB_SEED, _TEMPLATE_ID, _ROUND, value, _MODULUS)
""" + _HELPERS


_SAMPLE_FN = _PRELUDE + """\
_COLUMNS = {columns!r}
_KEY_SLOT = {key_slot}
_KEY_PARTS = {key_parts!r}
_KEY_MISSING = {key_missing!r}
_KEY_NULL = {key_null!r}
_SEED_LABEL = {seed_label!r}
_SAMPLES = {samples}
_JOB_SEED = {job_seed!r}
_TEMPLATE_ID = {template_id!r}
_ROUND = {round_number!r}
_SHARED = {shared!r}
_DUPLICATE = {duplicate!r}
_MODULUS = {modulus}


def {fn_name}(row):
    key = _drill(row[_KEY_SLOT], _KEY_PARTS, _KEY_MISSING)
    if key is None:
        raise ValueError(_KEY_NULL % (row[_KEY_SLOT],))
    if not _SHARED:
        _unique(key, _DUPLICATE)
    values = {{
        label: _cell(row[slot], parts, missing)
        for label, slot, parts, missing in _COLUMNS
    }}
    items = []
    for sample in range(_SAMPLES):
        value = values.copy()
        value[_SEED_LABEL] = _prf(
            _JOB_SEED, _TEMPLATE_ID, _ROUND, [key, sample], _MODULUS
        )
        items.append(value)
    return items
""" + _HELPERS


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
                f"{op_kind} '{op_id}' derives one seed per key, but rows {first} and"
                f" {index} share the key {identity}; declare 'pairing: shared' if the"
                " same key should draw the same seed"
            )
        seen[identity] = index


def seed_op_id(chat_op_id: str, scope: str | None = None) -> str:
    """The id of the row function generated for one op, held by no user op."""
    if scope is None:
        return f"{chat_op_id}{SEED_SUFFIX}"
    return make_id(scope, "seed", chat_op_id)


def sample_op_id(chat_op_id: str, scope: str | None = None) -> str:
    """The id of the fan-out function generated for one op, held by no user op."""
    if scope is None:
        return f"{chat_op_id}{SAMPLE_SUFFIX}"
    return make_id(scope, "sample", chat_op_id)


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
        + " derives one seed per key, but rows %r and %r of this task share a key;"
        " declare pairing: shared if the same key should draw the same seed"
    )


def key_target(
    op_id: str,
    spec: KeySeed,
    op_kind: str,
    rowwise_template: str | None,
    config: Any,
) -> tuple[str, str, Pairing, int]:
    """The ``(node, path)`` a key form reads, its pairing, and its samples per row."""
    column = spec.key
    samples = 1
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
        samples = generation_config.n or 1
    return column.node, column.path, spec.pairing, samples


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


def sample_source(
    fn_name: str,
    columns: list[tuple[str, int, list[str]]],
    key_slot: int,
    key_path: str,
    samples: int,
    job_seed: int,
    template_id: str,
    round_number: int | None,
    shared: bool,
    where: str,
) -> str:
    """The source of a fan-out function: one row becomes one row per sample."""
    parts = key_parts(key_path)
    key_at = ".".join(parts) or "<the value itself>"
    return _SAMPLE_FN.format(
        fn_name=fn_name,
        columns=[
            (label, slot, path, _seed_message(f"{where} has no value for {label!r}:"))
            for label, slot, path in columns
        ],
        key_slot=key_slot,
        key_parts=parts,
        key_missing=_seed_message(f"{where} has no row key at {key_at}:"),
        key_null=_seed_message(f"{where} has a null row key at {key_at}:"),
        seed_label=SAMPLE_SEED_LABEL,
        samples=samples,
        job_seed=job_seed,
        template_id=template_id,
        round_number=round_number,
        shared=shared,
        duplicate=_duplicate_message(where),
        modulus=SEED_MODULUS,
    )


@dataclass(frozen=True)
class FanOut:
    """The key, sample count and op identity one fan-out is built for."""

    op_id: str
    op_kind: str
    key_node: str
    key_path: str
    samples: int
    job_seed: int
    round_number: int | None
    template_id: str
    scope: str | None
    pairing: Pairing


def sample_op(
    spec: FanOut,
    columns: list[tuple[str, int, list[str]]],
    key_slot: int,
) -> dict[str, Any]:
    """The expand-mode ``LambdaOp`` that fans each row out into one per sample."""
    return {
        "_id": sample_op_id(spec.op_id, spec.scope),
        "_op": "LambdaOp",
        "_max_iter": None,
        "_inputs": [],
        "fn_name": SAMPLE_FN_NAME,
        "_code": sample_source(
            SAMPLE_FN_NAME,
            columns,
            key_slot,
            spec.key_path,
            spec.samples,
            spec.job_seed,
            spec.template_id,
            spec.round_number,
            spec.pairing == "shared",
            f"{spec.op_kind} '{spec.op_id}'",
        ),
        "mode": "expand",
    }


def rewire_sample_fan_out(
    op: dict[str, Any], spec: FanOut, field: str
) -> dict[str, Any]:
    """Point a chat op's columns and seed at a fan-out over its rows, one per sample."""
    op_id = spec.op_id
    columns = read_sample_columns(op_id, op.get("rowwise_columns"))
    fan_out_inputs = _fan_out_inputs(columns)
    whole_nodes = [column.node for column in columns if not _column_parts(column.path)]
    if spec.key_node not in fan_out_inputs:
        raise ValueError(
            f"{spec.op_kind} '{op_id}' keyed seed reads '{spec.key_node}'; no column"
            " binds it, so the fan-out cannot key each sample"
        )
    slots: list[tuple[str, int, list[str]]] = [
        (
            column.label,
            fan_out_inputs.index(column.node),
            _column_parts(column.path),
        )
        for column in columns
    ]
    seed_dict = sample_op(spec, slots, fan_out_inputs.index(spec.key_node))
    seed_dict["_inputs"] = fan_out_inputs
    if whole_nodes:
        seed_dict[UNDECODED_COLUMN_KEY] = list(dict.fromkeys(whole_nodes))
    fan_out = seed_dict["_id"]
    op["rowwise_columns"] = [
        column.model_copy(
            update={"path": f"items.output.{column.label}", "node": fan_out}
        ).model_dump(exclude_none=True)
        for column in columns
    ]
    # The fan-out already holds the n samples, so each request draws one.
    config = op["config"].copy()
    config.pop("n", None)
    op["config"] = config
    op["_inputs"] = [
        node for node in op.get("_inputs", []) if node not in fan_out_inputs
    ]
    op[field] = {"node": fan_out, "path": f"items.output.{SAMPLE_SEED_LABEL}"}
    if fan_out not in op["_inputs"]:
        op["_inputs"].append(fan_out)
    return seed_dict


def _column_parts(path: str | None) -> list[str]:
    """The fields a rowwise column's declared read path drills into its value."""
    if not path:
        return []
    parts = [part for part in path.split(".") if part]
    if parts[:1] == ["items"]:
        parts = parts[1:]
    if parts[:1] == [_OUTPUT]:
        parts = parts[1:]
    return parts


def _fan_out_inputs(columns: list[NodeColumn]) -> list[str]:
    """The distinct node ids a fan-out reads its column values from."""
    node_ids: list[str] = []
    for column in columns:
        if column.node not in node_ids:
            node_ids.append(column.node)
    return node_ids


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
        node_ref, path, pairing, samples = key_target(
            op_id,
            spec,
            op_kind,
            op.get("rowwise_template"),
            op.get("config", {}),
        )
        if node_ref not in op.get("_inputs", []):
            raise ValueError(
                f"{op_kind} '{op_id}' seed reads '{node_ref}', which is not listed in"
                " the op's inputs; add it there so the seed's producer is built before"
                " it"
            )
        if samples > 1:
            bound = {**op}
            generated = rewire_sample_fan_out(
                bound,
                FanOut(
                    op_id=op_id,
                    op_kind=op_kind,
                    key_node=node_ref,
                    key_path=path,
                    samples=samples,
                    job_seed=job_seed,
                    round_number=None,
                    template_id=op_id,
                    scope=None,
                    pairing=pairing,
                ),
                field,
            )
        else:
            generated = seed_op(
                op_id,
                node_ref,
                path,
                job_seed,
                None,
                op_kind,
                pairing=pairing,
            )
            bound = {
                **op,
                field: {"node": generated["_id"], "path": "items.output"},
                "_inputs": [*op.get("_inputs", []), generated["_id"]],
            }
        if generated["_id"] in graph:
            raise ValueError(
                f"{op_kind} '{op_id}' keyed seed generates the id '{generated['_id']}',"
                " which the graph already holds; rename that op so the generated row"
                " function has its own id"
            )
        rewritten[generated["_id"]] = generated
        rewritten[op_id] = bound
    return rewritten
