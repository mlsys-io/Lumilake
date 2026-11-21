"""Deriving a per-row seed from a row key.

``rowwise_seed: {key: {node, path}}`` names a column that holds each row's key;
the seed is then a pseudo-random function of the job seed, the op's id, the
round and that key, so it does not depend on batch position or worker. The
parser turns the key form into a generated row-mode ``LambdaOp`` that reads the
key column and returns one seed per row, and binds the op that draws with it to
its output. A row or expand ``LambdaOp`` takes the same key form under ``seed``
and receives the derived seed as its second argument.

Two rows of one op that carry the same key draw the same seed. That turns two
independent draws into one shared draw without saying so, so the key form is
refused unless it declares ``pairing: shared``; the refusal names the op and the
two rows, and it is checked over the whole input set at submit.
"""

import json
from typing import Any

from lumilake_server.common import GenerationConfig
from lumilake_server.ops.llm_ops import RowwiseSeed, check_rowwise_seed
from lumilake_server.parser.common import make_id

SEED_SUFFIX = "__seed"
SEED_FN_NAME = "seed_for_row"
_OUTPUT = "output"
SEED_MODULUS = 2**31
PAIRINGS = ("unique", "shared")

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
    """The field names a key path drills below ``items``: the whole output is
    the value itself, so it drills nothing. A forwarded reference binds to a
    flat input, whose whole value is the key."""
    if not path:
        return []
    if not path.startswith("items."):
        raise ValueError(f"key path must start with 'items.', got '{path}'")
    parts = [part for part in path[len("items.") :].split(".") if part]
    if parts[:1] == [_OUTPUT]:
        parts = parts[1:]
    return parts


def key_cells(values: list[str], parts: list[str]) -> list[Any]:
    """The key each row of a literal or forwarded key column carries, read the
    way the generated row function reads a cell: JSON object or array text is
    parsed, and any other cell is the key as it stands. A row with no key or a
    null key is the generated function's refusal, not this one's."""
    return [_decode_cell(value, parts) for value in values]


def _decode_cell(value: Any, parts: list[str]) -> Any:
    """One row's key out of its raw column cell, or ``None`` when the cell
    carries no key to compare."""
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
    """Refuse two rows of one key form that carry the same key unless the form
    declares shared pairing, naming the op and the first such row pair."""
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
    """The id of the row function generated for one op. The parser names it
    with the ``seed`` prefix, which no op type emits, so no user op can hold
    that id; a native graph, whose op ids are already internal, names it from
    the op's own id."""
    if scope is None:
        return f"{chat_op_id}{SEED_SUFFIX}"
    return make_id(scope, "seed", chat_op_id)


def seed_source(
    fn_name: str,
    job_seed: int,
    template_id: str,
    round_number: int | None,
    parts: list[str],
    *,
    op_kind: str = "LLMChatOp",
    shared: bool = False,
) -> str:
    """The source of a keyed-seed row function: it reads one row's key out of
    its input column and returns the seed that row draws with. The job seed,
    template id, round and key path are constants of the generated code, so the
    body never runs user code, and the hash is the one ``docs/OPS.md``
    documents."""
    named = f"{op_kind} '{template_id}'"
    return _SEED_FN.format(
        fn_name=fn_name,
        parts=parts,
        missing=_seed_message(
            f"{named} has no row key in the value with path"
            f" {'.'.join(parts) or '<the value itself>'}:"
        ),
        null=_seed_message(
            f"{named} has a null row key in"
            f" {'.'.join(parts) or '<the value itself>'}:"
        ),
        duplicate=_duplicate_message(named),
        unique=not shared,
        job_seed=job_seed,
        template_id=template_id,
        round_number=round_number,
        modulus=SEED_MODULUS,
    )


def _seed_message(message: str) -> str:
    """One runtime message, with its value placeholder appended and any ``%``
    in the baked op id or path escaped, so the function fills in just the
    value."""
    return message.replace("%", "%%") + " %s"


def _duplicate_message(named: str) -> str:
    """The refusal for a second row carrying a key an earlier row of the same
    task already carried; it names the op and both row positions in the task."""
    return (
        named.replace("%", "%%")
        + " derives one seed per key, but rows %r and %r of this task share a"
        " key; declare pairing: shared if the same key should draw the same seed"
    )


def key_target(
    op_id: str,
    seed: dict[str, Any],
    *,
    op_kind: str = "LLMChatOp",
    rowwise_template: str | None = None,
    config: Any = None,
) -> tuple[str, str]:
    """The ``(node, path)`` a key form reads, after checking the form fits this
    op, with the op named in the refusal. ``rowwise_template`` applies to a chat
    op only; a Lambda draws once per row by construction."""
    key = seed.get("key")
    if not isinstance(key, dict):
        raise ValueError(f"{op_kind} '{op_id}' seed key must be a mapping")
    beside = sorted(set(seed) - {"key", "pairing"})
    if beside:
        raise ValueError(
            f"{op_kind} '{op_id}' seed combines a key with {beside};"
            " the key form derives each row's seed, so the node and path it"
            " reads must sit inside 'key'"
        )
    key_pairing(seed, op_id, op_kind)
    node_ref = key.get("node")
    if not isinstance(node_ref, str):
        raise ValueError(f"{op_kind} '{op_id}' seed key must name a node")
    path = key.get("path")
    if not isinstance(path, str):
        raise ValueError(f"{op_kind} '{op_id}' seed key must name a path")
    if op_kind != "LLMChatOp":
        return node_ref, path
    if not isinstance(config, dict):
        raise ValueError(f"{op_kind} '{op_id}' config must be a mapping")
    try:
        generation_config = GenerationConfig(**config)
    except TypeError as exc:
        raise ValueError(f"{op_kind} '{op_id}' config is not valid: {exc}") from exc
    check_rowwise_seed(
        RowwiseSeed(node_ref, path), rowwise_template, generation_config, op_id
    )
    return node_ref, path


def key_pairing(seed: dict[str, Any], op_id: str, op_kind: str = "LLMChatOp") -> str:
    """The key form's pairing, which is ``unique`` unless declared shared."""
    pairing = seed.get("pairing", PAIRINGS[0])
    if pairing not in PAIRINGS:
        raise ValueError(
            f"{op_kind} '{op_id}' seed pairing must be one of {PAIRINGS},"
            f" got {pairing!r}"
        )
    return pairing


def seed_op(
    op_id: str,
    key_node: str,
    key_path: str,
    *,
    job_seed: int,
    round_number: int | None,
    template_id: str | None = None,
    scope: str | None = None,
    op_kind: str = "LLMChatOp",
    pairing: str = PAIRINGS[0],
) -> dict[str, Any]:
    """The row-mode ``LambdaOp`` that returns each row's seed from its key.
    ``template_id`` is the op id as the user wrote it; it defaults to the
    internal id. ``op_kind`` names the op that draws in the refusals."""
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
            op_kind=op_kind,
            shared=pairing == "shared",
        ),
        "mode": "row",
    }


def rewrite_native_keys(
    graph: dict[str, dict[str, Any]], *, job_seed: int
) -> dict[str, dict[str, Any]]:
    """Rewrite every key-form seed of a serialized native graph, adding each
    generated seed op and binding the op that draws with it to the seed op's
    output. The graph is built in insertion order, so a generated op is placed
    just ahead of the op that reads it."""
    rewritten: dict[str, dict[str, Any]] = {}
    for op_id, op in graph.items():
        field = "seed" if op.get("_op") == "LambdaOp" else "rowwise_seed"
        seed = op.get(field)
        if not isinstance(seed, dict) or "key" not in seed:
            rewritten[op_id] = op
            continue
        op_kind = op["_op"]
        node_ref, path = key_target(
            op_id,
            seed,
            op_kind=op_kind,
            rowwise_template=op.get("rowwise_template"),
            config=op.get("config", {}),
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
            job_seed=job_seed,
            round_number=None,
            op_kind=op_kind,
            pairing=key_pairing(seed, op_id, op_kind),
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
