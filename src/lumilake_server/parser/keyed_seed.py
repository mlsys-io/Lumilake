"""Deriving a rowwise chat op's per-row seed from a row key.

The parser turns ``rowwise_seed: {key: ...}`` into a generated row-mode
``LambdaOp`` that returns one seed per row from the job seed, the op id, the
round and the row's key.
"""

from typing import Any

from lumilake_server.common import GenerationConfig
from lumilake_server.ops.llm_ops import RowwiseSeed, check_rowwise_seed
from lumilake_server.parser.common import make_id

SEED_SUFFIX = "__seed"
SEED_FN_NAME = "seed_for_row"
_OUTPUT = "output"
SEED_MODULUS = 2**31

_SEED_FN = """\
import hashlib
import json


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
    payload = json.dumps(
        [{job_seed!r}, {template_id!r}, {round_number!r}, value],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % {modulus}
"""


def key_parts(path: str) -> list[str]:
    """The field names a key path drills below ``items``; a whole-output path
    drills nothing."""
    if not path:
        return []
    if not path.startswith("items."):
        raise ValueError(f"key path must start with 'items.', got '{path}'")
    parts = [part for part in path[len("items.") :].split(".") if part]
    if parts[:1] == [_OUTPUT]:
        parts = parts[1:]
    return parts


def seed_op_id(chat_op_id: str, scope: str | None = None) -> str:
    """The id of the row function generated for one chat op, which no user op
    can hold."""
    if scope is None:
        return f"{chat_op_id}{SEED_SUFFIX}"
    return make_id(scope, "seed", chat_op_id)


def seed_source(
    fn_name: str,
    job_seed: int,
    template_id: str,
    round_number: int | None,
    parts: list[str],
) -> str:
    """The source of a keyed-seed row function: it reads one row's key and
    returns the seed that row draws with."""
    return _SEED_FN.format(
        fn_name=fn_name,
        parts=parts,
        missing=_seed_message(
            f"LLMChatOp '{template_id}' has no row key in the value with path"
            f" {'.'.join(parts) or '<the value itself>'}:"
        ),
        null=_seed_message(
            f"LLMChatOp '{template_id}' has a null row key in"
            f" {'.'.join(parts) or '<the value itself>'}:"
        ),
        job_seed=job_seed,
        template_id=template_id,
        round_number=round_number,
        modulus=SEED_MODULUS,
    )


def _seed_message(message: str) -> str:
    """A runtime message with its value placeholder appended, any ``%`` in the
    baked text escaped."""
    return message.replace("%", "%%") + " %s"


def key_target(
    op_id: str,
    seed: dict[str, Any],
    *,
    rowwise_template: str | None,
    config: Any,
) -> tuple[str, str]:
    """The ``(node, path)`` a key form reads, after checking the form fits
    this chat op, with the op named in the refusal."""
    key = seed.get("key")
    if not isinstance(key, dict):
        raise ValueError(f"LLMChatOp '{op_id}' rowwise_seed key must be a mapping")
    beside = sorted(set(seed) - {"key"})
    if beside:
        raise ValueError(
            f"LLMChatOp '{op_id}' rowwise_seed combines a key with {beside};"
            " the key form derives each row's seed, so the node and path it"
            " reads must sit inside 'key'"
        )
    node_ref = key.get("node")
    if not isinstance(node_ref, str):
        raise ValueError(f"LLMChatOp '{op_id}' rowwise_seed key must name a node")
    path = key.get("path")
    if not isinstance(path, str):
        raise ValueError(f"LLMChatOp '{op_id}' rowwise_seed key must name a path")
    if not isinstance(config, dict):
        raise ValueError(f"LLMChatOp '{op_id}' config must be a mapping")
    try:
        generation_config = GenerationConfig(**config)
    except TypeError as exc:
        raise ValueError(f"LLMChatOp '{op_id}' config is not valid: {exc}") from exc
    check_rowwise_seed(
        RowwiseSeed(node_ref, path), rowwise_template, generation_config, op_id
    )
    return node_ref, path


def seed_op(
    chat_op_id: str,
    key_node: str,
    key_path: str,
    *,
    job_seed: int,
    round_number: int | None,
    template_id: str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    """The row-mode ``LambdaOp`` that returns each row's seed from its key."""
    return {
        "_id": seed_op_id(chat_op_id, scope),
        "_op": "LambdaOp",
        "_max_iter": None,
        "_inputs": [key_node],
        "fn_name": SEED_FN_NAME,
        "_code": seed_source(
            SEED_FN_NAME,
            job_seed,
            chat_op_id if template_id is None else template_id,
            round_number,
            key_parts(key_path),
        ),
        "mode": "row",
    }


def rewrite_native_keys(
    graph: dict[str, dict[str, Any]], *, job_seed: int
) -> dict[str, dict[str, Any]]:
    """Add each generated seed op to a serialized native graph, just ahead of
    the chat op that reads it."""
    rewritten: dict[str, dict[str, Any]] = {}
    for op_id, op in graph.items():
        seed = op.get("rowwise_seed")
        if (
            op.get("_op") != "LLMChatOp"
            or not isinstance(seed, dict)
            or "key" not in seed
        ):
            rewritten[op_id] = op
            continue
        node_ref, path = key_target(
            op_id,
            seed,
            rowwise_template=op.get("rowwise_template"),
            config=op.get("config", {}),
        )
        if node_ref not in op.get("_inputs", []):
            raise ValueError(
                f"LLMChatOp '{op_id}' rowwise_seed reads '{node_ref}', which is"
                " not listed in the op's inputs; add it there so the seed's"
                " producer is built before the chat op"
            )
        generated = seed_op(op_id, node_ref, path, job_seed=job_seed, round_number=None)
        if generated["_id"] in graph:
            raise ValueError(
                f"LLMChatOp '{op_id}' keyed rowwise_seed generates the id"
                f" '{generated['_id']}', which the graph already holds; rename"
                " that op so the generated row function has its own id"
            )
        rewritten[generated["_id"]] = generated
        rewritten[op_id] = {
            **op,
            "rowwise_seed": {"node": generated["_id"], "path": "items.output"},
            "_inputs": [*op.get("_inputs", []), generated["_id"]],
        }
    return rewritten
