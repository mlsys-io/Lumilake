"""Deriving a rowwise chat op's per-row seed from a row key.

``rowwise_seed: {key: {node, path}}`` names a column that holds each row's key;
the seed is then a pseudo-random function of the job seed, the op's id, the
round and that key, so it does not depend on batch position or worker. The
parser turns the key form into a generated row-mode ``LambdaOp`` that reads the
key column and returns one seed per row, and binds the chat op to its output.
"""

import hashlib
import json
from typing import Any

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
            value = json.loads(value)
        except ValueError:
            pass
    for part in {parts!r}:
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
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


def seed_op_id(chat_op_id: str) -> str:
    """The id of the row function generated for one chat op."""
    return f"{chat_op_id}{SEED_SUFFIX}"


def row_seed(
    job_seed: int, template_id: str, round_number: int | None, key: object
) -> int:
    """The seed one row draws with: a pseudo-random function of the job seed,
    the op's template id, the round and the row key, so the same four values
    give the same draw wherever the row runs."""
    payload = json.dumps(
        [job_seed, template_id, round_number, key],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % SEED_MODULUS


def seed_source(
    fn_name: str,
    job_seed: int,
    template_id: str,
    round_number: int | None,
    parts: list[str],
) -> str:
    """The source of a keyed-seed row function: it reads one row's key out of
    its input column and returns the seed that row draws with. The job seed,
    template id, round and key path are constants of the generated code, so the
    body never runs user code; ``row_seed`` is the same hash over the same
    inputs, so both sides of the runtime share one definition."""
    return _SEED_FN.format(
        fn_name=fn_name,
        parts=parts,
        missing=f"LLMChatOp '{template_id}' has no row key in '%s' with path"
        f" {'.'.join(parts) or '<the value itself>'}",
        null=f"LLMChatOp '{template_id}' has a null row key in '%s'",
        job_seed=job_seed,
        template_id=template_id,
        round_number=round_number,
        modulus=SEED_MODULUS,
    )


def key_target(
    op_id: str,
    seed: dict[str, Any],
    *,
    rowwise_template: str | None,
    config: dict[str, Any],
) -> tuple[str, str]:
    """The ``(node, path)`` a key form reads, after checking the form fits
    this chat op."""
    key = seed.get("key")
    if not isinstance(key, dict):
        raise ValueError(f"LLMChatOp '{op_id}' rowwise_seed key must be a mapping")
    if rowwise_template is None:
        raise ValueError(
            f"LLMChatOp '{op_id}' rowwise_seed key needs a rowwise_template;"
            " only a rowwise LLMChatOp seeds each row separately"
        )
    num_beams = config.get("num_beams")
    if num_beams is not None and num_beams > 1:
        raise ValueError(
            f"LLMChatOp '{op_id}' rowwise_seed is not supported with num_beams"
            " > 1: the beam path takes no per-row seed"
        )
    if config.get("api") is not None:
        raise ValueError(
            f"LLMChatOp '{op_id}' rowwise_seed is available only for local vLLM"
            " generation, not API mode, which takes no per-row seed"
        )
    node_ref = key.get("node")
    path = key.get("path")
    if not isinstance(node_ref, str):
        raise ValueError(f"LLMChatOp '{op_id}' rowwise_seed key must name a node")
    if not isinstance(path, str):
        raise ValueError(f"LLMChatOp '{op_id}' rowwise_seed key must name a path")
    return node_ref, path


def seed_op(
    chat_op_id: str,
    key_node: str,
    key_path: str,
    *,
    job_seed: int,
    round_number: int | None,
    template_id: str | None = None,
) -> dict[str, Any]:
    """The row-mode ``LambdaOp`` that returns each row's seed from its key.
    ``template_id`` is the op id as the user wrote it; it defaults to the
    internal id."""
    return {
        "_id": seed_op_id(chat_op_id),
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
    """Rewrite every key-form ``rowwise_seed`` of a serialized native graph,
    adding each generated seed op and binding the chat op to it."""
    rewritten = graph.copy()
    for op_id, op in graph.items():
        if op.get("_op") != "LLMChatOp":
            continue
        seed = op.get("rowwise_seed")
        if not isinstance(seed, dict) or "key" not in seed:
            continue
        node_ref, path = key_target(
            op_id,
            seed,
            rowwise_template=op.get("rowwise_template"),
            config=op.get("config") or {},
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
        rewritten[op_id] = {
            **op,
            "rowwise_seed": {"node": generated["_id"], "path": "items.output"},
            "_inputs": [*op.get("_inputs", []), generated["_id"]],
        }
        rewritten[generated["_id"]] = generated
    return rewritten
