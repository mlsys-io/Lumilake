"""Deriving a rowwise chat op's per-row seed from a row key.

``rowwise_seed: {key: {node, path}}`` names a column that holds each row's key;
the seed is then a pseudo-random function of the job seed, the op's id, the
round and that key, so it does not depend on batch position or worker. The
parser turns the key form into a generated list-mode ``LambdaOp`` that reads the
key column and returns one seed per row, and binds the chat op to its output.
"""

from typing import Any

from lumilake_server.utils import lambda_runtime

SEED_SUFFIX = "__seed"
SEED_FN_NAME = "seed_for_row"
_OUTPUT = "output"


def key_parts(path: str) -> list[str]:
    """The field names a key path drills below ``items.output``: the whole
    output is the value itself, so it drills nothing."""
    if not path.startswith("items."):
        raise ValueError(f"key path must start with 'items.', got '{path}'")
    parts = [part for part in path[len("items.") :].split(".") if part]
    if parts[:1] == [_OUTPUT]:
        parts = parts[1:]
    return parts


def seed_op_id(chat_op_id: str) -> str:
    """The id of the row function generated for one chat op."""
    return f"{chat_op_id}{SEED_SUFFIX}"


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
    if not isinstance(path, str) or not path:
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
    """The ``LambdaOp`` that returns one seed per row from each row's key.
    ``template_id`` is the op id as the user wrote it; it defaults to the
    internal id."""
    return {
        "_id": seed_op_id(chat_op_id),
        "_op": "LambdaOp",
        "_max_iter": None,
        "_inputs": [key_node],
        "fn_name": SEED_FN_NAME,
        "_code": lambda_runtime.seed_source(
            SEED_FN_NAME,
            job_seed,
            chat_op_id if template_id is None else template_id,
            round_number,
            key_parts(key_path),
        ),
        "mode": "list",
    }


def rewrite_native_keys(
    graph: dict[str, dict[str, Any]], *, job_seed: int
) -> dict[str, dict[str, Any]]:
    """Rewrite every key-form ``rowwise_seed`` of a serialized native graph,
    adding each generated seed op and binding the chat op to it."""
    rewritten = dict(graph)
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
        rewritten[op_id] = {
            **op,
            "rowwise_seed": {"node": generated["_id"], "path": "items.output"},
            "_inputs": [*op.get("_inputs", []), generated["_id"]],
        }
        rewritten[generated["_id"]] = generated
    return rewritten
