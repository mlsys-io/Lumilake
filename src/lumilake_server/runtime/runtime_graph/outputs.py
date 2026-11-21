from dataclasses import dataclass
from enum import Enum

from lumilake_server.common import retrieval_items_path
from lumilake_server.ops import (
    DataOp,
    DataRetrievalOp,
    FormatOp,
    InputOp,
    LambdaOp,
    LLMOp,
    MessageOp,
    Op,
)
from lumilake_server.ops.embedding_ops import EmbeddingOp
from lumilake_server.ops.join_ops import JoinOp
from lumilake_server.ops.llm_ops import (
    ImageGenerationOp,
    LLMChatOp,
    LLMCompletionOp,
    LLMVisionOp,
)
from lumilake_server.ops.ops import RowShape
from lumilake_server.ops.parse_ops import ParseOp


def item_relative_path(path: str) -> str:
    """Strip the result-envelope prefix from an upstream output path, so a
    python step reads it inside the item list it already unwrapped."""
    for prefix in ("value.items.", "items."):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


_WHOLE = "items.output"
_VALUE_ROOT = "value.items.output"
_API_ROW_PATH = "items.rows.json.choices[0].message.content"
_API_ITEM_PATH = "items.json.choices[0].message.content"


class OutputKind(Enum):
    """The envelope an op's consumers read, which fixes how a declared
    ``items.output`` path maps onto the runtime result."""

    DECLARED = "declared"
    FLATTENED = "flattened"
    API_ROW = "api_row"
    API_ITEM = "api_item"
    LIST_LAMBDA = "list_lambda"
    ROW_LAMBDA = "row_lambda"
    JOIN = "join"


_LAMBDA_KINDS = frozenset(
    {OutputKind.LIST_LAMBDA, OutputKind.ROW_LAMBDA, OutputKind.JOIN}
)


@dataclass(frozen=True, slots=True)
class OpOutput:
    """Where and how a consumer reads an upstream op's values; a ``row_count``
    of ``None`` is not knowable at build time, and a ``default_output_path`` of
    ``None`` means an OutputOp archives the result at its own item shape."""

    node: str
    path: str
    row_count: int | None
    default_output_path: str | None
    kind: OutputKind
    one_item_per_row: bool
    needs_flatten: bool

    def read(self, declared: str, consumer: str) -> str:
        """Map a declared ``items.output``-rooted read onto this output."""
        if self.kind is OutputKind.DECLARED:
            return declared
        if declared == _WHOLE:
            return self.path
        if declared.startswith(_WHOLE) and self.kind not in (
            OutputKind.API_ROW,
            OutputKind.API_ITEM,
        ):
            if self.kind is OutputKind.ROW_LAMBDA:
                return "value.items.value." + declared[len(_WHOLE) + 1 :]
            return "value." + declared
        raise ValueError(self._unreadable_message(declared, consumer))

    def _unreadable_message(self, declared: str, consumer: str) -> str:
        head = f"{consumer} reads '{declared}' from"
        if self.kind is OutputKind.FLATTENED:
            return (
                f"{head} per-row VLM '{self.node}'; only 'items.output' is supported."
            )
        noun = (
            "join"
            if self.kind is OutputKind.JOIN
            else "Lambda" if self.kind in _LAMBDA_KINDS else "API upstream"
        )
        return (
            f"{head} {noun} '{self.node}';"
            " only 'items.output' (the whole output) is supported."
        )


def is_api_task(op: Op) -> bool:
    """Whether an LLM op dispatches as an API task rather than a VLM."""
    return (
        isinstance(op, LLMOp)
        and not isinstance(op, LLMVisionOp)
        and op.config.api is not None
    )


def emits_one_item_per_row(op: Op) -> bool:
    """Whether a multi-row op's result holds one scalar-per-row item per input
    row, so a ``node``/``path`` reference resolves to one value per row."""
    if isinstance(op, LLMCompletionOp):
        return True
    return (
        isinstance(op, LLMChatOp)
        and op.rowwise_template is None
        and not is_api_task(op)
    )


def needs_flatten(op: Op) -> bool:
    """Whether a python step must flatten the op's grouped output to one value
    per row. A per-row VLM's regroup step already emits one list per group."""
    return (
        isinstance(op, LLMChatOp)
        and not isinstance(op, LLMVisionOp)
        and op.row_shape is RowShape.GROUPED
    )


def op_output(
    op: Op,
    flattened_vlm_ids: set[str],
    inputs_dict: dict[str, list[str]],
    graph_dict: dict[str, Op],
) -> OpOutput:
    """Describe how a consumer reads ``op``'s values: the emitting node, the
    envelope path, the row count, and what an OutputOp archives."""
    if op.id in flattened_vlm_ids:
        kind = OutputKind.FLATTENED
    elif is_api_task(op):
        rowwise = isinstance(op, LLMChatOp) and op.rowwise_template is not None
        kind = OutputKind.API_ROW if rowwise else OutputKind.API_ITEM
    elif isinstance(op, LambdaOp):
        kind = OutputKind.LIST_LAMBDA if op.mode != "row" else OutputKind.ROW_LAMBDA
    elif isinstance(op, JoinOp):
        # A join is enveloped like a list Lambda: one item per row.
        kind = OutputKind.JOIN
    elif isinstance(op, ParseOp):
        kind = OutputKind.ROW_LAMBDA
    else:
        kind = OutputKind.DECLARED
    default_path: str | None
    if isinstance(op, DataRetrievalOp):
        path = default_path = retrieval_items_path(op.data_spec["mode"])
    elif op.result_path is not None:
        path = default_path = op.result_path
    elif kind is OutputKind.API_ROW:
        path = default_path = _API_ROW_PATH
    elif kind is OutputKind.API_ITEM:
        path = default_path = _API_ITEM_PATH
    else:
        path = _WHOLE if kind is OutputKind.DECLARED else _VALUE_ROOT
        default_path = (
            _WHOLE if kind in (OutputKind.LIST_LAMBDA, OutputKind.JOIN) else None
        )
    return OpOutput(
        node=op.id,
        path=path,
        row_count=output_row_count(op, inputs_dict, graph_dict),
        default_output_path=default_path,
        kind=kind,
        one_item_per_row=emits_one_item_per_row(op),
        needs_flatten=needs_flatten(op),
    )


def output_row_count(
    op: Op,
    inputs_dict: dict[str, list[str]],
    graph_dict: dict[str, Op],
) -> int | None:
    """Static row count of an op's output, or ``None`` when not knowable at
    build time. A rowwise op's count comes from its ``rowwise_columns``."""
    if isinstance(op, InputOp):
        values = inputs_dict.get(op.name)
        return len(values) if values is not None else None
    if isinstance(op, DataOp):
        return len(op.data)
    row_source = op.row_source
    if row_source is not None:
        return output_row_count(row_source, inputs_dict, graph_dict)
    if isinstance(op, FormatOp) or (
        isinstance(op, LambdaOp) and op.mode in ("row", "aligned")
    ):
        return _max_known(
            [output_row_count(inp, inputs_dict, graph_dict) for inp in op.inputs]
        )
    if isinstance(op, (LLMCompletionOp, EmbeddingOp, ImageGenerationOp)):
        content = op.prompt if isinstance(op, LLMCompletionOp) else op.content
        return output_row_count(content, inputs_dict, graph_dict)
    if isinstance(op, LLMChatOp):
        return _llm_row_count(op, inputs_dict, graph_dict)
    return None


def _llm_row_count(
    op: LLMChatOp,
    inputs_dict: dict[str, list[str]],
    graph_dict: dict[str, Op],
) -> int | None:
    """A rowwise LLM counts its ``rowwise_columns``; any other takes the widest
    row count of the ops its message chain binds."""
    if op.rowwise_columns:
        counts: list[int | None] = []
        for col in op.rowwise_columns:
            data = col.get("data")
            if isinstance(data, dict):
                items = data.get("items")
                if isinstance(items, list):
                    counts.append(len(items))
                    continue
            node_ref = col.get("node")
            if isinstance(node_ref, str):
                upstream = graph_dict.get(node_ref)
                if upstream is not None:
                    counts.append(output_row_count(upstream, inputs_dict, graph_dict))
        return _max_known(counts)
    messages = op.messages.messages if isinstance(op.messages, MessageOp) else []
    return _max_known(
        [
            output_row_count(message.content, inputs_dict, graph_dict)
            for message in messages
            if isinstance(message.content, Op)
        ]
    )


def _max_known(counts: list[int | None]) -> int | None:
    known = [count for count in counts if count is not None]
    return max(known) if known else None
