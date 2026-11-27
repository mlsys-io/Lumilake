"""Typed ``data_spec`` for every runtime task type."""

import re
from collections.abc import Mapping
from enum import Enum, StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_serializer,
)

from lumilake_server.common import InputRef
from lumilake_server.utils.utils import parse_memory_to_bytes

PYTHON_TASK_TYPE = "python"
RETRIEVAL_TASK_TYPE = "data_retrieval"
PROFILE_TASK_TYPE = "data_profiling"
GPU_TYPE_WILDCARDS = frozenset({"any", "auto", "*"})
"""``GpuRequest.type`` values that place no type constraint."""

MODEL_TASK_TYPES = frozenset(
    {"inference", "diffusion", "embedding", "classification", "omni_text2image", "api"}
)
ALL_TASK_TYPES = MODEL_TASK_TYPES | {
    PYTHON_TASK_TYPE,
    RETRIEVAL_TASK_TYPE,
    PROFILE_TASK_TYPE,
}


class PythonMode(StrEnum):
    """The mode a python step runs in; each member's value is its YAML word."""

    ROW = "row"
    LIST = "list"
    ALIGNED = "aligned"
    FLATTEN = "flatten"
    REGROUP = "regroup"
    PAIRS = "pairs"
    FLATTEN_STEP = "flatten_step"
    EXPAND = "expand"
    REDUCE = "reduce"
    JOIN = "join"
    PARSE = "parse"


class JoinHow(StrEnum):
    """Which keys a lowered join keeps."""

    INNER = "inner"
    LEFT = "left"
    FULL = "full"


ReduceOnFail = Literal["drop_failed", "fail_group"]
ExpandOnEmpty = Literal["drop", "fail"]


class _StrictModel(BaseModel):
    """Base model that refuses unknown keys."""

    model_config = ConfigDict(extra="forbid")


class GpuRequest(_StrictModel):
    """GPUs a standalone python step asks FlowMesh for.

    ``type`` and ``memory`` narrow which device FlowMesh may pick.
    """

    count: int = Field(default=1, ge=1, le=8)
    type: str | None = Field(default=None, max_length=128)
    """Device-name substring; a wildcard such as ``any`` places no constraint."""
    memory: str | None = Field(
        default=None, max_length=6, pattern=r"^[1-9]\d{0,3}(Ki|Mi|Gi|Ti)$"
    )
    """Per-device memory floor (e.g. ``"24Gi"``)."""

    def allows_device(self, name: str | None, memory_total_bytes: int | None) -> bool:
        """True when one card, by name and total memory, meets this request."""
        wanted = (self.type or "").strip().lower()
        if (
            wanted
            and wanted not in GPU_TYPE_WILDCARDS
            and name is not None
            and wanted not in name.lower()
        ):
            return False
        floor = parse_memory_to_bytes(self.memory) if self.memory else None
        if floor is None or memory_total_bytes is None:
            return True
        return memory_total_bytes >= floor


class DataSpecModel(_StrictModel):
    """Base model of every runtime task's data spec."""

    @property
    def gpu_request(self) -> GpuRequest | None:
        """The GPUs this step's task asks FlowMesh for, if any."""
        return None


class NodeRead(DataSpecModel):
    """An upstream node read at a path."""

    node: str
    path: str = ""


class ListData(DataSpecModel):
    """A list-fed task reading literal items, an upstream node, or a
    placeholder that resolves to a whole upstream list."""

    type: Literal["list"] = "list"
    items: list[Any] | str | None = None
    node: str | None = None
    path: str | None = None
    grouped: bool | None = None
    lumid_cfg: dict[str, str] | None = None


class Message(DataSpecModel):
    """One chat message in a template's format step."""

    role: str | None = None
    content: str


class FormatArgument(DataSpecModel):
    """One placeholder value a format step fills."""

    label: str
    value: str


class FormatStep(DataSpecModel):
    """One format step holding a template or a function and its arguments."""

    label: str
    template: str | None = None
    function: str | None = None
    arguments: list[FormatArgument | str | list[Message]] = Field(default_factory=list)


class FormatConfig(DataSpecModel):
    """The format options a graph template carries."""

    messages: list[Message] = Field(default_factory=list)
    steps: list[FormatStep] | None = None


class TemplateOptions(DataSpecModel):
    """A graph template's options block."""

    format: FormatConfig


type ColumnList = list[Column]


class DataFrame(DataSpecModel):
    """A rowwise table with one column per value and the messages its rows render."""

    type: Literal["dataframe"]
    columns: ColumnList = Field(default_factory=list)
    messages: list[Message] | None = None
    seeds: list[int | None] | NodeRead | None = None


class Column(DataSpecModel):
    """One labelled column: a nested table, a literal list or an upstream read."""

    label: str | None = None
    data: DataFrame | ListData | None = None
    node: str | None = None
    path: str | None = None
    input_ref: InputRef | None = None

    @field_serializer("input_ref")
    def _keep_ref_path(self, value: InputRef | None) -> Any:
        """Serialize the ref whole: a dump would drop its null path."""
        return value


class LiteralColumn(Column):
    """A ``rowwise_columns`` or ``aggregate_table`` entry holding its own list."""

    label: str
    data: DataFrame | ListData
    node: None = None
    path: None = None


class NodeColumn(Column):
    """A rowwise table entry read from a node's output at an optional path."""

    label: str
    node: str
    data: None = None


ColumnTable = list[LiteralColumn | NodeColumn]


_ColumnTableEntry = Annotated[
    NodeColumn | LiteralColumn, Field(union_mode="left_to_right")
]
_COLUMN_TABLE: TypeAdapter[ColumnTable] = TypeAdapter(list[_ColumnTableEntry])


def build_column_table(table: Any, field: str) -> ColumnTable | None:
    """Validate one ``rowwise_columns`` or ``aggregate_table`` table."""
    if table is None:
        return None
    try:
        return _COLUMN_TABLE.validate_python(table)
    except ValidationError as exc:
        error = exc.errors(include_url=False)[0]
        where = "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in error["loc"]
            if part not in ("NodeColumn", "LiteralColumn")
        )
        message = error["msg"].removeprefix("Value error, ")
        raise ValueError(f"{field}{where}: {message}") from exc


class GraphTemplate(DataSpecModel):
    """The template a message-chain task renders."""

    name: str
    columns: list[Column] = Field(default_factory=list)
    options: TemplateOptions


class GraphTemplateData(DataSpecModel):
    """A message-chain task over a rendered graph template."""

    type: Literal["graph_template"]
    template: GraphTemplate
    image_embedding: NodeRead | None = None


class ColumnPlanEntry(DataSpecModel):
    """One python plan entry: an upstream stage read or literal values."""

    kind: Literal["stage", "literal"]
    node: str | None = None
    path: str | None = None
    flatten: Literal[True] | None = None
    role: Literal["seed"] | None = None
    values: list[Any] | None = None
    label: str | None = None
    grouped: bool | None = None
    file: str | None = None


class JoinDescription(DataSpecModel):
    """A lowered join: key paths, row-column plan indexes, attach values and sources."""

    node: str
    by: list[str]
    inputs: list[int]
    attach: list[tuple[str, int, int]]
    how: JoinHow
    require_match: bool
    on_missing: Literal["null_fill", "fail_row"]


class ParseDescription(DataSpecModel):
    """A parse step's format, pattern, required keys, flags and failure policy."""

    format: Literal["json", "regex"]
    on_fail: Literal["marker", "raw"]
    pattern: str | None = None
    require: list[str] | None = None
    flags: str | None = None


class PythonData(DataSpecModel):
    """A standalone or reshape python step."""

    code: str | None = None
    fn_name: str | None = None
    plan: list[ColumnPlanEntry] = Field(default_factory=list)
    mode: PythonMode | None = None
    by: str | None = None
    on_fail: ReduceOnFail | None = None
    on_empty: ExpandOnEmpty | None = None
    join: JoinDescription | None = None
    parse: ParseDescription | None = None
    template: str | None = None
    timeout_s: int | float = 600
    memory_mb: int | None = None
    image: str | None = None
    gpu: GpuRequest | None = None
    expected_runtime_s: float | None = None
    row_count: int | None = None
    cache: bool | None = None

    @property
    def gpu_request(self) -> GpuRequest | None:
        """The GPUs this step's task asks FlowMesh for, if any."""
        return self.gpu


class RetrievalParam(DataSpecModel):
    """One retrieval placeholder bound to a literal list or an upstream read,
    carrying any other key the user wrote for the worker."""

    model_config = ConfigDict(extra="allow")

    label: str | None = None
    data: ListData | None = None
    node: str | None = None
    path: str | None = None
    input_ref: InputRef | None = None

    @field_serializer("input_ref")
    def _keep_ref_path(self, value: InputRef | None) -> Any:
        """Serialize the ref whole: a dump would drop its null path."""
        return value


class RetrievalData(DataSpecModel):
    """A lumid-data backend retrieval."""

    type: Literal["lumid"]
    mode: Literal["sql", "s3", "agent"]
    rows: Literal["one", "table"] | None = None
    require_non_empty_rows: bool | None = None
    lumid_data_url: str | None = None
    lumid_data_token: str | None = None
    params: list[RetrievalParam] = Field(default_factory=list)
    template: str | None = None
    table: str | None = None
    output_format: str | None = None
    encoding: str | None = None
    description: str | None = None
    schema_scope: str | None = None
    max_steps: int | None = None
    model: str | None = None
    verify: bool | None = None


class SqlProfileData(DataSpecModel):
    """A SQL data-profile task."""

    type: Literal["sql"]
    template: str
    params: list[RetrievalParam] = Field(default_factory=list)
    constraints: list[dict[str, Any]] | None = None
    num_test_queries: int | None = None
    table: str | None = None


class S3ProfileData(DataSpecModel):
    """An S3 data-profile task."""

    type: Literal["s3"]
    template: str
    params: list[RetrievalParam] = Field(default_factory=list)
    encoding: str | None = None


class EmptyData(DataSpecModel):
    """A task that carries no data spec."""


ModelTaskData = Annotated[
    ListData | GraphTemplateData | DataFrame,
    Field(union_mode="left_to_right"),
]

_MODEL_DATA: TypeAdapter[ModelTaskData] = TypeAdapter(ModelTaskData)
_PROFILE_DATA: TypeAdapter[SqlProfileData | S3ProfileData] = TypeAdapter(
    Annotated[SqlProfileData | S3ProfileData, Field(union_mode="left_to_right")]
)


def parse_data_spec(task_type: str, value: dict[str, Any]) -> DataSpecModel:
    """The model a serialized ``data_spec`` dict of ``task_type`` carries."""
    if not value:
        return EmptyData()
    if task_type not in ALL_TASK_TYPES:
        raise ValueError(f"data_spec of unknown task type {task_type!r}")
    try:
        if task_type == PYTHON_TASK_TYPE:
            return PythonData.model_validate(value)
        if task_type == RETRIEVAL_TASK_TYPE:
            return RetrievalData.model_validate(value)
        if task_type == PROFILE_TASK_TYPE:
            return _PROFILE_DATA.validate_python(value)
        return _MODEL_DATA.validate_python(value)
    except ValidationError as exc:
        raise ValueError(f"invalid data_spec for a {task_type} task: {exc}") from exc


_PLACEHOLDER_RE = re.compile(r"\$\{([^}.]+)\.([^}]+)\}")


def _remap_entry(key: str, item: Any, mapping: Mapping[str, str]) -> Any:
    """A dict or model field with its value renamed when it is a mapped node."""
    if key == "node" and isinstance(item, str) and item in mapping:
        return mapping[item]
    return _remap_value(item, mapping)


def _remap_value(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, Enum):
        return value
    if isinstance(value, BaseModel):
        return remap_model(value, mapping)
    if isinstance(value, dict):
        return remap_node_dict(value, mapping)
    if isinstance(value, (list, tuple)):
        items = [_remap_value(item, mapping) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    if isinstance(value, str):
        return _PLACEHOLDER_RE.sub(
            lambda match: (
                f"${{{mapping[match.group(1)]}.{match.group(2)}}}"
                if match.group(1) in mapping
                else match.group(0)
            ),
            value,
        )
    return value


def remap_model[M: BaseModel](model: M, mapping: Mapping[str, str]) -> M:
    """Copy ``model`` with the ``node`` keys and ``${node.path}`` placeholders
    of mapped nodes renamed, including inside its extra keys."""
    stored = {**model.__dict__, **(model.model_extra or {})}
    return model.model_copy(
        update={
            name: _remap_entry(name, item, mapping) for name, item in stored.items()
        }
    )


def remap_node_dict(spec: dict[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    """Copy a free-form spec with the ``node`` keys and ``${node.path}``
    placeholders of mapped nodes renamed."""
    return {key: _remap_entry(key, item, mapping) for key, item in spec.items()}


Column.model_rebuild()
