"""Typed ``data_spec`` for every runtime task type.

Each task type's ``data_spec`` is one of these models. Builders construct the
model and ``RuntimeOp`` keeps it; readers use attributes. Serializing dumps the
model with unset fields omitted, which is the dict the builders wrote before.
"""

from typing import Annotated, Any, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

PYTHON_TASK_TYPE = "python"
RETRIEVAL_TASK_TYPE = "data_retrieval"
PROFILE_TASK_TYPE = "data_profiling"
MODELLED_TASK_TYPES = frozenset(
    {
        PYTHON_TASK_TYPE,
        RETRIEVAL_TASK_TYPE,
        PROFILE_TASK_TYPE,
        "inference",
        "diffusion",
        "embedding",
        "classification",
        "omni_text2image",
        "api",
    }
)


class DataSpecModel(BaseModel):
    """Base for every ``data_spec`` model: unknown keys are refused."""

    model_config = ConfigDict(extra="forbid")


class NodeRead(DataSpecModel):
    """An upstream node read at a path."""

    node: str
    path: str = ""


class ImageEmbedding(NodeRead):
    """The embedding artifact a VLM inference task reads."""


class ListData(DataSpecModel):
    """A list-fed task: literal items or an upstream node read."""

    type: Literal["list"] = "list"
    items: list[Any] | None = None
    node: str | None = None
    path: str | None = None
    grouped: bool | None = None
    lumid_cfg: dict[str, str] | None = None


class Value(DataSpecModel):
    """A column bound to one literal value."""

    type: Literal["value"]
    value: Any = None


class Message(DataSpecModel):
    """One chat message in a template's format step."""

    role: str | None = None
    content: str


class FormatArgument(DataSpecModel):
    """One placeholder value a format step fills."""

    label: str
    value: str


class FormatStep(DataSpecModel):
    """One format step: a template and its arguments, or a function with its
    serialized arguments."""

    label: str
    template: str | None = None
    function: str | None = None
    arguments: list[FormatArgument | str] = Field(default_factory=list)


class FormatConfig(DataSpecModel):
    """The ``format`` options a graph template carries."""

    messages: list[Message] = Field(default_factory=list)
    steps: list[FormatStep] | None = None
    template: str | None = None


class TemplateOptions(DataSpecModel):
    """A graph template's options block."""

    format: FormatConfig


class Column(DataSpecModel):
    """One column of a rowwise or aggregate table: a literal list, a nested
    table, one literal value, or an upstream node read, under a label."""

    label: str | None = None
    data: "DataFrame | ListData | Value | None" = None
    node: str | None = None
    path: str | None = None


class DataFrame(DataSpecModel):
    """A rowwise table: one column per value a row carries, plus the messages
    its rows render."""

    type: Literal["dataframe"]
    columns: list[Column] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    seeds: list[int | None] | NodeRead | None = None


class GraphTemplate(DataSpecModel):
    """The template a message-chain task renders."""

    name: str = "format"
    columns: list[Column] = Field(default_factory=list)
    options: TemplateOptions = Field(
        default_factory=lambda: TemplateOptions(format=FormatConfig())
    )


class GraphTemplateData(DataSpecModel):
    """A message-chain task over a rendered graph template."""

    type: Literal["graph_template"]
    template: GraphTemplate
    image_embedding: ImageEmbedding | None = None


class ColumnPlanEntry(DataSpecModel):
    """One entry of a python step's plan: an upstream stage read, or a literal
    value list. Flatten and regroup plans label their entries and mark the
    grouped ones."""

    kind: Literal["stage", "literal"]
    node: str | None = None
    path: str | None = None
    flatten: Literal[True] | None = None
    values: list[Any] | None = None
    file: str | None = None
    label: str | None = None
    grouped: bool | None = None


PythonMode = Literal[
    "row", "list", "aligned", "flatten", "regroup", "pairs", "flatten_step"
]


class PythonData(DataSpecModel):
    """A standalone or reshape python step."""

    code: str | None = None
    fn_name: str | None = None
    plan: list[ColumnPlanEntry] = Field(default_factory=list)
    mode: PythonMode | None = None
    template: str | None = None
    timeout_s: int | float = 600
    memory_mb: int | None = None
    image: str | None = None
    expected_runtime_s: float | None = None
    row_count: int | None = None
    cache: bool | None = None


class RetrievalParam(DataSpecModel):
    """One placeholder value of a retrieval template: a literal list or an
    upstream node read. Its placeholder is named ``label``, or ``name`` in the
    legacy form."""

    label: str | None = None
    name: str | None = None
    data: ListData | None = None
    node: str | None = None
    path: str | None = None


class RetrievalData(DataSpecModel):
    """A lumid-data backend retrieval, or a reduced list/sql/s3 spec a
    hand-built graph carries."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["lumid", "list", "sql", "s3"] = "lumid"
    mode: Literal["sql", "s3", "agent"] | None = None
    connection_string: str | None = None
    items: list[Any] | None = None
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
    template: str = ""
    params: list[RetrievalParam] = Field(default_factory=list)
    constraints: list[dict[str, Any]] | None = None
    num_test_queries: int | None = None
    table: str | None = None


class S3ProfileData(DataSpecModel):
    """An S3 data-profile task."""

    type: Literal["s3"]
    template: str = ""
    params: list[RetrievalParam] = Field(default_factory=list)
    encoding: str | None = None


class EmptyData(DataSpecModel):
    """A task that carries no data spec."""


class OpenData(DataSpecModel):
    """A spec of a task type this module does not model; its keys stay free."""

    model_config = ConfigDict(extra="allow")


class MessageChainData(DataSpecModel):
    """The system and user messages a reduced graph hands its LLM nodes."""

    messages: list[Message] = Field(default_factory=list)


ModelTaskData = Annotated[
    ListData | GraphTemplateData | DataFrame | MessageChainData,
    Field(union_mode="left_to_right"),
]

_MODEL_DATA: TypeAdapter[ModelTaskData] = TypeAdapter(ModelTaskData)
_PROFILE_DATA: TypeAdapter[SqlProfileData | S3ProfileData] = TypeAdapter(
    Annotated[SqlProfileData | S3ProfileData, Field(union_mode="left_to_right")]
)


def parse_data_spec(task_type: str, value: dict[str, Any] | BaseModel) -> DataSpecModel:
    """The typed model for a ``data_spec`` dict of ``task_type``.

    A model passes through unchanged; a dict is validated into the model its
    task type carries.
    """
    if isinstance(value, DataSpecModel):
        return value
    if isinstance(value, BaseModel):
        return OpenData.model_validate(value.model_dump())
    if not value:
        return EmptyData()
    if task_type not in MODELLED_TASK_TYPES:
        return OpenData.model_validate(value)
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


def is_sql_data(data_spec: BaseModel) -> TypeGuard[SqlProfileData | RetrievalData]:
    """Whether the spec reads a SQL source: a SQL profile node, or a retrieval
    whose ``type`` is ``sql``."""
    if isinstance(data_spec, SqlProfileData):
        return True
    return isinstance(data_spec, RetrievalData) and data_spec.type == "sql"


def is_s3_data(data_spec: BaseModel) -> TypeGuard[S3ProfileData | RetrievalData]:
    """Whether the spec reads an S3 source: an S3 profile node, or a retrieval
    whose ``type`` is ``s3``."""
    if isinstance(data_spec, S3ProfileData):
        return True
    return isinstance(data_spec, RetrievalData) and data_spec.type == "s3"


def dump_data_spec(model: BaseModel) -> dict[str, Any]:
    """The serialized dict for a typed ``data_spec`` model; a field the builder
    never set stays out of the wire dict."""
    return model.model_dump(exclude_none=True)


Column.model_rebuild()
