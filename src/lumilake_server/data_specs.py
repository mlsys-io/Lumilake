"""Typed ``data_spec`` for every runtime task type."""

from typing import Annotated, Any, Literal, TypeGuard

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

PYTHON_TASK_TYPE = "python"
RETRIEVAL_TASK_TYPE = "data_retrieval"
PROFILE_TASK_TYPE = "data_profiling"


class DataSpecModel(BaseModel):
    """Base model that refuses unknown keys."""

    model_config = ConfigDict(extra="forbid")


class NodeRead(DataSpecModel):
    """An upstream node read at a path."""

    node: str
    path: str = ""


class ImageEmbedding(NodeRead):
    """The embedding artifact a VLM inference task reads."""


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


class Column(DataSpecModel):
    """One labelled column: a nested table, a literal list or an upstream read."""

    label: str | None = None
    data: "DataFrame | ListData | None" = None
    node: str | None = None
    path: str | None = None


class DataFrame(DataSpecModel):
    """A rowwise table with one column per value and the messages its rows render."""

    type: Literal["dataframe"]
    columns: list[Column] = Field(default_factory=list)
    messages: list[Message] | None = None
    seeds: list[int | None] | NodeRead | None = None


class GraphTemplate(DataSpecModel):
    """The template a message-chain task renders."""

    name: str
    columns: list[Column] = Field(default_factory=list)
    options: TemplateOptions


class GraphTemplateData(DataSpecModel):
    """A message-chain task over a rendered graph template."""

    type: Literal["graph_template"]
    template: GraphTemplate
    image_embedding: ImageEmbedding | None = None


class ColumnPlanEntry(DataSpecModel):
    """One python plan entry: an upstream stage read or literal values."""

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
    """One retrieval placeholder bound to a literal list or an upstream read,
    carrying any other key the user wrote for the worker."""

    model_config = ConfigDict(extra="allow")

    label: str | None = None
    data: ListData | None = None
    node: str | None = None
    path: str | None = None


class RetrievalData(DataSpecModel):
    """A lumid-data backend retrieval."""

    type: Literal["lumid"]
    mode: Literal["sql", "s3", "agent"]
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


MODEL_TASK_TYPES = frozenset(
    {"inference", "diffusion", "embedding", "classification", "omni_text2image", "api"}
)
ALL_TASK_TYPES = MODEL_TASK_TYPES | {
    PYTHON_TASK_TYPE,
    RETRIEVAL_TASK_TYPE,
    PROFILE_TASK_TYPE,
}


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
            return validate_profile(value)
        return validate_model_data(value)
    except ValidationError as exc:
        raise ValueError(f"invalid data_spec for a {task_type} task: {exc}") from exc


def validate_model_data(value: dict[str, Any]) -> ModelTaskData:
    """The list, dataframe or graph-template spec a model task carries."""
    return _MODEL_DATA.validate_python(value)


def validate_profile(value: dict[str, Any]) -> SqlProfileData | S3ProfileData:
    """The SQL or S3 spec a data-profile task carries."""
    return _PROFILE_DATA.validate_python(value)


def is_sql_data(data_spec: BaseModel) -> TypeGuard[SqlProfileData | RetrievalData]:
    """Whether the spec reads a SQL source."""
    if isinstance(data_spec, SqlProfileData):
        return True
    return isinstance(data_spec, RetrievalData) and data_spec.mode == "sql"


def is_s3_data(data_spec: BaseModel) -> TypeGuard[S3ProfileData | RetrievalData]:
    """Whether the spec reads an S3 source."""
    if isinstance(data_spec, S3ProfileData):
        return True
    return isinstance(data_spec, RetrievalData) and data_spec.mode == "s3"


def dump_data_spec(model: BaseModel) -> dict[str, Any]:
    """The serialized dict for a typed ``data_spec`` model."""
    return model.model_dump(exclude_none=True)


Column.model_rebuild()
