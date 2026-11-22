"""The typed ``data_spec`` a test's runtime op carries."""

from typing import Any, Literal

from lumilake_server.data_specs import (
    DataFrame,
    DataSpecModel,
    EmptyData,
    FormatConfig,
    GraphTemplate,
    GraphTemplateData,
    ListData,
    Message,
    PythonData,
    PythonMode,
    RetrievalData,
    S3ProfileData,
    SqlProfileData,
    TemplateOptions,
    parse_data_spec,
)


def spec(task_type: str, value: dict[str, Any]) -> DataSpecModel:
    """The model a runtime op of ``task_type`` carries for ``value``."""
    return parse_data_spec(task_type, value)


def empty() -> EmptyData:
    """A task that carries no data spec."""
    return EmptyData()


def python_spec(mode: PythonMode | None = None) -> PythonData:
    """A python step with the given mode and no plan."""
    return PythonData(mode=mode)


def retrieval_spec(
    mode: Literal["sql", "s3", "agent"], template: str | None = None
) -> RetrievalData:
    """A lumid-data retrieval in the given mode."""
    return RetrievalData(type="lumid", mode=mode, template=template)


def list_spec(*items: Any) -> ListData:
    """A list-fed task over literal items."""
    return ListData(items=list(items))


def dataframe_spec(*contents: str) -> DataFrame:
    """A rowwise table with one user message per content."""
    return DataFrame(
        type="dataframe",
        messages=[Message(role="user", content=c) for c in contents],
    )


def graph_template_spec(
    template: GraphTemplate | None = None,
) -> GraphTemplateData:
    """A message-chain task over a graph template."""
    return GraphTemplateData(
        type="graph_template",
        template=(
            template
            if template is not None
            else GraphTemplate(
                name="format",
                options=TemplateOptions(format=FormatConfig()),
            )
        ),
    )


def sql_profile_spec(template: str = "SELECT 1") -> SqlProfileData:
    """A SQL data-profile task."""
    return SqlProfileData(type="sql", template=template)


def s3_profile_spec(template: str = "data.json") -> S3ProfileData:
    """An S3 data-profile task."""
    return S3ProfileData(type="s3", template=template)
