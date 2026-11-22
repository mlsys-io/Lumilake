"""A runtime op for tests: its dict ``data_spec`` validates at the boundary."""

from typing import Any

from lumilake_server.data_specs import (
    DataSpecModel,
    EmptyData,
    FormatConfig,
    GraphTemplate,
    TemplateOptions,
    parse_data_spec,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _model(data_spec: dict[str, Any] | DataSpecModel, task_type: str) -> DataSpecModel:
    """The model a dict spec validates into, or the model itself."""
    if isinstance(data_spec, dict):
        return parse_data_spec(task_type, data_spec)
    return data_spec


def runtime_op(
    node_id: str,
    task_type: str,
    backend: str,
    model: str,
    data_spec: dict[str, Any] | DataSpecModel,
    model_spec: dict[str, Any],
    inference_spec: dict[str, Any],
    api_spec: dict[str, Any] | None = None,
    output_enforcement: dict[str, Any] | None = None,
    dependencies: tuple[str, ...] = (),
    output_spec: dict[str, Any] | None = None,
    condition: dict[str, str] | None = None,
    skip_when_all_inputs_empty: bool = False,
    samples: bool = False,
) -> RuntimeOp:
    """A runtime op whose dict ``data_spec`` validates through the boundary."""
    return RuntimeOp(
        node_id=node_id,
        task_type=task_type,
        backend=backend,
        model=model,
        data_spec=_model(data_spec, task_type),
        model_spec=model_spec,
        inference_spec=inference_spec,
        api_spec={} if api_spec is None else api_spec,
        output_enforcement=output_enforcement,
        dependencies=dependencies,
        output_spec=output_spec,
        condition=condition,
        skip_when_all_inputs_empty=skip_when_all_inputs_empty,
        samples=samples,
    )


def empty() -> EmptyData:
    """A task that carries no data spec."""
    return EmptyData()


def graph_template_spec():
    """A message-chain task over an empty format template."""
    return _model(
        {
            "type": "graph_template",
            "template": GraphTemplate(
                name="format",
                options=TemplateOptions(format=FormatConfig()),
            ),
        },
        "inference",
    )
