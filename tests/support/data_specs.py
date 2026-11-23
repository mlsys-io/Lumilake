"""Runtime ops for tests, built from their typed ``data_spec`` models."""

from typing import Any

from lumilake_server.data_specs import DataSpecModel, dump_data_spec
from lumilake_server.runtime.runtime_ops import RuntimeOp


def specs(op: RuntimeOp) -> dict[str, Any]:
    """A runtime op's data spec as the dict a worker receives."""
    return dump_data_spec(op.data_spec)


def runtime_op(
    node_id: str,
    task_type: str,
    backend: str,
    model: str,
    data_spec: DataSpecModel,
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
    """A runtime op carrying its typed ``data_spec``."""
    return RuntimeOp(
        node_id=node_id,
        task_type=task_type,
        backend=backend,
        model=model,
        data_spec=data_spec,
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
