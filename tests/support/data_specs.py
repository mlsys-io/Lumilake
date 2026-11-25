"""Runtime ops for tests, built from their serialized ``data_spec`` dicts."""

from typing import Any

from lumilake_server.data_specs import parse_data_spec
from lumilake_server.runtime.runtime_ops import RuntimeOp


def runtime_op(
    node_id: str,
    task_type: str,
    backend: str,
    model: str,
    data_spec: dict[str, Any],
    model_spec: dict[str, Any],
    inference_spec: dict[str, Any],
    **fields: Any,
) -> RuntimeOp:
    """A runtime op whose ``data_spec`` is parsed from its dict."""
    spec = parse_data_spec(task_type, data_spec)
    return RuntimeOp(
        node_id, task_type, backend, model, spec, model_spec, inference_spec, **fields
    )
