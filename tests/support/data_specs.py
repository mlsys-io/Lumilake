"""Runtime ops for tests, built from their serialized ``data_spec`` dicts."""

from typing import Any

from lumilake_server.data_specs import dump_data_spec, parse_data_spec
from lumilake_server.runtime.runtime_ops import RuntimeOp


def wire(op: RuntimeOp) -> dict[str, Any]:
    """The data spec as the dict a worker receives."""
    return dump_data_spec(op.data_spec)


def runtime_op(
    node_id: str, task_type: str, backend: str, model: str, data_spec: dict, *a, **kw
) -> RuntimeOp:
    """A runtime op whose ``data_spec`` is parsed from its dict."""
    spec = parse_data_spec(task_type, data_spec)
    return RuntimeOp(node_id, task_type, backend, model, spec, *a, **kw)
