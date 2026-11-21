import json
from collections.abc import Iterator
from typing import Any

from lumilake import envs
from lumilake.log import Logger, LogLevel, init_child_logger
from lumilake_server.common import project_input_keys
from lumilake_server.ops import (
    InputOp,
    LambdaOp,
    LLMOp,
    Op,
    ParseOp,
)
from lumilake_server.runtime.lambda_task_helpers import FAIL_KEY, failure_of
from lumilake_server.runtime.runtime_graph.outputs import (
    OpOutput,
    op_output,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _holds_marker(value: Any) -> bool:
    """Whether a literal value is a marker, or decodes to a list holding one:
    a grouped value reaches a consumer as the JSON text of its per-row list."""
    if not isinstance(value, str):
        return failure_of(value) is not None
    if FAIL_KEY not in value:
        return False
    try:
        decoded = json.loads(value)
    except ValueError:
        return False
    if isinstance(decoded, list):
        return any(failure_of(item) is not None for item in decoded)
    return failure_of(decoded) is not None


def _carries_markers(
    op: Op, inputs_dict: dict[str, list[str]], cache: dict[str, bool]
) -> bool:
    """Whether an op's output can hold a failed-record marker: a ``ParseOp``
    emits one unless it keeps the raw text, an ``InputOp`` carries one of its
    literal values, and every other op passes on the markers of what it reads,
    except a router branch that drops them (its rule and default branches).
    ``cache`` holds each op's answer for one build, so shared ancestors are
    walked once."""
    if op.id in cache:
        return cache[op.id]
    if isinstance(op, ParseOp) and op.emits_markers:
        carries = True
    elif isinstance(op, InputOp):
        carries = any(_holds_marker(value) for value in inputs_dict.get(op.name, []))
    elif isinstance(op, LambdaOp) and op.clears_markers:
        carries = False
    else:
        carries = any(_carries_markers(inp, inputs_dict, cache) for inp in op.inputs)
    cache[op.id] = carries
    return carries


def _default_output_destination() -> dict[str, Any]:
    if envs.FLOWMESH_OUTPUT_DESTINATION == "http":
        return {"type": "http", "timeoutSec": 3600}
    return {"type": "local"}


class _RuntimeGraphBuilderBase:
    """Shared builder state and read-path helpers for the op-family mixins."""

    def __init__(
        self,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
        schema_cache: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.logger = init_child_logger("RuntimeGraphBuilder", logger, log_level)
        self._schema_cache = schema_cache if schema_cache is not None else {}
        # Per-row VLMs over a retrieval, flattened to one embedding + one
        # inference + a regroup step; readers bind to the regrouped output.
        self._flattened_vlm_ids: set[str] = set()
        # The workflow being built, so every read site describes the same one.
        self._inputs_dict: dict[str, list[str]] = {}
        self._graph_dict: dict[str, Op] = {}
        # Each op's marker answer for one build, so guarded columns of a shared
        # ancestor cost one walk.
        self._marker_cache: dict[str, bool] = {}

    def _create_runtime_op(
        self,
        name: str,
        task_type: str,
        data_spec: dict[str, Any],
        model_spec: dict[str, Any],
        inference_spec: dict[str, Any],
        backend: str,
        model: str,
        dependencies: list[str] | None = None,
        output_spec: dict[str, Any] | None = None,
        condition: dict[str, str] | None = None,
        api_spec: dict[str, Any] | None = None,
        skip_when_all_inputs_empty: bool = False,
        output_enforcement: dict[str, Any] | None = None,
    ) -> RuntimeOp:
        data_spec = self._attach_lumid_cfg(data_spec)
        return RuntimeOp(
            node_id=name,
            task_type=task_type,
            backend=backend,
            model=model,
            data_spec=data_spec,
            model_spec=model_spec,
            inference_spec=inference_spec,
            api_spec=api_spec or {},
            output_enforcement=output_enforcement,
            dependencies=tuple(dependencies or []),
            output_spec=output_spec,
            condition=condition,
            skip_when_all_inputs_empty=skip_when_all_inputs_empty,
        )

    def _build_model_spec(
        self,
        config: Any,
        backend: str = "vllm",
        backend_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = {
            "source": {
                "type": "huggingface",
                "identifier": config.model,
                "revision": "main",
            }
        }
        overlay = config.engine_overlay()
        if backend == "vllm":
            vllm_cfg = backend_config or self._build_default_vllm_backend_config()
            vllm_cfg.update(overlay)
            spec["vllm"] = vllm_cfg
        elif backend == "transformers":
            spec["transformers"] = backend_config or {
                "mode": "visual-embedding",
                "device_map": "auto",
                "trust_remote_code": True,
            }
        elif backend == "diffusers":
            diffusers_cfg = backend_config or {"dtype": "bf16", "use_safetensors": True}
            # Only dtype is a typed engine field meaningful for diffusers;
            # the rest are vLLM-only and would be silently dropped here.
            invalid = sorted(
                name
                for name in (
                    "max_model_len",
                    "gpu_memory_utilization",
                    "tensor_parallel_size",
                )
                if getattr(config, name) is not None
            )
            if invalid:
                raise ValueError(
                    f"diffusers backend does not accept typed engine fields: {invalid}"
                )
            diffusers_cfg.update(overlay)
            spec["diffusers"] = diffusers_cfg
        return spec

    def _build_default_vllm_backend_config(
        self, enable_mm_embeds: bool = False
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "max_num_batched_tokens": envs.LUMILAKE_VLLM_MAX_NUM_BATCHED_TOKENS,
            "max_cudagraph_capture_size": envs.LUMILAKE_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE,
            "gpu_memory_utilization": envs.LUMILAKE_VLLM_GPU_MEMORY_UTILIZATION,
            "trust_remote_code": True,
            "env_vars": {"VLLM_ATTENTION_BACKEND": "FLASHINFER"},
        }
        if envs.LUMILAKE_VLLM_MAX_MODEL_LEN > 0:
            cfg["max_model_len"] = envs.LUMILAKE_VLLM_MAX_MODEL_LEN
        if enable_mm_embeds:
            cfg["enable_mm_embeds"] = True
            cfg["limit_mm_per_prompt"] = {"image": 1}
        return cfg

    def _build_output_spec(
        self,
        destination: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "destination": destination or _default_output_destination(),
            "artifacts": artifacts or ["results.json", "logs"],
        }

    def _attach_lumid_cfg(self, data_spec: dict[str, Any]) -> dict[str, Any]:
        if data_spec.get("type") != "list" or "lumid_cfg" in data_spec:
            return data_spec
        items = data_spec.get("items")
        if not isinstance(items, list):
            return data_spec
        has_s3 = any(
            isinstance(item, str) and item.startswith("s3://") for item in items
        )
        if not has_s3:
            return data_spec
        if not envs.LUMID_DATA_URL:
            raise ValueError(
                "LUMID_DATA_URL is required for s3:// list inputs (see .env.example)"
            )
        if not envs.LUMID_DATA_TOKEN:
            raise ValueError(
                "LUMID_DATA_TOKEN is required for s3:// list inputs (see .env.example)"
            )
        updated = dict(data_spec)
        updated["lumid_cfg"] = {
            "lumid_data_url": envs.LUMID_DATA_WORKER_URL or envs.LUMID_DATA_URL,
            "lumid_data_token": envs.LUMID_DATA_TOKEN,
        }
        return updated

    def _node_ref_column(
        self,
        *,
        consumer_id: str,
        label: str,
        node_ref: str,
        path: str,
        upstream: Op | None,
        inputs_dict: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Bind a ``node:`` reference into a spec column, mapping the declared
        path to the upstream's read path."""
        if isinstance(upstream, InputOp):
            values = inputs_dict.get(upstream.name)
            if values is None:
                raise ValueError(
                    f"Column '{label}' of consumer '{consumer_id}' references"
                    f" InputOp '{node_ref}' but no values were supplied for"
                    " that input."
                )
            # InputOps never materialize a runtime envelope, so a non-empty
            # ``path`` (which would drill into the envelope) is meaningless
            # here. The scalar value is inlined verbatim; ``path`` must be
            # absent / empty.
            if path not in (None, ""):
                raise ValueError(
                    f"Column '{label}' of consumer '{consumer_id}' references"
                    f" InputOp '{node_ref}' with drill path {path!r}; "
                    "InputOp-derived columns cannot be drilled — wrap "
                    "record-shaped inputs in an upstream op that exposes the "
                    "target field, or drop the ``path`` for scalar inputs."
                )
            return {
                "label": label,
                "data": {"type": "list", "items": list(values)},
            }
        if upstream is not None:
            path = self._op_output(upstream).read(
                path, f"Column '{label}' of consumer '{consumer_id}'"
            )
        return {"label": label, "node": node_ref, "path": path}

    def _project_input_field(
        self, values: list[str], path: str, input_name: str
    ) -> list[str]:
        """Project a field path out of literal input values (JSON object
        strings) server-side, so an InputOp field ref binds a literal list."""
        if path != "items.output" and not path.startswith("items.output."):
            raise ValueError(
                f"InputOp '{input_name}' field path must start with"
                f" 'items.output', got '{path}'"
            )
        if path == "items.output":
            return list(values)
        keys = path[len("items.output.") :].split(".")
        projected = project_input_keys(values, keys, path, input_name)
        return [
            value if isinstance(value, str) else json.dumps(value)
            for value in projected
        ]

    def _collect_graph_template_dependencies(
        self, template_spec: dict[str, Any]
    ) -> list[str]:
        deps: list[str] = []
        seen: set[str] = set()

        def add_dep(node_id: Any) -> None:
            if not isinstance(node_id, str) or node_id in seen:
                return
            seen.add(node_id)
            deps.append(node_id)

        def visit_column(column: Any) -> None:
            if not isinstance(column, dict):
                return
            add_dep(column.get("node"))
            data = column.get("data")
            if not isinstance(data, dict):
                return
            if data.get("type") == "dataframe":
                for nested in data.get("columns", []) or []:
                    visit_column(nested)
            elif data.get("type") == "graph_template":
                nested_template = data.get("template")
                if isinstance(nested_template, dict):
                    for nested in nested_template.get("columns", []) or []:
                        visit_column(nested)

        for column in template_spec.get("columns", []) or []:
            visit_column(column)
        return deps

    @staticmethod
    def _iter_runtime_refs(node: RuntimeOp) -> Iterator[str]:
        """Yield every runtime node reference in a RuntimeOp: data_spec
        column/data-source/list-item refs (a mapping with ``node`` plus
        ``path``, or ``expr`` outside literal list items), python plan stage
        entries, and the condition node. Literal payloads (``node`` without
        ``path``/``expr``) are data, not references."""
        yield from _RuntimeGraphBuilderBase._iter_spec_refs(node.data_spec)
        condition = node.condition
        if condition is not None:
            cond_node = condition.get("node")
            if isinstance(cond_node, str):
                yield cond_node

    @staticmethod
    def _iter_spec_refs(value: Any, allow_expr: bool = True) -> Iterator[str]:
        """Yield node references from a data_spec, schema-aware: a mapping
        with a string ``node`` and a ``path`` key is a reference; ``expr``
        counts only outside literal list items. Recurse into nested mappings
        and lists, disabling ``expr`` below a ``type: list`` payload."""
        if isinstance(value, dict):
            node = value.get("node")
            if isinstance(node, str) and (
                "path" in value or (allow_expr and "expr" in value)
            ):
                yield node
                return
            if value.get("type") == "list":
                yield from _RuntimeGraphBuilderBase._iter_spec_refs(
                    value.get("items"), allow_expr=False
                )
                return
            if value.get("kind") == "stage":
                stage_node = value.get("node")
                if isinstance(stage_node, str):
                    yield stage_node
                    return
            for v in value.values():
                yield from _RuntimeGraphBuilderBase._iter_spec_refs(v, allow_expr)
        elif isinstance(value, list):
            for v in value:
                yield from _RuntimeGraphBuilderBase._iter_spec_refs(v, allow_expr)

    def _guard_single_row_node_ref(
        self,
        *,
        consumer_id: str,
        upstream_id: str,
        upstream: Op | None,
        kind: str,
    ) -> None:
        """Fail closed when a consumer binds a single ``${node.path}`` reference
        to a multi-row upstream."""
        if not isinstance(upstream, LLMOp):
            return
        row_count = self._op_output(upstream).row_count
        if row_count is not None and row_count > 1:
            raise ValueError(
                f"Op '{consumer_id}' {kind} references '{upstream_id}',"
                " which produces multiple rows; a single node reference can"
                " only carry one row, so wiring this to the single unsuffixed"
                " output would silently drop every row but the first."
            )

    def _guard_marked_column(
        self,
        *,
        consumer_id: str,
        label: str,
        upstream: Op | None,
    ) -> None:
        """Refuse a model-op column that reads a source carrying failed-record
        markers, which the model would render as an ordinary value."""
        if upstream is None or not _carries_markers(
            upstream, self._inputs_dict, self._marker_cache
        ):
            return
        raise ValueError(
            f"Column '{label}' of model op '{consumer_id}' reads '{upstream.id}',"
            " which carries failed-record markers; a model op would render a"
            " marker as an ordinary value. Read a router's rule or default"
            " branch, so failed records never reach the model."
        )

    def _op_output(self, op: Op) -> OpOutput:
        """How a consumer reads the op's values."""
        return op_output(
            op, self._flattened_vlm_ids, self._inputs_dict, self._graph_dict
        )
