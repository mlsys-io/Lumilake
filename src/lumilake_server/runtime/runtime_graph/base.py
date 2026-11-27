import json
from collections.abc import Iterator, Sequence
from typing import Any

from lumilake import envs
from lumilake.log import Logger, LogLevel, init_child_logger
from lumilake_server.common import project_input_keys
from lumilake_server.data_specs import (
    Column,
    ColumnTable,
    DataFrame,
    DataSpecModel,
    GraphTemplate,
    GraphTemplateData,
    ListData,
    NodeColumn,
    NodeRead,
    PythonData,
    RetrievalData,
    S3ProfileData,
    SqlProfileData,
)
from lumilake_server.ops import (
    DataOp,
    InputOp,
    LLMOp,
    Op,
)
from lumilake_server.runtime.runtime_graph.outputs import (
    OpOutput,
    op_output,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _default_output_destination() -> dict[str, Any]:
    if envs.FLOWMESH_OUTPUT_DESTINATION == "http":
        return {"type": "http", "timeoutSec": 3600}
    return {"type": "local"}


def _iter_literal_item_refs(items: list[Any] | str | None) -> Iterator[str]:
    """Yield the node reads FlowMesh performs inside literal list items: an
    item mapping whose ``node`` comes with ``path``. A ``node`` alone, or with
    ``expr``, is ordinary data."""
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and "path" in item:
                node = item.get("node")
                if isinstance(node, str):
                    yield node


def _iter_columns(columns: Sequence[Column]) -> Iterator[str]:
    """Yield the node each column reads, recursing into its nested table."""
    for column in columns:
        if column.node is not None:
            yield column.node
        data = column.data
        if isinstance(data, DataFrame):
            yield from _iter_columns(data.columns)
        elif isinstance(data, ListData):
            yield from _iter_literal_item_refs(data.items)


def _iter_data_spec_refs(spec: DataSpecModel) -> Iterator[str]:
    """Yield every node a typed data_spec reads: column reads, a list source,
    a graph template's columns, a retrieval parameter, or a plan stage."""
    if isinstance(spec, ListData):
        if spec.node is not None:
            yield spec.node
        yield from _iter_literal_item_refs(spec.items)
    elif isinstance(spec, DataFrame):
        yield from _iter_columns(spec.columns)
        if isinstance(spec.seeds, NodeRead):
            yield spec.seeds.node
    elif isinstance(spec, GraphTemplateData):
        yield from _iter_columns(spec.template.columns)
        if spec.image_embedding is not None:
            yield spec.image_embedding.node
    elif isinstance(spec, (RetrievalData, SqlProfileData, S3ProfileData)):
        for param in spec.params:
            if param.node is not None:
                yield param.node
            data = param.data
            if data is not None:
                if data.node is not None:
                    yield data.node
                yield from _iter_literal_item_refs(data.items)
    elif isinstance(spec, PythonData):
        for entry in spec.plan:
            if entry.kind == "stage" and entry.node is not None:
                yield entry.node


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

    def _create_runtime_op(
        self,
        name: str,
        task_type: str,
        data_spec: DataSpecModel,
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
        source_op: Op | None = None,
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
            samples=source_op.samples if source_op is not None else False,
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

    def _attach_lumid_cfg(self, data_spec: DataSpecModel) -> DataSpecModel:
        """Give an s3-backed list spec the lumid-data endpoint it reads through."""
        if not isinstance(data_spec, ListData) or data_spec.lumid_cfg is not None:
            return data_spec
        items = data_spec.items
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
        return data_spec.model_copy(
            update={
                "lumid_cfg": {
                    "lumid_data_url": envs.LUMID_DATA_WORKER_URL or envs.LUMID_DATA_URL,
                    "lumid_data_token": envs.LUMID_DATA_TOKEN,
                }
            }
        )

    def _node_ref_column(
        self,
        *,
        consumer_id: str,
        column: NodeColumn,
        upstream: Op | None,
        inputs_dict: dict[str, list[str]],
    ) -> Column:
        """Bind a node column into a spec column at the upstream's read path."""
        label = column.label
        node_ref = column.node
        path = column.path or ""
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
            if column.path:
                raise ValueError(
                    f"Column '{label}' of consumer '{consumer_id}' references InputOp"
                    f" '{node_ref}' with drill path {column.path!r}; InputOp-derived"
                    " columns cannot be drilled — wrap record-shaped inputs in an"
                    " upstream op that exposes the target field, or drop the ``path``"
                    " for scalar inputs."
                )
            return Column(label=label, data=ListData(items=list(values)))
        if isinstance(upstream, DataOp):
            # A DataOp has no runtime envelope, so its values bind whole and a path
            # would read something the op never produces.
            if column.path is not None:
                raise ValueError(
                    f"Column '{label}' of consumer '{consumer_id}' references DataOp"
                    f" '{node_ref}' with drill path {column.path!r}; a DataOp-derived"
                    " column carries the op's values whole, so it takes no path."
                )
            return Column(label=label, data=ListData(items=upstream.data.copy()))
        if upstream is not None:
            path = self._op_output(upstream).read(
                path, f"Column '{label}' of consumer '{consumer_id}'"
            )
        return Column(label=label, node=node_ref, path=path)

    def _mark_literal_column_producers_visited(
        self, columns: ColumnTable | None, visited_node_ids: set[str]
    ) -> None:
        """Mark each literal column producer visited, so the orphan check counts it."""
        for column in columns or []:
            if isinstance(column, NodeColumn) and isinstance(
                self._graph_dict.get(column.node), (InputOp, DataOp)
            ):
                visited_node_ids.add(column.node)

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
        self, template_spec: GraphTemplate
    ) -> list[str]:
        """The runtime node ids a rendered graph template reads, in template order."""
        deps: list[str] = []
        seen: set[str] = set()

        def visit_column(column: Column) -> None:
            node = column.node
            if node is not None and node not in seen:
                seen.add(node)
                deps.append(node)
            if isinstance(column.data, DataFrame):
                for nested in column.data.columns:
                    visit_column(nested)

        for column in template_spec.columns:
            visit_column(column)
        return deps

    @staticmethod
    def _iter_runtime_refs(node: RuntimeOp) -> Iterator[str]:
        """Yield every runtime node reference a RuntimeOp carries: the reads of
        its data_spec, the stage entries of its python plan, and its condition."""
        yield from _iter_data_spec_refs(node.data_spec)
        condition = node.condition
        if condition is not None:
            cond_node = condition.get("node")
            if isinstance(cond_node, str):
                yield cond_node

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

    def _op_output(self, op: Op) -> OpOutput:
        """How a consumer reads the op's values."""
        return op_output(
            op, self._flattened_vlm_ids, self._inputs_dict, self._graph_dict
        )
