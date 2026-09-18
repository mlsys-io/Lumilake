import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self

import sqlparse
from lumilake import envs
from lumilake.log import Logger, LogLevel, init_child_logger
from pydantic import BaseModel, ConfigDict, Field
from sqlparse.sql import TokenList
from sqlparse.tokens import Keyword

from lumilake_server.common import ApiConfig
from lumilake_server.graphs import CompiledGraph
from lumilake_server.ops import (
    DataOp,
    DataRetrievalOp,
    FormatOp,
    InputOp,
    LambdaOp,
    LLMOp,
    MessageOp,
    Op,
    OutputOp,
)
from lumilake_server.ops.embedding_ops import EmbeddingOp
from lumilake_server.ops.llm_ops import ImageGenerationOp, LLMChatOp, LLMVisionOp
from lumilake_server.runtime.flowmesh_client import (
    is_api_origin_trusted,
    resolve_api_credential,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp, RuntimeOpSchema
from lumilake_server.runtime.sensitive import redact_sensitive
from lumilake_server.utils.data_profile_offload import (
    _build_sample_data_profile_queries,
    _type_default_sample,
)
from lumilake_server.utils.func_serialization import safe_materialize_function
from lumilake_server.utils.graph import topological_sort
from lumilake_server.utils.lumid_data_client import (
    retrieve_sample as lumid_retrieve_sample,
)

_PLACEHOLDER_RE = re.compile(r"\$\{([^}.]+)\.([^}]+)\}")


class RuntimeGraphSchema(BaseModel):
    """Pydantic schema for a serialized ``RuntimeGraph``.

    The dataclass below stays the in-process representation. This model
    is used at boundaries (HTTP, IPC) to enforce field shape, types,
    required fields, and reject unknown keys.
    """

    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, RuntimeOpSchema]
    node_order: list[str]
    output_node_map: dict[str, str] = Field(default_factory=dict)
    output_paths: dict[str, str] = Field(default_factory=dict)
    dsl_to_runtime: dict[str, list[str]] = Field(default_factory=dict)


class Roles(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def _default_output_destination() -> dict[str, Any]:
    if envs.FLOWMESH_OUTPUT_DESTINATION == "http":
        return {"type": "http", "timeoutSec": 3600}
    return {"type": "local"}


def _sanitize_node_prefix(prefix: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_")
    return safe or "graph"


_DEFAULT_API_URL = "https://lum.id/llm/v1/chat/completions"


def make_node_prefix(name: str) -> str:
    safe = _sanitize_node_prefix(name)
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"


def _default_sql_sampler(query: str) -> list[Any]:
    return lumid_retrieve_sample(query)


def _default_s3_sampler(prefix: str) -> list[str]:
    # The default sampler intentionally rejects so the caller raises the
    # sample-value error. Live S3 sampling is opt-in via a test or deploy
    # override of ``_RuntimeProfileSamplers.s3``.
    raise NotImplementedError(
        "S3 upstream sample requires 'sample_value' in the upstream data_spec"
    )


@dataclass
class _RuntimeProfileSamplerRegistry:
    sql: Any = field(default=_default_sql_sampler)
    s3: Any = field(default=_default_s3_sampler)


_RuntimeProfileSamplers = _RuntimeProfileSamplerRegistry()


def _inline_single_value_list_params(
    template: str,
    params: list[Any],
) -> tuple[str, list[Any]]:
    """Render single-element list params directly into the SQL/S3 template."""
    rendered = template
    remaining: list[Any] = []
    for param in params:
        if not isinstance(param, dict):
            remaining.append(param)
            continue
        label = param.get("label")
        data = param.get("data")
        if (
            not isinstance(label, str)
            or not isinstance(data, dict)
            or data.get("type") != "list"
        ):
            remaining.append(param)
            continue
        items = data.get("items")
        if not isinstance(items, list) or len(items) != 1:
            remaining.append(param)
            continue
        placeholder = "{" + label + "}"
        if placeholder not in rendered:
            remaining.append(param)
            continue
        value = str(items[0]).replace("'", "''")
        rendered = rendered.replace(placeholder, value)
    return rendered, remaining


@dataclass
class RuntimeGraph:
    nodes: dict[str, RuntimeOp]
    node_order: list[str]
    output_node_map: dict[str, str]
    output_paths: dict[str, str] = field(default_factory=dict)
    dsl_to_runtime: dict[str, list[str]] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def serialize(self) -> dict[str, Any]:
        payload = RuntimeGraphSchema(
            nodes={
                nid: RuntimeOpSchema.model_validate(op.serialize())
                for nid, op in self.nodes.items()
            },
            node_order=self.node_order,
            output_node_map=self.output_node_map,
            output_paths=self.output_paths,
            dsl_to_runtime=self.dsl_to_runtime,
        ).model_dump(exclude_none=True)
        return redact_sensitive(payload)

    @classmethod
    def from_schema(cls, schema: "RuntimeGraphSchema") -> "RuntimeGraph":
        nodes = {
            nid: RuntimeOp.from_schema(nid, op_schema)
            for nid, op_schema in schema.nodes.items()
        }
        missing = [nid for nid in schema.node_order if nid not in nodes]
        if missing:
            raise ValueError(f"node_order references unknown nodes: {missing}")
        return cls(
            nodes=nodes,
            node_order=schema.node_order,
            output_node_map=schema.output_node_map,
            output_paths=schema.output_paths,
            dsl_to_runtime=schema.dsl_to_runtime,
        )

    @classmethod
    def deserialize(cls, payload: Mapping[str, Any]) -> "RuntimeGraph":
        return cls.from_schema(RuntimeGraphSchema.model_validate(payload))

    def to_flowmesh_nodes(self) -> list[dict[str, Any]]:
        # ``RuntimeOp.dependencies`` may reference DSL op ids that aren't
        # materialized as runtime/FlowMesh nodes (e.g. an ``InputOp``
        # consumed only as a retrieval param). FlowMesh schedules tasks
        # whose deps it can resolve — including a non-FlowMesh id in
        # ``dependsOn`` pins the task in PENDING forever. Filter the
        # exported deps to real runtime nodes here. ``topological_order``
        # already preserves ordering across these deps.
        valid_node_ids = set(self.nodes)
        exported: list[dict[str, Any]] = []
        for node_id in self.node_order:
            node_payload = self.nodes[node_id].to_flowmesh_node()
            if "dependsOn" in node_payload:
                filtered = [
                    dep for dep in node_payload["dependsOn"] if dep in valid_node_ids
                ]
                if filtered:
                    node_payload["dependsOn"] = filtered
                else:
                    node_payload.pop("dependsOn", None)
            exported.append(node_payload)
        return exported

    def topological_order(self) -> list[str]:
        # A runtime node's ``dependencies`` may reference DSL-layer op ids
        # that never materialized as runtime nodes (e.g. an ``InputOp``
        # consumed as a retrieval param). We include those in the
        # topological-sort graph so their ordering constraints are
        # respected, then filter the result back down to ids that
        # actually exist in ``self.nodes`` — callers iterate the result
        # expecting to index into ``self.nodes``.
        graph: dict[str, set[str]] = {node_id: set() for node_id in self.nodes}
        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in graph:
                    graph[dep] = set()
                graph[dep].add(node_id)
        order_index = {node_id: idx for idx, node_id in enumerate(self.node_order)}
        return [
            node_id
            for node_id in topological_sort(
                graph,
                secondary_key=lambda node_id: order_index.get(
                    node_id,
                    len(order_index),
                ),
            )
            if node_id in self.nodes
        ]

    def with_node_prefix(self, prefix: str, separator: str = "__") -> Self:
        if not prefix:
            return self

        mapping = {node_id: f"{prefix}{separator}{node_id}" for node_id in self.nodes}

        def _remap_placeholder(text: str, rewritable: dict[str, str]) -> str:
            def _sub(match: re.Match[str]) -> str:
                node = match.group(1)
                prefixed = rewritable.get(node)
                if prefixed is None:
                    return match.group(0)
                return f"${{{prefixed}.{match.group(2)}}}"

            return _PLACEHOLDER_RE.sub(_sub, text)

        def remap(value: Any, rewritable: dict[str, str]) -> Any:
            if isinstance(value, dict):
                updated: dict[str, Any] = {}
                for key, item in value.items():
                    if key == "node" and isinstance(item, str) and item in mapping:
                        updated[key] = mapping[item]
                    else:
                        updated[key] = remap(item, rewritable)
                return updated
            if isinstance(value, list):
                return [remap(item, rewritable) for item in value]
            if isinstance(value, tuple):
                return tuple(remap(item, rewritable) for item in value)
            if isinstance(value, str):
                return _remap_placeholder(value, rewritable)
            return value

        nodes: dict[str, RuntimeOp] = {}
        for old_id, op in self.nodes.items():
            new_id = mapping[old_id]
            rewritable = {
                dep: mapping[dep] for dep in op.dependencies if dep in mapping
            }
            nodes[new_id] = RuntimeOp(
                node_id=new_id,
                task_type=op.task_type,
                backend=op.backend,
                model=op.model,
                data_spec=remap(op.data_spec, rewritable),
                model_spec=remap(op.model_spec, rewritable),
                inference_spec=remap(op.inference_spec, rewritable),
                api_spec=remap(op.api_spec, rewritable),
                dependencies=tuple(mapping.get(dep, dep) for dep in op.dependencies),
                output_spec=(
                    remap(op.output_spec, rewritable)
                    if op.output_spec is not None
                    else None
                ),
                condition=(
                    remap(op.condition, rewritable)
                    if op.condition is not None
                    else None
                ),
            )

        node_order = [mapping[node_id] for node_id in self.node_order]
        output_node_map = {
            mapping[node_id]: output_name
            for node_id, output_name in self.output_node_map.items()
        }
        output_paths = {
            mapping[node_id]: path for node_id, path in self.output_paths.items()
        }
        dsl_to_runtime = {
            op_id: [mapping.get(node_id, node_id) for node_id in runtime_ids]
            for op_id, runtime_ids in self.dsl_to_runtime.items()
        }

        return type(self)(
            nodes=nodes,
            node_order=node_order,
            output_node_map=output_node_map,
            output_paths=output_paths,
            dsl_to_runtime=dsl_to_runtime,
        )


def merge_runtime_graphs(
    graphs: dict[str, RuntimeGraph],
) -> tuple[RuntimeGraph, dict[str, tuple[str, str]]]:
    nodes: dict[str, RuntimeOp] = {}
    node_order: list[str] = []
    output_node_map: dict[str, str] = {}
    output_paths: dict[str, str] = {}
    dsl_to_runtime: dict[str, list[str]] = {}
    output_mapping: dict[str, tuple[str, str]] = {}

    for graph_name, graph in graphs.items():
        for node_id in graph.node_order:
            if node_id in nodes:
                raise ValueError(f"Duplicate runtime node id across graphs: {node_id}")
            nodes[node_id] = graph.nodes[node_id]
            node_order.append(node_id)
        for op_id, runtime_ids in graph.dsl_to_runtime.items():
            existing = dsl_to_runtime.get(op_id)
            if existing is None:
                dsl_to_runtime[op_id] = list(runtime_ids)
                continue
            for runtime_id in runtime_ids:
                if runtime_id not in existing:
                    existing.append(runtime_id)
        for node_id, output_name in graph.output_node_map.items():
            output_node_map[node_id] = output_name
            output_mapping[node_id] = (graph_name, output_name)
        for node_id, path in graph.output_paths.items():
            output_paths[node_id] = path

    return (
        RuntimeGraph(
            nodes=nodes,
            node_order=node_order,
            output_node_map=output_node_map,
            output_paths=output_paths,
            dsl_to_runtime=dsl_to_runtime,
        ),
        output_mapping,
    )


class RuntimeGraphBuilder:
    def __init__(
        self,
        logger: Logger | None = None,
        log_level: LogLevel | None = None,
        schema_cache: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.logger = init_child_logger("RuntimeGraphBuilder", logger, log_level)
        self._schema_cache = schema_cache if schema_cache is not None else {}

    def build(
        self,
        compiled_graph: CompiledGraph,
        task_type_override: str | None = None,
        node_prefix: str | None = None,
    ) -> RuntimeGraph:
        graph_dict = compiled_graph.graph.as_dict()
        inputs_dict = compiled_graph.inputs

        visited_node_ids: set[str] = set()
        llm_ops: dict[str, LLMOp] = {}
        retrieval_ops: dict[str, DataRetrievalOp] = {}
        output_source_to_outputop: dict[str, tuple[str, str | None]] = {}
        for op_id, op in graph_dict.items():
            if isinstance(op, LLMOp):
                llm_ops[op_id] = op
                visited_node_ids.add(op_id)
            elif isinstance(op, DataRetrievalOp):
                retrieval_ops[op_id] = op
                visited_node_ids.add(op_id)
            if isinstance(op, OutputOp):
                assert len(op.inputs) == 1, "OutputOp should have exactly one input"
                source = op.inputs[0]
                if not isinstance(source, (LLMOp, DataRetrievalOp)):
                    raise ValueError(
                        f"OutputOp '{op.name}' input must be an LLMOp or "
                        f"DataRetrievalOp (got {type(source).__name__})"
                    )
                visited_node_ids.add(op_id)
                output_source_to_outputop[source.id] = (op.name, op.path)

        if not llm_ops and not retrieval_ops:
            raise ValueError("Graph must contain at least one LLMOp or DataRetrievalOp")

        nodes: dict[str, RuntimeOp] = {}
        node_order: list[str] = []
        output_node_map: dict[str, str] = {}
        output_paths: dict[str, str] = {}
        dsl_to_runtime: dict[str, list[str]] = {}

        for retrieval_op_id, retrieval_op in retrieval_ops.items():
            self._mark_retrieval_upstream_nodes_visited(
                op=retrieval_op,
                graph_dict=graph_dict,
                visited_node_ids=visited_node_ids,
            )
            if task_type_override == "data_profile":
                runtime_op = self._build_data_profile_node_from_data_retrieval_op(
                    retrieval_op_id,
                    retrieval_op,
                    graph_dict,
                    inputs_dict,
                )
                if runtime_op is None:
                    continue
            else:
                runtime_op = self._build_node_from_data_retrieval_op(
                    retrieval_op_id,
                    retrieval_op,
                    graph_dict,
                    inputs_dict,
                )
            nodes[runtime_op.node_id] = runtime_op
            node_order.append(runtime_op.node_id)
            dsl_to_runtime[retrieval_op_id] = [runtime_op.node_id]

            if retrieval_op_id in output_source_to_outputop:
                output_name, path_override = output_source_to_outputop[retrieval_op_id]
                output_node_map[runtime_op.node_id] = output_name
                mode = retrieval_op.data_spec["mode"]
                # Mode-derived defaults match the FlowMesh executor's item
                # shape; agent replays a SQL plan so it emits ``table`` too.
                default_path = {
                    "sql": "items.table",
                    "s3": "items.content",
                    "agent": "items.table",
                }.get(mode, "items.table")
                output_paths[runtime_op.node_id] = path_override or default_path

        for llm_op_id, llm_op in llm_ops.items():
            if task_type_override == "data_profile":
                continue

            runtime_ops: list[RuntimeOp] = []
            mapping: list[str] = []
            output_node_ids: list[str] = [llm_op_id]
            if isinstance(llm_op, LLMVisionOp):
                runtime_ops, embedding_node_id = self._build_vlm_nodes_from_image_op(
                    llm_op_id,
                    llm_op,
                    graph_dict,
                    inputs_dict,
                    visited_node_ids,
                    dsl_to_runtime=dsl_to_runtime,
                    runtime_nodes=nodes,
                )
                mapping = [embedding_node_id, llm_op_id]
            elif task_type_override != "data_profile" and llm_op.config.api is not None:
                upstream_llm_ids, template_spec = self._infer_structural_messages(
                    llm_op_id,
                    graph_dict,
                    inputs_dict,
                    visited_node_ids,
                    dsl_to_runtime=dsl_to_runtime,
                    runtime_nodes=nodes,
                )
                runtime_ops = self._build_api_llm_op(
                    llm_op_id=llm_op_id,
                    llm_op=llm_op,
                    api_config=llm_op.config.api,
                    template_spec=template_spec,
                    upstream_llm_ids=upstream_llm_ids,
                    output_spec=None,
                    condition=(
                        llm_op.condition if isinstance(llm_op, LLMChatOp) else None
                    ),
                    graph_dict=graph_dict,
                    inputs_dict=inputs_dict,
                    dsl_to_runtime=dsl_to_runtime,
                )
                mapping = [runtime_op.node_id for runtime_op in runtime_ops]
                output_node_ids = mapping
            else:
                runtime_ops = [
                    self._build_node_from_llm_op(
                        llm_op_id,
                        llm_op,
                        graph_dict,
                        inputs_dict,
                        visited_node_ids,
                        task_type_override=task_type_override,
                        dsl_to_runtime=dsl_to_runtime,
                        runtime_nodes=nodes,
                    )
                ]
                mapping = [llm_op_id]

            for runtime_op in runtime_ops:
                if runtime_op.node_id in nodes:
                    continue
                nodes[runtime_op.node_id] = runtime_op
                node_order.append(runtime_op.node_id)

            dsl_to_runtime[llm_op_id] = mapping

            if llm_op_id in output_source_to_outputop:
                output_name, path_override = output_source_to_outputop[llm_op_id]
                for output_node_id in output_node_ids:
                    output_node_map[output_node_id] = output_name
                    if path_override:
                        output_paths[output_node_id] = path_override

        all_node_ids = set(graph_dict.keys())
        unvisited_node_ids = all_node_ids - visited_node_ids

        if task_type_override != "data_profile" and unvisited_node_ids:
            raise ValueError(
                f"Graph transformation failed: {len(unvisited_node_ids)} node(s) were"
                " not visited during traversal, indicating orphaned or unreachable"
                f" ops. Unvisited node IDs: {sorted(unvisited_node_ids)}"
            )

        runtime_graph = RuntimeGraph(
            nodes=nodes,
            node_order=node_order,
            output_node_map=output_node_map,
            output_paths=output_paths,
            dsl_to_runtime=dsl_to_runtime,
        )
        runtime_graph.node_order = runtime_graph.topological_order()
        if node_prefix:
            runtime_graph = runtime_graph.with_node_prefix(
                make_node_prefix(node_prefix)
            )
        return runtime_graph

    def _mark_retrieval_upstream_nodes_visited(
        self,
        *,
        op: DataRetrievalOp,
        graph_dict: dict[str, Op],
        visited_node_ids: set[str],
    ) -> None:
        # Retrieval nodes can consume InputOp values that are not represented as
        # runtime dependencies. Mark those upstream op ids as visited so strict
        # traversal validation does not flag legitimate input-only branches.
        for input_op in op.inputs:
            visited_node_ids.add(input_op.id)

        spec = op.data_spec or {}
        params = spec.get("params")
        if not isinstance(params, list):
            return
        for param in params:
            if not isinstance(param, dict):
                continue
            node = param.get("node")
            if isinstance(node, str) and node in graph_dict:
                visited_node_ids.add(node)

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
            dependencies=tuple(dependencies or []),
            output_spec=output_spec,
            condition=condition,
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
            "encoding": "utf-8",
        }
        self.logger.warning(
            "Workflow uses 'type: list' with s3:// items; this path requires the "
            "FlowMesh worker to support lumid_cfg. If your FlowMesh worker does not "
            "yet support lumid_cfg, the workflow will fail at retrieval. "
            "See docs/WORKFLOWS.md."
        )
        return updated

    def _build_vlm_nodes_from_image_op(
        self,
        llm_op_id: str,
        llm_op: LLMVisionOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited_node_ids: set[str],
        dsl_to_runtime: dict[str, list[str]] | None = None,
        runtime_nodes: dict[str, RuntimeOp] | None = None,
    ) -> tuple[list[RuntimeOp], str]:
        if llm_op.rowwise_template:
            upstream_llm_ids, _ = self._infer_structural_messages(
                llm_op_id,
                graph_dict,
                inputs_dict,
                visited_node_ids,
                dsl_to_runtime=dsl_to_runtime,
                runtime_nodes=runtime_nodes,
            )
            columns: list[dict[str, Any]] = []
            for col in llm_op.rowwise_columns or []:
                label = col.get("label")
                data = col.get("data")
                node_ref = col.get("node")
                path = col.get("path")
                if isinstance(label, str) and isinstance(data, dict):
                    columns.append({"label": label, "data": data})
                    continue
                if (
                    isinstance(label, str)
                    and isinstance(node_ref, str)
                    and isinstance(path, str)
                ):
                    columns.append(
                        self._node_ref_column(
                            consumer_id=llm_op_id,
                            label=label,
                            node_ref=node_ref,
                            path=path,
                            upstream=graph_dict.get(node_ref),
                            inputs_dict=inputs_dict,
                            graph_dict=graph_dict,
                            dsl_to_runtime=dsl_to_runtime,
                            kind="rowwise column",
                        )
                    )
            if not columns:
                raise ValueError(f"LLMVisionOp {llm_op_id} has empty rowwise_columns")

            messages: list[dict[str, str]] = []
            for system_msg in llm_op.system_messages or []:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": llm_op.rowwise_template})
            template_spec: dict[str, Any] = {
                "name": "format",
                "columns": columns,
                "options": {"format": {"messages": messages}},
            }
        else:
            upstream_llm_ids, template_spec = self._infer_structural_messages(
                llm_op_id,
                graph_dict,
                inputs_dict,
                visited_node_ids,
                dsl_to_runtime=dsl_to_runtime,
                runtime_nodes=runtime_nodes,
            )

        image_source_id = llm_op.image_source
        visited_node_ids.add(image_source_id)
        (
            embedding_data_spec,
            batch_column,
            source_dependencies,
        ) = self._resolve_vlm_image_source(
            llm_op=llm_op,
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
        )
        template_spec["columns"] = list(template_spec.get("columns") or []) + [
            batch_column
        ]

        embedding_node_id = f"{llm_op_id}_embedding"
        model_spec = self._build_model_spec(llm_op.config, "transformers")
        embedding_node = self._create_runtime_op(
            name=embedding_node_id,
            task_type="embedding",
            data_spec=embedding_data_spec,
            model_spec=model_spec,
            inference_spec={},
            backend="transformers",
            model=llm_op.config.model,
            dependencies=source_dependencies,
            output_spec={
                "destination": _default_output_destination(),
                "artifacts": ["results.json", "visual_embeddings.pt"],
            },
        )

        inference_spec = llm_op.config.inference_spec()
        backend_config = self._build_default_vllm_backend_config(enable_mm_embeds=True)

        template_dependencies = self._collect_graph_template_dependencies(template_spec)
        dependencies = list(upstream_llm_ids) if upstream_llm_ids else []
        dependencies.extend(template_dependencies)
        dependencies.append(embedding_node_id)
        deduped_deps = []
        seen = set()
        for dep in dependencies:
            if dep in seen:
                continue
            seen.add(dep)
            deduped_deps.append(dep)

        vlm_node = self._create_runtime_op(
            name=llm_op_id,
            task_type="inference",
            data_spec={
                "type": "graph_template",
                "template": template_spec,
                "image_embedding": {
                    "node": embedding_node_id,
                    "path": "embedding_file",
                },
            },
            model_spec=self._build_model_spec(llm_op.config, "vllm", backend_config),
            inference_spec=inference_spec,
            backend="vllm",
            model=llm_op.config.model,
            dependencies=deduped_deps,
        )

        return [embedding_node, vlm_node], embedding_node_id

    def _resolve_vlm_image_source(
        self,
        llm_op: LLMVisionOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        image_source_id = llm_op.image_source
        if image_source_id not in graph_dict:
            raise ValueError(
                f"LLMVisionOp {llm_op.id} references unknown image source"
                f" '{image_source_id}'."
            )
        image_source_op = graph_dict[image_source_id]
        if isinstance(image_source_op, InputOp):
            if image_source_op.name not in inputs_dict:
                raise ValueError(
                    f"Missing inputs for image source '{image_source_op.name}'."
                )
            image_items = inputs_dict[image_source_op.name]
            if not isinstance(image_items, list):
                raise ValueError(
                    f"Image inputs for '{image_source_op.name}' must be a list."
                )
            embedding_data_spec: dict[str, Any] = {
                "type": "list",
                "items": image_items,
            }
            batch_column: dict[str, Any] = {
                "label": f"{image_source_op.id}_batch",
                "data": {
                    "type": "list",
                    "items": ["" for _ in range(len(image_items))],
                },
            }
            return embedding_data_spec, batch_column, []

        if isinstance(image_source_op, ImageGenerationOp):
            resolved_path = (
                "items.image" if llm_op.image_path == "images" else llm_op.image_path
            )
            embedding_data_spec = {
                "type": "list",
                "node": image_source_id,
                "path": resolved_path,
            }
            batch_column = {
                "label": f"{image_source_id}_batch",
                "node": image_source_id,
                "path": resolved_path,
            }
            return embedding_data_spec, batch_column, [image_source_id]

        if isinstance(image_source_op, DataRetrievalOp):
            image_path = (
                llm_op.image_path if llm_op.image_path != "images" else "items.content"
            )
            embedding_data_spec = {
                "type": "list",
                "node": image_source_id,
                "path": image_path,
            }
            batch_column = {
                "label": f"{image_source_id}_batch",
                "node": image_source_id,
                "path": image_path,
            }
            return embedding_data_spec, batch_column, [image_source_id]

        raise ValueError(
            "LLMVisionOp "
            f"{llm_op.id} image source must be an InputOp, ImageGenerationOp, "
            "or DataRetrievalOp "
            f"(got {type(image_source_op).__name__})."
        )

    def _build_node_from_data_retrieval_op(
        self,
        op_id: str,
        op: DataRetrievalOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        data_spec_override: dict[str, Any] | None = None,
    ) -> RuntimeOp:
        spec = (
            data_spec_override
            if data_spec_override is not None
            else (op.data_spec or {})
        )
        if spec.get("type") != "lumid":
            raise ValueError(
                f"DataRetrievalOp {op_id} requires type: lumid "
                f"(got {spec.get('type')!r})"
            )
        mode = spec["mode"]
        if mode not in {"sql", "s3", "agent"}:
            raise ValueError(
                f"DataRetrievalOp {op_id} mode must be 'sql', 's3', or 'agent' "
                f"(got {mode!r})"
            )
        if not envs.LUMID_DATA_URL:
            raise ValueError(
                f"DataRetrievalOp {op_id} requires LUMID_DATA_URL to be configured "
                "(see .env.example)"
            )
        if not envs.LUMID_DATA_TOKEN:
            raise ValueError(
                f"DataRetrievalOp {op_id} requires LUMID_DATA_TOKEN to be configured "
                "(see .env.example)"
            )

        params = spec.get("params") or []
        if not isinstance(params, list):
            raise ValueError(f"DataRetrievalOp {op_id} params must be a list")

        template: str | None = None
        if mode in {"sql", "s3"}:
            template = spec.get("template")
            if not isinstance(template, str):
                raise ValueError(
                    f"DataRetrievalOp {op_id} (mode={mode}) requires template"
                )
        else:
            description = spec.get("description")
            schema_scope = spec.get("schema_scope")
            if not isinstance(description, str):
                raise ValueError(
                    f"DataRetrievalOp {op_id} (mode=agent) requires description"
                )
            if schema_scope is not None and not isinstance(schema_scope, str):
                raise ValueError(
                    f"DataRetrievalOp {op_id} (mode=agent) "
                    "schema_scope must be a string"
                )
            template = description

        dependencies: list[str] = []
        seen: set[str] = set()
        for input_op in op.inputs:
            if isinstance(input_op, InputOp):
                continue
            if input_op.id in seen:
                continue
            seen.add(input_op.id)
            dependencies.append(input_op.id)

        # A template param with ``node: <InputOp.id>`` can't be resolved
        # by a FlowMesh worker — InputOps aren't dispatched as tasks, so
        # the upstream lookup would return null. Substitute those params
        # inline with the concrete input values from ``inputs_dict``
        # before the spec reaches the worker. Non-InputOp node refs
        # (real SQL/retrieval upstreams) keep their node pointer and
        # become genuine task dependencies.
        resolved_params: list[Any] = []
        for param in params:
            if not isinstance(param, dict):
                resolved_params.append(param)
                continue
            node = param.get("node")
            if isinstance(node, str):
                upstream = graph_dict.get(node)
                if isinstance(upstream, InputOp):
                    values = inputs_dict.get(upstream.name)
                    if values is None:
                        raise ValueError(
                            f"DataRetrievalOp '{op_id}' template param "
                            f"{param.get('label')!r} references InputOp "
                            f"{upstream.name!r} but no values were supplied "
                            "for that input."
                        )
                    path = param.get("path")
                    # InputOps never materialize a runtime envelope, so a
                    # non-empty ``path`` (which would drill into the
                    # envelope) is meaningless here. The scalar value is
                    # inlined verbatim; ``path`` must be absent / empty.
                    if path not in (None, ""):
                        raise ValueError(
                            f"DataRetrievalOp '{op_id}' template param "
                            f"{param.get('label')!r} references InputOp "
                            f"{upstream.name!r} with drill path {path!r}; "
                            "InputOp-derived params cannot be drilled — "
                            "wrap record-shaped inputs in an upstream op "
                            "that exposes the target field, or drop the "
                            "``path`` for scalar inputs."
                        )
                    literal_param: dict[str, Any] = {
                        "label": param.get("label"),
                        "data": {"type": "list", "items": list(values)},
                    }
                    resolved_params.append(literal_param)
                    continue
                if isinstance(upstream, LLMOp):
                    row_count = self._static_output_row_count(
                        upstream, inputs_dict, graph_dict
                    )
                    if row_count is not None and row_count > 1:
                        raise ValueError(
                            f"DataRetrievalOp '{op_id}' template param "
                            f"{param.get('label')!r} references '{node}', which"
                            " produces multiple rows; a retrieval param binds a"
                            " single node reference, so wiring this to the"
                            " unsuffixed output would silently drop every row"
                            " but the first."
                        )
                if node not in seen:
                    seen.add(node)
                    dependencies.append(node)
            resolved_params.append(param)

        if mode in {"sql", "s3"}:
            template, resolved_params = _inline_single_value_list_params(
                template, resolved_params
            )

        data_spec: dict[str, Any] = {
            "type": "lumid",
            "mode": mode,
            "lumid_data_url": envs.LUMID_DATA_WORKER_URL or envs.LUMID_DATA_URL,
            "lumid_data_token": envs.LUMID_DATA_TOKEN,
            "params": resolved_params,
        }
        if mode in {"sql", "s3"}:
            data_spec["template"] = template
        if mode == "sql":
            table = spec.get("table")
            if isinstance(table, str) and table.strip():
                data_spec["table"] = table.strip()
            else:
                try:
                    data_spec["table"] = self._extract_table_from_sql_template(template)
                except ValueError:
                    pass
            output_format = spec.get("output_format", "jsonl")
            data_spec["output_format"] = output_format
        if mode == "s3":
            data_spec["encoding"] = spec.get("encoding", "utf-8")
        if mode == "agent":
            data_spec["description"] = template
            if spec.get("schema_scope"):
                data_spec["schema_scope"] = spec["schema_scope"]
            for optional in ("output_format", "max_steps", "model"):
                if optional in spec:
                    data_spec[optional] = spec[optional]
        if "verify" in spec:
            data_spec["verify"] = spec["verify"]

        return self._create_runtime_op(
            name=op_id,
            task_type="data_retrieval",
            data_spec=data_spec,
            model_spec={},
            inference_spec={},
            backend="data_retrieval",
            model="data_retrieval",
            dependencies=dependencies if dependencies else None,
        )

    def _resolve_profile_param(
        self,
        owner_node_id: str,
        param: Any,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited: set[str],
    ) -> dict[str, Any] | None:
        if not isinstance(param, dict):
            return None
        if "data" in param and isinstance(param.get("data"), dict):
            return param
        label = param.get("label")
        node_id = param.get("node")
        if not isinstance(label, str) or not isinstance(node_id, str):
            return None
        upstream = graph_dict.get(node_id)
        if upstream is None:
            return None
        if isinstance(upstream, InputOp):
            values = inputs_dict.get(upstream.name) or []
            if not values:
                return None
            return {
                "label": label,
                "data": {"type": "list", "items": list(values)},
            }
        if isinstance(upstream, LLMChatOp):
            sample = self._sample_value_from_structural_outputs(
                owner_node_id=owner_node_id,
                label=label,
                path=param.get("path"),
                upstream_id=node_id,
                upstream_op_kind=type(upstream).__name__,
                structural_outputs=upstream.structural_outputs,
            )
            return {
                "label": label,
                "data": {"type": "list", "items": [sample]},
            }
        if isinstance(upstream, DataRetrievalOp):
            sample = self._sample_value_from_upstream_retrieval(
                owner_node_id=owner_node_id,
                label=label,
                path=param.get("path"),
                upstream_id=node_id,
                upstream=upstream,
                graph_dict=graph_dict,
                inputs_dict=inputs_dict,
                visited=visited,
            )
            return {
                "label": label,
                "data": {"type": "list", "items": [sample]},
            }
        raise ValueError(
            "Data profile preflight does not support upstream op kind "
            f"'{type(upstream).__name__}' for placeholder '{label}' at "
            f"node '{owner_node_id}'. Supported upstream kinds: InputOp, "
            "DataRetrievalOp, LLMChatOp with structural_outputs. Add a "
            "'sample_value' to the upstream data_spec, attach "
            "'structural_outputs' to the LLM, or set "
            "LUMILAKE_DISABLE_DATA_PROFILE=1."
        )

    def _build_data_profile_node_from_data_retrieval_op(
        self,
        op_id: str,
        op: DataRetrievalOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited: set[str] | None = None,
    ) -> RuntimeOp | None:
        spec = op.data_spec or {}
        mode = spec["mode"]
        if mode not in {"sql", "s3"}:
            return None
        template = spec.get("template")
        params = spec.get("params") or []
        if not isinstance(template, str):
            raise ValueError(f"DataRetrievalOp {op_id} missing template")
        if not isinstance(params, list):
            raise ValueError(f"DataRetrievalOp {op_id} params must be a list")
        resolver_visited: set[str] = set(visited) if visited is not None else set()
        resolver_visited.add(op_id)
        if mode == "sql":
            resolved_params = [
                resolved
                for param in params
                if (
                    resolved := self._resolve_profile_param(
                        op_id,
                        param,
                        graph_dict,
                        inputs_dict,
                        resolver_visited,
                    )
                )
                is not None
            ]
            constraints = self._build_data_profile_constraints(params, graph_dict)

            table = spec.get("table")
            if not isinstance(table, str):
                table = self._extract_table_from_sql_template(template)
            data_spec: dict[str, Any] = {
                "type": "sql",
                "template": template,
                "params": resolved_params,
                "constraints": constraints,
                "num_test_queries": envs.LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES,
                "table": table,
            }
        else:
            resolved_params = [
                resolved
                for param in params
                if (
                    resolved := self._resolve_profile_param(
                        op_id,
                        param,
                        graph_dict,
                        inputs_dict,
                        resolver_visited,
                    )
                )
                is not None
            ]
            data_spec = {
                "type": "s3",
                "template": template,
                "params": resolved_params,
                "encoding": spec.get("encoding", "utf-8"),
            }
        return self._create_runtime_op(
            name=op_id,
            task_type="data_profiling",
            data_spec=data_spec,
            model_spec={},
            inference_spec={},
            backend="data_profiling",
            model="data_profiling",
            dependencies=None,
        )

    def _validate_embedding_texts(self, op_id: str, items: Sequence[Any]) -> None:
        for idx, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"EmbeddingOp '{op_id}' text input[{idx}] must be a non-empty "
                    f"string (got {text!r})"
                )

    @staticmethod
    def _upstream_output_path(op: Op) -> str:
        """Result path for an upstream op's text output: ``text`` for an
        API task, ``items.output`` otherwise."""
        if RuntimeGraphBuilder._is_api_task(op):
            return "text"
        return "items.output"

    @staticmethod
    def _is_api_task(op: Op) -> bool:
        """Whether an LLMOp dispatches as an API task (not a VLM)."""
        return (
            isinstance(op, LLMOp)
            and not isinstance(op, LLMVisionOp)
            and (op.config.api is not None)
        )

    @staticmethod
    def _fanout_row_ids(
        op_id: str, dsl_to_runtime: dict[str, list[str]] | None
    ) -> list[str]:
        """Runtime node ids an op fanned out into, or ``[op_id]`` when it did
        not fan out. The actual fanout (``dsl_to_runtime``) is authoritative:
        a static row-count estimate misses API rowwise/aggregate fanout that
        is driven by literal column values rather than the input row count.
        Only a mapping that follows the ``<op_id>`` / ``<op_id>__row<i>`` shape
        counts as row fanout; a VLM maps to ``[<id>_embedding, <id>]`` (two
        implementation stages of one logical op), which is not row fanout."""
        if dsl_to_runtime is not None:
            row_ids = dsl_to_runtime.get(op_id)
            if row_ids is not None and len(row_ids) > 1:
                if row_ids == [op_id] + [
                    f"{op_id}__row{i}" for i in range(1, len(row_ids))
                ]:
                    return list(row_ids)
        return [op_id]

    def _guard_single_row_node_ref(
        self,
        *,
        consumer_id: str,
        upstream_id: str,
        upstream: Op | None,
        inputs_dict: dict[str, list[str]],
        graph_dict: dict[str, Op],
        dsl_to_runtime: dict[str, list[str]] | None,
        kind: str,
    ) -> None:
        """Fail closed when a consumer binds a single ``${node.path}`` reference
        to an upstream that produces multiple rows — whether by fanning out into
        several runtime nodes or by emitting several rows from one node (a
        rowwise op). Binding only the unsuffixed row-0 node would silently drop
        every row but the first."""
        if not isinstance(upstream, LLMOp):
            return
        row_count = self._static_output_row_count(upstream, inputs_dict, graph_dict)
        fanned = len(self._fanout_row_ids(upstream_id, dsl_to_runtime)) > 1
        if (row_count is not None and row_count > 1) or fanned:
            raise ValueError(
                f"Op '{consumer_id}' {kind} references '{upstream_id}',"
                " which produces multiple rows; API mode can only carry one row"
                " per node reference, so wiring this to the single unsuffixed"
                " output would silently drop every row but the first."
            )

    def _node_ref_column(
        self,
        *,
        consumer_id: str,
        label: str,
        node_ref: str,
        path: str,
        upstream: Op | None,
        inputs_dict: dict[str, list[str]],
        graph_dict: dict[str, Op],
        dsl_to_runtime: dict[str, list[str]] | None,
        kind: str,
    ) -> dict[str, Any]:
        """Bind a ``node:`` reference into a spec column, guarding multi-row
        upstreams. Not a single chokepoint — callers must call the guard."""
        self._guard_single_row_node_ref(
            consumer_id=consumer_id,
            upstream_id=node_ref,
            upstream=upstream,
            inputs_dict=inputs_dict,
            graph_dict=graph_dict,
            dsl_to_runtime=dsl_to_runtime,
            kind=kind,
        )
        return {"label": label, "node": node_ref, "path": path}

    def _source_row_ids(
        self,
        node: str,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None,
    ) -> list[str]:
        """Runtime row ids a condition source fanned out into. Prefers the
        actual fanout from ``dsl_to_runtime``, but falls back to the static row
        count (which sees rowwise columns) when the source is not yet built —
        the consumer may be built before its condition source, so the fanout
        must be derivable without ``dsl_to_runtime``. Row ids follow the
        deterministic ``<node>`` / ``<node>__row<i>`` pattern. Only an API task
        fans out into multiple runtime nodes; a local rowwise producer emits
        multiple items from a single node, so it has no ``__row<i>`` nodes to
        gate on and is treated as unfanned (single node)."""
        row_ids = self._fanout_row_ids(node, dsl_to_runtime)
        if len(row_ids) > 1:
            return row_ids
        upstream = graph_dict.get(node)
        if upstream is not None and self._is_api_task(upstream):
            count = self._static_output_row_count(upstream, inputs_dict, graph_dict)
            if count is not None and count > 1:
                return [node] + [f"{node}__row{i}" for i in range(1, count)]
        return [node]

    def _row_condition(
        self,
        condition: dict[str, str] | None,
        row_index: int,
        consumer_row_count: int,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None,
    ) -> dict[str, str] | None:
        """Remap a condition's ``node`` to the matching row of a fanned source.
        A condition referencing a node that fanned out must gate row ``i`` of
        the consumer against row ``i`` of the source; leaving it on the
        unsuffixed row-0 node would gate every consumer row against row 0. The
        consumer and source must fan out to the same number of rows — a
        mismatch would silently gate some consumer rows on the wrong source row
        (or escape as an IndexError), so it fails closed instead."""
        if condition is None:
            return None
        node = condition.get("node")
        if node is None:
            return condition
        row_ids = self._source_row_ids(node, graph_dict, inputs_dict, dsl_to_runtime)
        if len(row_ids) <= 1:
            return condition
        if len(row_ids) != consumer_row_count:
            raise ValueError(
                f"Condition node '{node}' fanned out into {len(row_ids)} rows"
                f" but the consumer has {consumer_row_count} rows; a condition"
                " can only gate a fanned consumer when both fan out to the same"
                " number of rows."
            )
        if row_index == 0:
            return condition
        remapped = dict(condition)
        remapped["node"] = row_ids[row_index]
        return remapped

    def _api_prior_prompt(
        self, op: LLMChatOp, inputs_dict: dict[str, list[str]]
    ) -> list[str] | None:
        """Resolve an API-backed op's user message to a literal prior prompt
        (an API result carries no ``metadata.prompt``, so inline the message
        that was sent; only literal content can be inlined). Returns ``None``
        when the prompt is runtime-derived; the caller fails closed in that
        case, because API mode cannot reconstruct the history at dispatch
        time."""
        messages = op.messages.messages if isinstance(op.messages, MessageOp) else []
        user_msgs = [m for m in messages if m.role == "user"]
        if not user_msgs:
            raise ValueError(
                f"LLMChatOp '{op.id}' return_history needs a user message."
            )
        content = user_msgs[-1].content
        if isinstance(content, str):
            return [content]
        if isinstance(content, InputOp):
            values = inputs_dict.get(content.name)
            if not values:
                raise ValueError(
                    f"LLMChatOp '{op.id}' return_history references InputOp"
                    f" '{content.name}' with no values supplied."
                )
            return list(values)
        if isinstance(content, DataOp):
            return list(content.data)
        return None

    def _build_node_from_embedding_op(
        self,
        llm_op_id: str,
        llm_op: EmbeddingOp,
        inputs_dict: dict[str, list[str]],
        visited_node_ids: set[str],
        graph_dict: dict[str, Op] | None = None,
        dsl_to_runtime: dict[str, list[str]] | None = None,
    ) -> RuntimeOp:
        visited_node_ids.add(llm_op_id)
        content_op = llm_op.content
        visited_node_ids.add(content_op.id)

        if isinstance(content_op, InputOp):
            items = inputs_dict.get(content_op.name)
            if not items:
                raise ValueError(
                    f"EmbeddingOp '{llm_op_id}' content references InputOp "
                    f"{content_op.name!r} with no values supplied."
                )
            self._validate_embedding_texts(llm_op_id, items)
            data_spec: dict[str, Any] = {"type": "list", "items": list(items)}
            dependencies: list[str] = []
        elif isinstance(content_op, DataOp):
            if not content_op.data:
                raise ValueError(f"EmbeddingOp '{llm_op_id}' has empty text input")
            self._validate_embedding_texts(llm_op_id, content_op.data)
            data_spec = {"type": "list", "items": list(content_op.data)}
            dependencies = []
        else:
            if graph_dict is not None:
                self._guard_single_row_node_ref(
                    consumer_id=llm_op_id,
                    upstream_id=content_op.id,
                    upstream=content_op,
                    inputs_dict=inputs_dict,
                    graph_dict=graph_dict,
                    dsl_to_runtime=dsl_to_runtime,
                    kind="content",
                )
            data_spec = {
                "type": "list",
                "node": content_op.id,
                "path": self._upstream_output_path(content_op),
            }
            dependencies = [content_op.id]

        # ``model.vllm`` presence routes to the vLLM embedding executor; the
        # executor sets ``runner`` internally, so it is intentionally omitted.
        model_spec = self._build_model_spec(llm_op.config, "vllm", {"convert": "embed"})
        return self._create_runtime_op(
            name=llm_op_id,
            task_type="embedding",
            data_spec=data_spec,
            model_spec=model_spec,
            inference_spec={},
            backend="vllm",
            model=llm_op.config.model,
            dependencies=dependencies or None,
            # Vectors are returned as a safetensors artifact, not inline; the
            # tensor is row-aligned to the input texts.
            output_spec=self._build_output_spec(
                _default_output_destination(),
                ["results.json", "embeddings.safetensors"],
            ),
        )

    def _build_node_from_llm_op(
        self,
        llm_op_id: str,
        llm_op: LLMOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited_node_ids: set[str],
        task_type_override: str | None = None,
        dsl_to_runtime: dict[str, list[str]] | None = None,
        runtime_nodes: dict[str, RuntimeOp] | None = None,
    ) -> RuntimeOp:
        if isinstance(llm_op, EmbeddingOp):
            return self._build_node_from_embedding_op(
                llm_op_id,
                llm_op,
                inputs_dict,
                visited_node_ids,
                graph_dict=graph_dict,
                dsl_to_runtime=dsl_to_runtime,
            )

        if isinstance(llm_op, ImageGenerationOp):
            visited_node_ids.add(llm_op_id)
            visited_node_ids.add(llm_op.content.id)
            inference_spec = {
                "num_inference_steps": 8,
                "guidance_scale": 1.0,
                "height": 1024,
                "width": 1024,
                **llm_op.config.inference_spec(),
            }
            content_op = llm_op.content
            if (
                isinstance(content_op, FormatOp)
                and content_op.template == "{ref0}"
                and len(content_op.inputs) == 1
            ):
                visited_node_ids.add(content_op.id)
                content_op = content_op.inputs[0]
            if isinstance(content_op, InputOp):
                items = inputs_dict.get(content_op.name)
                if items is None:
                    raise ValueError(
                        f"ImageGenerationOp '{llm_op_id}' content references"
                        f" InputOp {content_op.name!r} with no values supplied."
                    )
                content_data_spec: dict[str, Any] = {
                    "type": "list",
                    "items": list(items),
                }
                content_dependencies: list[str] = []
            else:
                self._guard_single_row_node_ref(
                    consumer_id=llm_op_id,
                    upstream_id=content_op.id,
                    upstream=content_op,
                    inputs_dict=inputs_dict,
                    graph_dict=graph_dict,
                    dsl_to_runtime=dsl_to_runtime,
                    kind="content",
                )
                content_data_spec = {
                    "type": "list",
                    "node": content_op.id,
                    "path": self._upstream_output_path(content_op),
                }
                content_dependencies = [content_op.id]
            return self._create_runtime_op(
                name=llm_op_id,
                task_type="omni_text2image",
                data_spec=content_data_spec,
                model_spec=self._build_model_spec(llm_op.config, "omni"),
                inference_spec=inference_spec,
                backend="omni",
                model=llm_op.config.model,
                dependencies=content_dependencies or None,
                output_spec=self._build_output_spec(
                    _default_output_destination(),
                    ["results.json", "images/"],
                ),
            )

        upstream_llm_ids, template_spec = self._infer_structural_messages(
            llm_op_id,
            graph_dict,
            inputs_dict,
            visited_node_ids,
            dsl_to_runtime=dsl_to_runtime,
            runtime_nodes=runtime_nodes,
        )

        if task_type_override == "data_profile":
            task_type = "data_profiling"
            backend = "data_profiling"
        else:
            task_type = task_type_override or "inference"
            backend = "vllm"

        inference_spec = llm_op.config.inference_spec()
        output_spec = None

        if isinstance(llm_op, LLMChatOp) and llm_op.structural_outputs:
            inference_spec["templates"] = llm_op.structural_outputs

        if isinstance(llm_op, LLMChatOp) and llm_op.rowwise_template:
            columns: list[dict[str, Any]] = []
            row_dependencies = list(upstream_llm_ids) if upstream_llm_ids else []
            for col in llm_op.rowwise_columns or []:
                label = col.get("label")
                data = col.get("data")
                node_ref = col.get("node")
                path = col.get("path")
                if isinstance(label, str) and isinstance(data, dict):
                    columns.append({"label": label, "data": data})
                    continue
                if (
                    isinstance(label, str)
                    and isinstance(node_ref, str)
                    and isinstance(path, str)
                ):
                    columns.append(
                        self._node_ref_column(
                            consumer_id=llm_op_id,
                            label=label,
                            node_ref=node_ref,
                            path=path,
                            upstream=graph_dict.get(node_ref),
                            inputs_dict=inputs_dict,
                            graph_dict=graph_dict,
                            dsl_to_runtime=dsl_to_runtime,
                            kind="rowwise column",
                        )
                    )
                    if node_ref not in row_dependencies:
                        row_dependencies.append(node_ref)

            if not columns:
                raise ValueError(f"LLMChatOp {llm_op_id} has empty rowwise_columns")

            messages: list[dict[str, str]] = []
            for system_msg in llm_op.system_messages or []:
                messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": llm_op.rowwise_template})

            return self._create_runtime_op(
                name=llm_op_id,
                task_type=task_type,
                data_spec={
                    "type": "dataframe",
                    "columns": columns,
                    "messages": messages,
                },
                model_spec=self._build_model_spec(llm_op.config, backend),
                inference_spec=inference_spec,
                backend=backend,
                model=llm_op.config.model,
                dependencies=row_dependencies if row_dependencies else None,
                output_spec=output_spec,
                condition=self._row_condition(
                    llm_op.condition,
                    0,
                    1,
                    graph_dict,
                    inputs_dict,
                    dsl_to_runtime,
                ),
            )

        if isinstance(llm_op, LLMChatOp) and llm_op.aggregate_table:
            table_columns: list[dict[str, Any]] = []
            aggregate_dependencies = list(upstream_llm_ids) if upstream_llm_ids else []
            base_columns = template_spec.get("columns", [])
            if not isinstance(base_columns, list):
                raise ValueError(
                    f"LLMChatOp {llm_op_id} aggregate template columns must be a list"
                )
            format_options = template_spec.get("options", {}).get("format", {})
            if not isinstance(format_options, dict):
                raise ValueError(
                    f"LLMChatOp {llm_op_id} aggregate template format options must be a"
                    " dict"
                )
            base_messages = format_options.get("messages", [])
            if not isinstance(base_messages, list):
                raise ValueError(
                    f"LLMChatOp {llm_op_id} aggregate template messages must be a list"
                )
            base_steps = format_options.get("steps", [])
            if not isinstance(base_steps, list):
                raise ValueError(
                    f"LLMChatOp {llm_op_id} aggregate template steps must be a list"
                )
            for col in llm_op.aggregate_table:
                label = col.get("label")
                node_ref = col.get("node")
                path = col.get("path")
                if not (
                    isinstance(label, str)
                    and isinstance(node_ref, str)
                    and isinstance(path, str)
                ):
                    continue
                table_columns.append(
                    self._node_ref_column(
                        consumer_id=llm_op_id,
                        label=label,
                        node_ref=node_ref,
                        path=path,
                        upstream=graph_dict.get(node_ref),
                        inputs_dict=inputs_dict,
                        graph_dict=graph_dict,
                        dsl_to_runtime=dsl_to_runtime,
                        kind="aggregate column",
                    )
                )
                if node_ref not in aggregate_dependencies:
                    aggregate_dependencies.append(node_ref)
            for dep in self._collect_graph_template_dependencies(template_spec):
                if dep not in aggregate_dependencies:
                    aggregate_dependencies.append(dep)

            merged_columns = [*base_columns]
            merged_columns.append(
                {
                    "label": "df",
                    "data": {
                        "type": "dataframe",
                        "columns": table_columns,
                    },
                }
            )
            merged_column_labels = {
                col.get("label")
                for col in merged_columns
                if isinstance(col, dict) and isinstance(col.get("label"), str)
            }
            rendered_steps: list[dict[str, Any]] = []
            for step in base_steps:
                if not isinstance(step, dict):
                    raise ValueError(
                        f"LLMChatOp {llm_op_id} aggregate format step must be an object"
                    )
                step_template = step.get("template")
                step_arguments = step.get("arguments", [])
                if not isinstance(step_arguments, list):
                    raise ValueError(
                        f"LLMChatOp {llm_op_id} aggregate format step arguments must be"
                        " a list"
                    )
                arguments = [*step_arguments]
                existing_labels = {
                    arg.get("label")
                    for arg in arguments
                    if isinstance(arg, dict) and isinstance(arg.get("label"), str)
                }
                if isinstance(step_template, str):
                    placeholder_labels = {
                        match.group(1)
                        for match in re.finditer(
                            r"\{([A-Za-z_][A-Za-z0-9_]*)\}", step_template
                        )
                    }
                    for label in sorted(placeholder_labels):
                        if label in existing_labels:
                            continue
                        if label not in merged_column_labels:
                            continue
                        arguments.append({"label": label, "value": label})
                rendered_steps.append({**step, "arguments": arguments})

            format_payload: dict[str, Any] = {"messages": base_messages}
            if rendered_steps:
                format_payload["steps"] = rendered_steps

            aggregate_template_spec: dict[str, Any] = {
                "name": "format",
                "columns": merged_columns,
                "options": {"format": format_payload},
            }
            data_spec: dict[str, Any] = {
                "type": "graph_template",
                "template": aggregate_template_spec,
            }
            return self._create_runtime_op(
                name=llm_op_id,
                task_type=task_type,
                data_spec=data_spec,
                model_spec=self._build_model_spec(llm_op.config, backend),
                inference_spec=inference_spec,
                backend=backend,
                model=llm_op.config.model,
                dependencies=aggregate_dependencies if aggregate_dependencies else None,
                output_spec=output_spec,
                condition=self._row_condition(
                    llm_op.condition,
                    0,
                    1,
                    graph_dict,
                    inputs_dict,
                    dsl_to_runtime,
                ),
            )

        default_dependencies: list[str] | None = None
        if task_type != "data_profile":
            merged_dependencies: list[str] = []
            seen_deps: set[str] = set()
            for dep in [
                *upstream_llm_ids,
                *self._collect_graph_template_dependencies(template_spec),
            ]:
                if dep in seen_deps:
                    continue
                seen_deps.add(dep)
                merged_dependencies.append(dep)
            default_dependencies = merged_dependencies or None
        return self._create_runtime_op(
            name=llm_op_id,
            task_type=task_type,
            data_spec={
                "type": "graph_template",
                "template": template_spec,
            },
            model_spec=self._build_model_spec(llm_op.config, backend),
            inference_spec=inference_spec,
            backend=backend,
            model=llm_op.config.model,
            dependencies=default_dependencies,
            output_spec=output_spec,
            condition=self._row_condition(
                llm_op.condition if isinstance(llm_op, LLMChatOp) else None,
                0,
                1,
                graph_dict,
                inputs_dict,
                dsl_to_runtime,
            ),
        )

    def _api_request_headers(
        self, llm_op_id: str, api_config: ApiConfig, url: str
    ) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if is_api_origin_trusted(url):
            server_token = resolve_api_credential(url)
            if not server_token:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' API mode targets trusted endpoint "
                    f"{url} but no PAT is configured; set LUMILAKE_RUNTIME_TOKEN."
                )
            headers["Authorization"] = f"Bearer {server_token}"
        elif api_config.authorization:
            headers["Authorization"] = api_config.authorization
        else:
            raise ValueError(
                f"LLMChatOp '{llm_op_id}' API mode targets untrusted endpoint "
                f"{url}; supply config.api.authorization or add its origin to "
                "LUMILAKE_API_TRUSTED_ORIGINS."
            )
        return headers

    def _build_api_llm_op(
        self,
        *,
        llm_op_id: str,
        llm_op: LLMOp,
        api_config: ApiConfig,
        template_spec: dict[str, Any],
        upstream_llm_ids: list[str],
        output_spec: dict[str, Any] | None,
        condition: dict[str, str] | None,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None = None,
    ) -> list[RuntimeOp]:
        """Build FlowMesh ``api`` tasks for an externally-hosted LLM: render
        resolved ``graph_template`` messages into flat OpenAI-style chat bodies,
        with upstream node references as ``${node.path}`` dispatch placeholders
        that fan out one node per row."""
        if isinstance(llm_op, LLMChatOp) and llm_op.rowwise_template:
            return self._build_api_rowwise_op(
                llm_op_id=llm_op_id,
                llm_op=llm_op,
                api_config=api_config,
                output_spec=output_spec,
                condition=condition,
                graph_dict=graph_dict,
                inputs_dict=inputs_dict,
                dsl_to_runtime=dsl_to_runtime,
            )
        if isinstance(llm_op, LLMChatOp) and llm_op.aggregate_table:
            return self._build_api_aggregate_op(
                llm_op_id=llm_op_id,
                llm_op=llm_op,
                api_config=api_config,
                template_spec=template_spec,
                upstream_llm_ids=upstream_llm_ids,
                output_spec=output_spec,
                condition=condition,
                graph_dict=graph_dict,
                inputs_dict=inputs_dict,
                dsl_to_runtime=dsl_to_runtime,
            )
        resolved, row_count = self._resolve_api_messages(llm_op_id, template_spec)

        model = llm_op.config.resolved_model()
        url = api_config.url or _DEFAULT_API_URL
        headers = self._api_request_headers(llm_op_id, api_config, url)

        merged_dependencies: list[str] = []
        seen_deps: set[str] = set()
        for dep in [
            *upstream_llm_ids,
            *self._collect_graph_template_dependencies(template_spec),
        ]:
            if dep in seen_deps:
                continue
            seen_deps.add(dep)
            merged_dependencies.append(dep)
        dependencies = merged_dependencies or None
        runtime_ops: list[RuntimeOp] = []
        for row_index in range(row_count):
            messages = [
                {
                    "role": role,
                    "content": rows[row_index] if len(rows) > 1 else rows[0],
                }
                for role, rows in resolved
            ]
            body: dict[str, Any] = {"model": model, "messages": messages}
            body.update(llm_op.config.inference_spec())
            if isinstance(llm_op, LLMChatOp) and llm_op.structural_outputs:
                body["templates"] = llm_op.structural_outputs
            api_spec: dict[str, Any] = {
                "method": "POST",
                "url": url,
                "headers": dict(headers),
                "json": body,
                "response": {
                    "parse_json": True,
                    "return_body": True,
                    "raise_for_status": True,
                },
            }
            if api_config.timeout_sec is not None:
                api_spec["timeout_sec"] = api_config.timeout_sec
            node_id = llm_op_id if row_index == 0 else f"{llm_op_id}__row{row_index}"
            runtime_ops.append(
                self._create_runtime_op(
                    name=node_id,
                    task_type="api",
                    data_spec={"type": "graph_template", "template": template_spec},
                    model_spec={},
                    inference_spec={},
                    api_spec=api_spec,
                    backend="api",
                    model=model,
                    dependencies=dependencies,
                    output_spec=output_spec,
                    condition=self._row_condition(
                        condition,
                        row_index,
                        row_count,
                        graph_dict,
                        inputs_dict,
                        dsl_to_runtime,
                    ),
                )
            )
        return runtime_ops

    def _build_api_rowwise_op(
        self,
        *,
        llm_op_id: str,
        llm_op: LLMChatOp,
        api_config: ApiConfig,
        output_spec: dict[str, Any] | None,
        condition: dict[str, str] | None,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None = None,
    ) -> list[RuntimeOp]:
        """Build API tasks for a rowwise LLMChatOp: each ``rowwise_column``
        resolves to a row of values, the template is formatted per row, and the
        op fans out into one API task per row."""
        template = llm_op.rowwise_template
        assert template is not None
        columns: list[dict[str, Any]] = []
        dependencies: list[str] = []
        for col in llm_op.rowwise_columns or []:
            label = col.get("label")
            data = col.get("data")
            node_ref = col.get("node")
            path = col.get("path")
            if isinstance(label, str) and isinstance(data, dict):
                items = data.get("items")
                if not isinstance(items, list) or not items:
                    raise ValueError(
                        f"LLMChatOp {llm_op_id} rowwise column '{label}' has no"
                        " literal items"
                    )
                columns.append({"label": label, "values": [str(i) for i in items]})
            elif (
                isinstance(label, str)
                and isinstance(node_ref, str)
                and isinstance(path, str)
            ):
                upstream = graph_dict.get(node_ref)
                upstream_row_count = (
                    self._static_output_row_count(upstream, inputs_dict, graph_dict)
                    if isinstance(upstream, LLMOp)
                    else None
                )
                if isinstance(upstream, LLMOp) and (
                    (upstream_row_count is not None and upstream_row_count > 1)
                    or len(self._fanout_row_ids(node_ref, dsl_to_runtime)) > 1
                ):
                    raise ValueError(
                        f"LLMChatOp '{llm_op_id}' rowwise column '{label}'"
                        f" references '{node_ref}', which produces multiple rows;"
                        " API mode can only carry one row per node reference, so"
                        " wiring this to the single unsuffixed output would"
                        " silently drop every row but the first."
                    )
                resolved_path = path
                if resolved_path == "items" or resolved_path.startswith("items."):
                    resolved_path = f"items.0{resolved_path[len('items'):]}"
                columns.append(
                    {"label": label, "values": [f"${{{node_ref}.{resolved_path}}}"]}
                )
                if node_ref not in dependencies:
                    dependencies.append(node_ref)
            else:
                raise ValueError(
                    f"LLMChatOp {llm_op_id} rowwise column '{label}' must have"
                    " data or node+path"
                )
        if not columns:
            raise ValueError(f"LLMChatOp {llm_op_id} has empty rowwise_columns")
        row_count = max(len(c["values"]) for c in columns)
        for c in columns:
            if len(c["values"]) == 1 and row_count > 1:
                c["values"] = c["values"] * row_count
            elif len(c["values"]) != row_count:
                raise ValueError(
                    f"LLMChatOp {llm_op_id} rowwise columns have mismatched row"
                    " counts"
                )
        system_messages = llm_op.system_messages or []
        model = llm_op.config.resolved_model()
        url = api_config.url or _DEFAULT_API_URL
        headers = self._api_request_headers(llm_op_id, api_config, url)
        runtime_ops: list[RuntimeOp] = []
        for row_index in range(row_count):
            row_values = {c["label"]: c["values"][row_index] for c in columns}
            user_content = template.format(**row_values)
            messages = [{"role": "system", "content": m} for m in system_messages] + [
                {"role": "user", "content": user_content}
            ]
            body: dict[str, Any] = {"model": model, "messages": messages}
            body.update(llm_op.config.inference_spec())
            if llm_op.structural_outputs:
                body["templates"] = llm_op.structural_outputs
            api_spec: dict[str, Any] = {
                "method": "POST",
                "url": url,
                "headers": dict(headers),
                "json": body,
                "response": {
                    "parse_json": True,
                    "return_body": True,
                    "raise_for_status": True,
                },
            }
            if api_config.timeout_sec is not None:
                api_spec["timeout_sec"] = api_config.timeout_sec
            node_id = llm_op_id if row_index == 0 else f"{llm_op_id}__row{row_index}"
            runtime_ops.append(
                self._create_runtime_op(
                    name=node_id,
                    task_type="api",
                    data_spec={"type": "graph_template", "template": {}},
                    model_spec={},
                    inference_spec={},
                    api_spec=api_spec,
                    backend="api",
                    model=model,
                    dependencies=dependencies or None,
                    output_spec=output_spec,
                    condition=self._row_condition(
                        condition,
                        row_index,
                        row_count,
                        graph_dict,
                        inputs_dict,
                        dsl_to_runtime,
                    ),
                )
            )
        return runtime_ops

    def _resolve_api_messages(
        self, llm_op_id: str, template_spec: dict[str, Any]
    ) -> tuple[list[tuple[str, list[str]]], int]:
        """Resolve a graph-template spec into per-row message content for API
        mode; returns ``(resolved, row_count)`` where each entry is
        ``(role, rows)``."""
        options = template_spec.get("options") or {}
        format_options = options.get("format") or {}
        messages_spec = format_options.get("messages") or []
        columns_spec = template_spec.get("columns") or []
        steps_spec = format_options.get("steps") or []

        column_by_label: dict[str, dict[str, Any]] = {}
        for col in columns_spec:
            if isinstance(col, dict) and isinstance(col.get("label"), str):
                column_by_label[col["label"]] = col

        step_by_label: dict[str, dict[str, Any]] = {}
        for step in steps_spec:
            if isinstance(step, dict) and isinstance(step.get("label"), str):
                step_by_label[step["label"]] = step

        def _runtime_reference_placeholder(
            label: str, column: dict[str, Any]
        ) -> str | None:
            node = column.get("node")
            if not isinstance(node, str) or not node:
                return None
            path = column.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' API mode column '{label}'"
                    f" references node '{node}' without a result path."
                )
            if path == "items" or path.startswith("items."):
                path = f"items.0{path[len('items'):]}"
            return f"${{{node}.{path}}}"

        def _resolve_dataframe_column(label: str, column: dict[str, Any]) -> list[str]:
            nested = column.get("data", {}).get("columns") or []
            if not isinstance(nested, list) or not nested:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' API mode dataframe column"
                    f" '{label}' has no nested columns."
                )
            resolved_cols: list[dict[str, Any]] = []
            row_count = 1
            for nested_col in nested:
                if not isinstance(nested_col, dict):
                    continue
                nested_label = nested_col.get("label")
                if not isinstance(nested_label, str):
                    continue
                rows = _resolve_literal_column(nested_label, nested_col)
                if len(rows) > 1:
                    if row_count != 1 and row_count != len(rows):
                        raise ValueError(
                            f"LLMChatOp '{llm_op_id}' API mode dataframe column"
                            f" '{label}' has mismatched row counts."
                        )
                    row_count = len(rows)
                resolved_cols.append({"label": nested_label, "rows": rows})
            for rc in resolved_cols:
                if len(rc["rows"]) == 1 and row_count > 1:
                    rc["rows"] = rc["rows"] * row_count
            header = " | ".join(rc["label"] for rc in resolved_cols)
            sep = " | ".join("---" for _ in resolved_cols)
            lines = [header, sep]
            for i in range(row_count):
                lines.append(" | ".join(rc["rows"][i] for rc in resolved_cols))
            return ["\n".join(lines)]

        def _resolve_literal_column(label: str, column: dict[str, Any]) -> list[str]:
            data = column.get("data")
            if isinstance(data, dict) and data.get("type") == "dataframe":
                return _resolve_dataframe_column(label, column)
            if not isinstance(data, dict) or data.get("type") != "list":
                placeholder = _runtime_reference_placeholder(label, column)
                if placeholder is not None:
                    return [placeholder]
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' API mode column '{label}' has"
                    " neither literal data nor a node reference to render."
                )
            items = data.get("items")
            if not isinstance(items, list) or not items:
                count = len(items) if isinstance(items, list) else "none"
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' API mode requires at least one row"
                    f" per message column, got {count}."
                )
            return [str(item) for item in items]

        def _resolve_label_rows(label: str) -> list[str]:
            column = column_by_label.get(label)
            if column is not None:
                return _resolve_literal_column(label, column)

            step = step_by_label[label]
            if "function" in step:
                return _resolve_function_step(label, step)
            template = step["template"]
            arguments = step.get("arguments") or []
            arg_rows: dict[str, list[str]] = {}
            step_row_count = 1
            for argument in arguments:
                arg_label = argument["label"]
                value_label = argument["value"]
                rows = _resolve_label_rows(value_label)
                if len(rows) > 1:
                    if step_row_count != 1 and step_row_count != len(rows):
                        raise ValueError(
                            f"LLMChatOp '{llm_op_id}' API mode step '{label}' has"
                            " mismatched row counts across its arguments."
                        )
                    step_row_count = len(rows)
                arg_rows[arg_label] = rows
            return [
                template.format(
                    **{
                        arg_label: rows[index] if len(rows) > 1 else rows[0]
                        for arg_label, rows in arg_rows.items()
                    }
                )
                for index in range(step_row_count)
            ]

        def _resolve_function_step(label: str, step: dict[str, Any]) -> list[str]:
            code = step["function"]
            arguments = step.get("arguments") or []
            arg_rows: list[list[str]] = []
            step_row_count = 1
            for argument in arguments:
                rows = _resolve_label_rows(argument)
                if len(rows) > 1:
                    if step_row_count != 1 and step_row_count != len(rows):
                        raise ValueError(
                            f"LLMChatOp '{llm_op_id}' API mode step '{label}' has"
                            " mismatched row counts across its arguments."
                        )
                    step_row_count = len(rows)
                arg_rows.append(rows)
            if any(
                _PLACEHOLDER_RE.search(value) for rows in arg_rows for value in rows
            ):
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' API mode cannot render message step"
                    f" '{label}': a Lambda message transform over a runtime output"
                    " is not supported in API mode. The API request body cannot"
                    " carry a graph_template function step, so the transform"
                    " cannot be evaluated at dispatch time. Apply the transform"
                    " in a separate local op, or keep this op local."
                )
            fn = safe_materialize_function(code)
            return [
                str(
                    fn(
                        tuple(
                            rows[index] if len(rows) > 1 else rows[0]
                            for rows in arg_rows
                        )
                    )
                )
                for index in range(step_row_count)
            ]

        def _resolve_content_rows(content: Any) -> list[str]:
            if not isinstance(content, str):
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' API mode requires string message"
                    f" content, got {type(content).__name__}."
                )
            if content in column_by_label or content in step_by_label:
                return _resolve_label_rows(content)
            return [content]

        resolved: list[tuple[str, list[str]]] = []
        row_count = 1
        for message in messages_spec:
            if not isinstance(message, dict):
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' API mode requires message objects."
                )
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' API mode requires a message role."
                )
            rows = _resolve_content_rows(message.get("content"))
            if len(rows) > 1:
                if row_count != 1 and row_count != len(rows):
                    raise ValueError(
                        f"LLMChatOp '{llm_op_id}' API mode message columns have"
                        f" mismatched row counts: {row_count} vs {len(rows)}."
                    )
                row_count = len(rows)
            resolved.append((role, rows))

        if not resolved:
            raise ValueError(f"LLMChatOp '{llm_op_id}' API mode requires messages.")
        return resolved, row_count

    def _build_api_aggregate_op(
        self,
        *,
        llm_op_id: str,
        llm_op: LLMChatOp,
        api_config: ApiConfig,
        template_spec: dict[str, Any],
        upstream_llm_ids: list[str],
        output_spec: dict[str, Any] | None,
        condition: dict[str, str] | None,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None = None,
    ) -> list[RuntimeOp]:
        """Build API tasks for an aggregate LLMChatOp: merge the base template
        columns with a ``df`` dataframe column built from ``aggregate_table``."""
        base_columns = template_spec.get("columns", [])
        if not isinstance(base_columns, list):
            raise ValueError(
                f"LLMChatOp {llm_op_id} aggregate template columns must be a list"
            )
        table_columns: list[dict[str, Any]] = []
        aggregate_dependencies = list(upstream_llm_ids) if upstream_llm_ids else []
        for col in llm_op.aggregate_table or []:
            label = col.get("label")
            node_ref = col.get("node")
            path = col.get("path")
            if not (
                isinstance(label, str)
                and isinstance(node_ref, str)
                and isinstance(path, str)
            ):
                continue
            upstream = graph_dict.get(node_ref)
            upstream_row_count = (
                self._static_output_row_count(upstream, inputs_dict, graph_dict)
                if isinstance(upstream, LLMOp)
                else None
            )
            if isinstance(upstream, LLMOp) and (
                (upstream_row_count is not None and upstream_row_count > 1)
                or len(self._fanout_row_ids(node_ref, dsl_to_runtime)) > 1
            ):
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' aggregate column '{label}'"
                    f" references '{node_ref}', which produces multiple rows;"
                    " API mode can only carry one row per node reference, so"
                    " wiring this to the single unsuffixed output would"
                    " silently drop every row but the first."
                )
            table_columns.append({"label": label, "node": node_ref, "path": path})
            if node_ref not in aggregate_dependencies:
                aggregate_dependencies.append(node_ref)
        for dep in self._collect_graph_template_dependencies(template_spec):
            if dep not in aggregate_dependencies:
                aggregate_dependencies.append(dep)

        merged_columns = [*base_columns]
        merged_columns.append(
            {
                "label": "df",
                "data": {"type": "dataframe", "columns": table_columns},
            }
        )
        format_options = template_spec.get("options", {}).get("format", {})
        base_messages = format_options.get("messages", [])
        if not isinstance(base_messages, list):
            raise ValueError(
                f"LLMChatOp {llm_op_id} aggregate template messages must be a list"
            )
        base_steps = format_options.get("steps", [])
        if not isinstance(base_steps, list):
            raise ValueError(
                f"LLMChatOp {llm_op_id} aggregate template steps must be a list"
            )
        merged_column_labels = {
            col.get("label")
            for col in merged_columns
            if isinstance(col, dict) and isinstance(col.get("label"), str)
        }
        rendered_steps: list[dict[str, Any]] = []
        for step in base_steps:
            if not isinstance(step, dict):
                raise ValueError(
                    f"LLMChatOp {llm_op_id} aggregate format step must be an object"
                )
            step_template = step.get("template")
            step_arguments = step.get("arguments", [])
            if not isinstance(step_arguments, list):
                raise ValueError(
                    f"LLMChatOp {llm_op_id} aggregate format step arguments must be"
                    " a list"
                )
            arguments = [*step_arguments]
            existing_labels = {
                arg.get("label")
                for arg in arguments
                if isinstance(arg, dict) and isinstance(arg.get("label"), str)
            }
            if isinstance(step_template, str):
                placeholder_labels = {
                    match.group(1)
                    for match in re.finditer(
                        r"\{([A-Za-z_][A-Za-z0-9_]*)\}", step_template
                    )
                }
                for label in sorted(placeholder_labels):
                    if label in existing_labels:
                        continue
                    if label not in merged_column_labels:
                        continue
                    arguments.append({"label": label, "value": label})
            rendered_steps.append({**step, "arguments": arguments})

        format_payload: dict[str, Any] = {"messages": base_messages}
        if rendered_steps:
            format_payload["steps"] = rendered_steps
        aggregate_template_spec: dict[str, Any] = {
            "name": "format",
            "columns": merged_columns,
            "options": {"format": format_payload},
        }
        resolved, row_count = self._resolve_api_messages(
            llm_op_id, aggregate_template_spec
        )

        model = llm_op.config.resolved_model()
        url = api_config.url or _DEFAULT_API_URL
        headers = self._api_request_headers(llm_op_id, api_config, url)
        runtime_ops: list[RuntimeOp] = []
        for row_index in range(row_count):
            messages = [
                {
                    "role": role,
                    "content": rows[row_index] if len(rows) > 1 else rows[0],
                }
                for role, rows in resolved
            ]
            body: dict[str, Any] = {"model": model, "messages": messages}
            body.update(llm_op.config.inference_spec())
            if llm_op.structural_outputs:
                body["templates"] = llm_op.structural_outputs
            api_spec: dict[str, Any] = {
                "method": "POST",
                "url": url,
                "headers": dict(headers),
                "json": body,
                "response": {
                    "parse_json": True,
                    "return_body": True,
                    "raise_for_status": True,
                },
            }
            if api_config.timeout_sec is not None:
                api_spec["timeout_sec"] = api_config.timeout_sec
            node_id = llm_op_id if row_index == 0 else f"{llm_op_id}__row{row_index}"
            runtime_ops.append(
                self._create_runtime_op(
                    name=node_id,
                    task_type="api",
                    data_spec={
                        "type": "graph_template",
                        "template": aggregate_template_spec,
                    },
                    model_spec={},
                    inference_spec={},
                    api_spec=api_spec,
                    backend="api",
                    model=model,
                    dependencies=aggregate_dependencies or None,
                    output_spec=output_spec,
                    condition=self._row_condition(
                        condition,
                        row_index,
                        row_count,
                        graph_dict,
                        inputs_dict,
                        dsl_to_runtime,
                    ),
                )
            )
        return runtime_ops

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

    def _static_output_row_count(
        self,
        op: Op,
        inputs_dict: dict[str, list[str]],
        graph_dict: dict[str, Op] | None = None,
    ) -> int | None:
        """Static row count of an op's output, or ``None`` when not knowable
        at build time (e.g. a retrieval whose row count is only known at
        execution). A rowwise op's row count is driven by its ``rowwise_columns``
        (literal item counts, or the upstream row count for node-ref columns),
        not by its message content."""
        if isinstance(op, InputOp):
            values = inputs_dict.get(op.name)
            return len(values) if values is not None else None
        if isinstance(op, DataOp):
            return len(op.data)
        if isinstance(op, LLMChatOp):
            if op.rowwise_columns:
                counts: list[int] = []
                for col in op.rowwise_columns:
                    data = col.get("data")
                    if isinstance(data, dict):
                        items = data.get("items")
                        if isinstance(items, list):
                            counts.append(len(items))
                            continue
                    node_ref = col.get("node")
                    if isinstance(node_ref, str) and graph_dict is not None:
                        upstream = graph_dict.get(node_ref)
                        if upstream is not None:
                            count = self._static_output_row_count(
                                upstream, inputs_dict, graph_dict
                            )
                            if count is not None:
                                counts.append(count)
                return max(counts) if counts else None
            messages = (
                op.messages.messages if isinstance(op.messages, MessageOp) else []
            )
            op_counts: list[int] = []
            for message in messages:
                if isinstance(message.content, Op):
                    count = self._static_output_row_count(
                        message.content, inputs_dict, graph_dict
                    )
                    if count is not None:
                        op_counts.append(count)
            return max(op_counts) if op_counts else None
        if isinstance(op, FormatOp):
            format_counts = [
                self._static_output_row_count(inp, inputs_dict, graph_dict)
                for inp in op.inputs
            ]
            known = [c for c in format_counts if c is not None]
            return max(known) if known else None
        if isinstance(op, LambdaOp):
            lambda_counts = [
                self._static_output_row_count(inp, inputs_dict, graph_dict)
                for inp in op.inputs
            ]
            known = [c for c in lambda_counts if c is not None]
            return max(known) if known else None
        return None

    def _infer_structural_messages(
        self,
        llm_op_id: str,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited_node_ids: set[str],
        dsl_to_runtime: dict[str, list[str]] | None = None,
        runtime_nodes: dict[str, RuntimeOp] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        target_llm_op = graph_dict[llm_op_id]
        assert isinstance(target_llm_op, LLMOp), "Target op must be an LLMOp"
        upstream_llm_ids: set[str] = set()

        columns: dict[str, dict[str, Any]] = {}
        steps: dict[
            str, tuple[str, Sequence[str | dict[str, str] | list[dict[str, str]]], bool]
        ] = {}

        ancestor_buffer: dict[str, list[str | tuple[Roles, str]]] = {}

        def _unwrap_msg(msg: str | tuple[Roles, str]):
            return msg[1] if isinstance(msg, tuple) else msg

        def _trace_ancestors(op: Op) -> list[str | tuple[Roles, str]]:
            if op.id in ancestor_buffer:
                return ancestor_buffer[op.id]
            visited_node_ids.add(op.id)

            if isinstance(op, LLMOp):
                assert (
                    op.id != target_llm_op.id
                ), "Encountered starting LLMOp again unexpectedly"
                row_ids = self._fanout_row_ids(op.id, dsl_to_runtime)
                is_api_ancestor = self._is_api_task(op)
                target_is_api = self._is_api_task(target_llm_op)
                fanned = len(row_ids) > 1
                if fanned and (not is_api_ancestor or not target_is_api):
                    raise ValueError(
                        f"LLMOp '{llm_op_id}' consumes '{op.id}', which fanned"
                        " out into multiple row-aligned runtime nodes; only"
                        " API mode message columns can carry that per-row"
                        " alignment, so wiring this downstream node to the"
                        " single unsuffixed output would silently drop every"
                        " row but the first."
                    )
                if op.id not in upstream_llm_ids:
                    upstream_llm_ids.update(row_ids)
                    if not is_api_ancestor:
                        row_count = self._static_output_row_count(
                            op, inputs_dict, graph_dict
                        )
                        if row_count is not None and row_count > 1:
                            raise ValueError(
                                f"LLMOp '{llm_op_id}' consumes '{op.id}', which"
                                " produces multiple rows; API mode message columns"
                                " can only carry one row per node reference, so"
                                " wiring this downstream node to the single"
                                " unsuffixed output would silently drop every row"
                                " but the first."
                            )
                    if isinstance(op, LLMChatOp) and op.return_history:
                        if fanned:
                            raise ValueError(
                                f"LLMChatOp '{llm_op_id}' consumes '{op.id}',"
                                " which fanned out into multiple row-aligned"
                                " runtime nodes and has return_history. API mode"
                                " cannot reconstruct per-row history for a"
                                " fanned upstream, so this shape is not"
                                " supported."
                            )
                        if is_api_ancestor:
                            prior = self._api_prior_prompt(op, inputs_dict)
                            if prior is None:
                                raise ValueError(
                                    f"LLMChatOp '{llm_op_id}' consumes '{op.id}',"
                                    " which has return_history and a runtime-derived"
                                    " prior prompt. API mode cannot inline that"
                                    " prompt at build time, and an API task result"
                                    " carries no metadata.prompt at dispatch time,"
                                    " so the history cannot be reconstructed. Use a"
                                    " literal prior prompt, or keep this op local."
                                )
                            columns[f"{op.id}_context"] = {
                                "data": {"type": "list", "items": prior},
                            }
                        else:
                            columns[f"{op.id}_context"] = {
                                "node": op.id,
                                "path": "items.metadata.prompt",
                            }
                    if fanned:
                        # API-only branch: every row's output path is ``text``.
                        columns[f"{op.id}_output"] = {
                            "data": {
                                "type": "list",
                                "items": [f"${{{rid}.text}}" for rid in row_ids],
                            }
                        }
                    else:
                        columns[f"{op.id}_output"] = {
                            "node": op.id,
                            "path": self._upstream_output_path(op),
                        }

                if isinstance(op, LLMChatOp) and op.return_history:
                    ancestor_buffer[op.id] = [
                        f"{op.id}_context",
                        (Roles.ASSISTANT, f"{op.id}_output"),
                    ]
                else:
                    ancestor_buffer[op.id] = [(Roles.ASSISTANT, f"{op.id}_output")]

            elif isinstance(op, InputOp):
                if op.id not in columns:
                    columns[op.id] = {
                        "data": {"type": "list", "items": inputs_dict[op.name]}
                    }
                ancestor_buffer[op.id] = [(Roles.USER, op.id)]

            elif isinstance(op, DataOp):
                if op.id not in columns:
                    columns[op.id] = {"data": {"type": "list", "items": op.data}}
                ancestor_buffer[op.id] = [(Roles.USER, op.id)]

            elif isinstance(op, DataRetrievalOp):
                if op.id not in columns:
                    data_spec = op.data_spec
                    mode = data_spec.get("mode")
                    if mode == "sql":
                        path = "items.table"
                    elif mode == "s3":
                        path = "items.content"
                    else:
                        path = "items.output"
                    columns[op.id] = {"node": op.id, "path": path}
                ancestor_buffer[op.id] = [(Roles.USER, op.id)]

            elif isinstance(op, MessageOp):
                ancestor_buffer[op.id] = []
                for message in op.messages:
                    if isinstance(message.content, str):
                        ancestor_buffer[op.id].append(
                            (Roles(message.role), message.content)
                        )
                    else:
                        input_messages = _trace_ancestors(message.content)
                        preserve_roles = (
                            isinstance(message.content, LLMChatOp)
                            and message.content.return_history
                        )
                        for packed_msg in input_messages:
                            if isinstance(packed_msg, str):
                                ancestor_buffer[op.id].append(
                                    (Roles(message.role), packed_msg)
                                )
                                continue
                            role, msg = packed_msg
                            if preserve_roles:
                                ancestor_buffer[op.id].append((role, msg))
                                continue
                            assert role == Roles.USER, (
                                "MessageOp overwrites messages whose role is not USER."
                                " Please check if this is intended."
                            )
                            ancestor_buffer[op.id].append((Roles(message.role), msg))

            elif isinstance(op, FormatOp):
                assert len(op.inputs) >= 1, "FormatOp should have at least one input"
                message_labels = {
                    inp_op.id: _trace_ancestors(inp_op) for inp_op in op.inputs
                }
                multi = [
                    (len(msgs), inp_id, msgs)
                    for inp_id, msgs in message_labels.items()
                    if len(msgs) != 1
                ]
                assert all(len(msgs) == 1 for msgs in message_labels.values()), (
                    "FormatOp inputs should each resolve to a single message, but got"
                    f" multiple in {op.id}: {multi}"
                )
                template = op.template
                format_kwargs = [
                    {"label": k, "value": _unwrap_msg(message_labels[v.id][0])}
                    for k, v in op.format_kwargs.items()
                ]
                label = f"format_{op.id}"
                steps[label] = (template, format_kwargs, False)
                ancestor_buffer[op.id] = [(Roles.USER, label)]

            elif isinstance(op, LambdaOp):
                message_labels = {
                    inp_op.id: _trace_ancestors(inp_op) for inp_op in op.inputs
                }

                def _resolve_literal_label(label: str) -> str | None:
                    col = columns.get(label)
                    if col is not None:
                        data = col.get("data")
                        if (
                            isinstance(data, dict)
                            and data.get("type") == "list"
                            and isinstance(data.get("items"), list)
                            and len(data["items"]) == 1
                        ):
                            return str(data["items"][0])
                        return None
                    step = steps.get(label)
                    if step is None:
                        return label
                    template, kwargs, is_function = step
                    if is_function:
                        return None
                    resolved_kwargs: dict[str, str] = {}
                    for kwarg in kwargs:
                        if not isinstance(kwarg, dict):
                            return None
                        value = _resolve_literal_label(kwarg["value"])
                        if value is None:
                            return None
                        resolved_kwargs[kwarg["label"]] = value
                    return template.format(**resolved_kwargs)

                literal_args: list[str] = []
                can_evaluate = True
                for inp_op in op.inputs:
                    msgs = message_labels[inp_op.id]
                    if len(msgs) != 1:
                        can_evaluate = False
                        break
                    value = _resolve_literal_label(_unwrap_msg(msgs[0]))
                    if value is None:
                        can_evaluate = False
                        break
                    literal_args.append(value)
                if can_evaluate:
                    label = f"lambda_{op.id}"
                    columns[label] = {
                        "data": {
                            "type": "list",
                            "items": [str(op.fn(tuple(literal_args)))],
                        }
                    }
                    ancestor_buffer[op.id] = [(Roles.USER, label)]
                else:
                    fn_args = [
                        [
                            (
                                {"role": message[0].value, "content": message[1]}
                                if isinstance(message, tuple)
                                else {"content": message}
                            )
                            for message in message_labels[inp_op.id]
                        ]
                        for inp_op in op.inputs
                    ]
                    fn_args_serialized = [
                        fn_arg if len(fn_arg) > 1 else fn_arg[0]["content"]
                        for fn_arg in fn_args
                    ]
                    label = f"lambda_{op.id}"
                    steps[label] = (op.code, fn_args_serialized, True)
                    ancestor_buffer[op.id] = [(Roles.USER, label)]

            else:
                raise NotImplementedError(
                    f"Unsupported op type '{type(op)}' (id: {op.id}) in input chain. "
                    "Please add support for this op type."
                )

            return ancestor_buffer[op.id]

        message_order = _trace_ancestors(target_llm_op.inputs[0])

        columns_spec = [
            {"label": label, **column_spec} for label, column_spec in columns.items()
        ]
        steps_spec = [
            {
                "label": label,
                ("function" if is_function else "template"): step_spec_string,
                "arguments": kwargs,
            }
            for label, (step_spec_string, kwargs, is_function) in steps.items()
        ]
        messages_spec = [
            (
                {"role": message[0].value, "content": message[1]}
                if isinstance(message, tuple)
                else {"content": message}
            )
            for message in message_order
        ]

        step_config: dict[str, Any] = {
            "name": "format",
            "columns": columns_spec,
            "options": {
                "format": {
                    "steps": steps_spec,
                    "messages": messages_spec,
                }
            },
        }

        return list(upstream_llm_ids), step_config

    def _extract_table_from_sql_template(self, template: str) -> str:
        match = re.search(r"\bFROM\s+([^\s]+)", template, re.IGNORECASE)
        if not match:
            raise ValueError(f"Unable to infer SQL table from template: {template}")
        return match.group(1).strip()

    def _sample_value_from_structural_outputs(
        self,
        *,
        owner_node_id: str,
        label: str,
        path: Any,
        upstream_id: str,
        upstream_op_kind: str,
        structural_outputs: Any,
    ) -> Any:
        _remediation = (
            "Attach 'structural_outputs' to the LLM op, add a 'sample_value' "
            "to the upstream data_spec, or set LUMILAKE_DISABLE_DATA_PROFILE=1."
        )

        if not isinstance(structural_outputs, list) or not structural_outputs:
            raise ValueError(
                f"Data profile preflight cannot resolve placeholder '{label}' "
                f"at node '{owner_node_id}': upstream node '{upstream_id}' "
                f"({upstream_op_kind}) has no 'structural_outputs'. "
                f"Path requested: '{path}'. {_remediation}"
            )

        field_name: str | None = None
        if isinstance(path, str) and path.startswith("items.output."):
            field_name = path[len("items.output.") :]

        template_entry: dict[str, Any] | None = None
        if field_name:
            template_entry = next(
                (
                    item
                    for item in structural_outputs
                    if isinstance(item, dict) and item.get("name") == field_name
                ),
                None,
            )
            if template_entry is None:
                available = [
                    item.get("name")
                    for item in structural_outputs
                    if isinstance(item, dict) and item.get("name") is not None
                ]
                raise ValueError(
                    f"Data profile preflight cannot resolve placeholder '{label}' "
                    f"at node '{owner_node_id}': upstream node '{upstream_id}' "
                    f"({upstream_op_kind}) has 'structural_outputs' but does not "
                    f"contain a field named '{field_name}' (path: '{path}'). "
                    f"Available fields: {available}. {_remediation}"
                )
        elif len(structural_outputs) == 1 and isinstance(structural_outputs[0], dict):
            template_entry = structural_outputs[0]

        if template_entry is None:
            raise ValueError(
                f"Data profile preflight cannot resolve placeholder '{label}' "
                f"at node '{owner_node_id}': upstream node '{upstream_id}' "
                f"({upstream_op_kind}) has 'structural_outputs' but the path "
                f"'{path}' could not be matched to any field. {_remediation}"
            )

        candidates = template_entry.get("candidates")
        if isinstance(candidates, list) and candidates:
            return candidates[0]
        min_value = template_entry.get("min")
        if min_value is not None:
            return min_value
        type_name = template_entry.get("type")
        return _type_default_sample(type_name if isinstance(type_name, str) else None)

    def _sample_value_from_upstream_retrieval(
        self,
        *,
        owner_node_id: str,
        label: str,
        path: Any,
        upstream_id: str,
        upstream: DataRetrievalOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited: set[str],
    ) -> Any:
        if upstream_id in visited:
            raise ValueError(
                "Data profile preflight detected a cycle while resolving "
                f"placeholder '{label}' at node '{owner_node_id}': upstream "
                f"'{upstream_id}' was already in the resolution stack."
            )
        upstream_spec = upstream.data_spec or {}
        if not isinstance(upstream_spec, dict):
            raise ValueError(
                f"DataRetrievalOp '{upstream_id}' has a non-dict data_spec"
            )
        sample_value = upstream_spec.get("sample_value")
        if sample_value is not None:
            return self._project_sample_value(sample_value, path)

        if not envs.LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING:
            raise ValueError(
                f"Data profile preflight cannot sample upstream "
                f"DataRetrievalOp '{upstream_id}' for placeholder '{label}' at "
                f"node '{owner_node_id}': live sampling is off by default. "
                f"Set 'sample_value' on the upstream data_spec to supply a "
                f"representative value without any live execution, or set "
                f"LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING=1 to explicitly "
                f"opt in to bounded live queries. "
                f"Set LUMILAKE_DISABLE_DATA_PROFILE=1 to skip data profiling "
                f"entirely."
            )

        column = self._column_from_path(path)
        next_visited = visited | {upstream_id}
        rendered_query = self._render_upstream_sample_query(
            upstream_id=upstream_id,
            upstream=upstream,
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
            visited=next_visited,
        )
        upstream_mode = upstream_spec.get("mode")
        if upstream_mode == "sql":
            return self._fetch_sql_sample_value(
                owner_node_id=owner_node_id,
                label=label,
                upstream_id=upstream_id,
                query=rendered_query,
                column=column,
            )
        if upstream_mode == "s3":
            return self._fetch_s3_sample_value(
                owner_node_id=owner_node_id,
                label=label,
                upstream_id=upstream_id,
                prefix=rendered_query,
                column=column,
                encoding=upstream_spec.get("encoding", "utf-8"),
            )
        raise ValueError(
            "Data profile preflight cannot sample upstream "
            f"DataRetrievalOp '{upstream_id}' with mode "
            f"{upstream_mode!r}. Add a 'sample_value' to its data_spec or "
            "set LUMILAKE_DISABLE_DATA_PROFILE=1."
        )

    @staticmethod
    def _column_from_path(path: Any) -> str | None:
        if not isinstance(path, str):
            return None
        for prefix in ("items.output.", "items.table.", "items."):
            if path.startswith(prefix):
                tail = path[len(prefix) :]
                return tail or None
        return path or None

    @staticmethod
    def _project_sample_value(value: Any, path: Any) -> Any:
        if not isinstance(path, str) or not path:
            return value
        column = RuntimeGraphBuilder._column_from_path(path)
        if column is None:
            return value
        if isinstance(value, dict) and column in value:
            return value[column]
        return value

    def _render_upstream_sample_query(
        self,
        *,
        upstream_id: str,
        upstream: DataRetrievalOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited: set[str],
    ) -> str:
        upstream_spec = upstream.data_spec or {}
        template = upstream_spec.get("template")
        if not isinstance(template, str):
            raise ValueError(
                f"Upstream DataRetrievalOp '{upstream_id}' has no template"
            )
        params = upstream_spec.get("params") or []
        if not isinstance(params, list):
            raise ValueError(
                f"Upstream DataRetrievalOp '{upstream_id}' params must be a list"
            )
        resolved_params = [
            resolved
            for param in params
            if (
                resolved := self._resolve_profile_param(
                    upstream_id,
                    param,
                    graph_dict,
                    inputs_dict,
                    visited,
                )
            )
            is not None
        ]
        constraints = self._build_data_profile_constraints(params, graph_dict)
        queries = _build_sample_data_profile_queries(
            template=template,
            params=resolved_params,
            constraints=constraints,
            num_samples=envs.LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES,
            node_id=upstream_id,
        )
        if not queries:
            raise ValueError(
                f"Upstream DataRetrievalOp '{upstream_id}' produced no sample query"
            )
        upstream_mode = upstream_spec.get("mode")
        if upstream_mode == "sql":
            return self._apply_limit_to_sample_sql(
                queries[0],
                max(1, int(envs.LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES)),
            )
        return queries[0]

    _DISALLOWED_SAMPLE_KEYWORDS: frozenset[str] = frozenset(
        {
            "INSERT",
            "UPDATE",
            "DELETE",
            "MERGE",
            "TRUNCATE",
            "CREATE",
            "DROP",
            "ALTER",
            "GRANT",
            "REVOKE",
            "CALL",
            "EXECUTE",
            "COPY",
            "VACUUM",
            "LOCK",
        }
    )

    @staticmethod
    def _assert_sample_query_is_readonly(query: str) -> None:
        statements = [s for s in sqlparse.parse(query) if s.value.strip()]
        if len(statements) > 1:
            raise ValueError(
                "Data profile preflight rejects multi-statement SQL scripts. "
                "Add a 'sample_value' to the upstream data_spec to skip the "
                "live sample query."
            )
        if not statements:
            return
        parsed = statements[0]
        if parsed.get_type() != "SELECT":
            raise ValueError(
                f"Data profile preflight requires a SELECT-only sample query "
                f"but the statement type is '{parsed.get_type()}'. Add a "
                f"'sample_value' to the upstream data_spec to skip the live "
                f"sample query."
            )
        disallowed = RuntimeGraphBuilder._DISALLOWED_SAMPLE_KEYWORDS

        def _walk_tokens(token: Any) -> str | None:
            if token.ttype in (Keyword, Keyword.DML, Keyword.DDL):
                upper = token.normalized.upper()
                if upper in disallowed:
                    return upper
            if isinstance(token, TokenList):
                for child in token.tokens:
                    found = _walk_tokens(child)
                    if found is not None:
                        return found
            return None

        offender = _walk_tokens(parsed)
        if offender is not None:
            raise ValueError(
                f"Data profile preflight found a disallowed keyword '{offender}' "
                f"in the sample query (including inside CTEs). Add a "
                f"'sample_value' to the upstream data_spec to skip the live "
                f"sample query."
            )

    @staticmethod
    def _apply_limit_to_sample_sql(query: str, n: int) -> str:
        RuntimeGraphBuilder._assert_sample_query_is_readonly(query)
        if re.search(r"\bFOR\s+(?:UPDATE|SHARE)\b", query, re.IGNORECASE):
            raise ValueError(
                "Data profile preflight cannot append LIMIT to a sample SQL "
                "query that contains a FOR UPDATE / FOR SHARE clause. Add a "
                "'sample_value' to the upstream data_spec to avoid issuing a "
                "live sample query."
            )
        inner = query.rstrip().rstrip(";")
        return f"SELECT * FROM ({inner}) AS _lumilake_sample LIMIT {int(n)}"

    def _fetch_sql_sample_value(
        self,
        *,
        owner_node_id: str,
        label: str,
        upstream_id: str,
        query: str,
        column: str | None,
    ) -> Any:
        if not envs.LUMID_DATA_URL or not envs.LUMID_DATA_TOKEN:
            raise ValueError(
                f"Upstream DataRetrievalOp '{upstream_id}' requires LUMID_DATA_URL "
                "and LUMID_DATA_TOKEN for live SQL sample query via lumid-data-app"
            )
        sampler = _RuntimeProfileSamplers.sql
        try:
            rows = sampler(query)
        except Exception as exc:
            raise ValueError(
                "Data profile preflight could not sample upstream "
                f"'{upstream_id}' for placeholder '{label}' at node "
                f"'{owner_node_id}': {exc}. Add 'sample_value' to the "
                "upstream data_spec or set LUMILAKE_DISABLE_DATA_PROFILE=1."
            ) from exc
        if not rows:
            raise ValueError(
                "Data profile preflight rejected placeholder "
                f"'{label}' at node '{owner_node_id}': no sample_value "
                f"and upstream '{upstream_id}' returned 0 rows."
            )
        row = rows[0]
        if isinstance(row, dict):
            if column is not None and column in row:
                return row[column]
            return next(iter(row.values()), "")
        if isinstance(row, (list, tuple)) and row:
            return row[0]
        return row

    def _fetch_s3_sample_value(
        self,
        *,
        owner_node_id: str,
        label: str,
        upstream_id: str,
        prefix: str,
        column: str | None,
        encoding: Any,
    ) -> Any:
        if not envs.LUMID_DATA_URL or not envs.LUMID_DATA_TOKEN:
            raise ValueError(
                f"Upstream DataRetrievalOp '{upstream_id}' requires LUMID_DATA_URL "
                "and LUMID_DATA_TOKEN for live S3 sample via lumid-data-app"
            )
        sampler = _RuntimeProfileSamplers.s3
        try:
            keys = sampler(prefix)
        except Exception as exc:
            raise ValueError(
                "Data profile preflight could not sample upstream "
                f"'{upstream_id}' for placeholder '{label}' at node "
                f"'{owner_node_id}': {exc}. Add 'sample_value' to the "
                "upstream data_spec or set LUMILAKE_DISABLE_DATA_PROFILE=1."
            ) from exc
        if not keys:
            raise ValueError(
                "Data profile preflight rejected placeholder "
                f"'{label}' at node '{owner_node_id}': no sample_value "
                f"and upstream '{upstream_id}' returned 0 keys."
            )
        return keys[0]

    def _build_data_profile_constraints(
        self,
        params: list[dict[str, Any]],
        graph_dict: dict[str, Op],
    ) -> list[dict[str, Any]]:
        constraints: list[dict[str, Any]] = []
        seen: set[str] = set()
        for param in params:
            label = param.get("label")
            node_id = param.get("node")
            path = param.get("path")
            if not isinstance(label, str) or not isinstance(node_id, str):
                continue
            if not isinstance(path, str) or not path.startswith("items.output"):
                continue
            if label in seen:
                continue

            field_name: str | None = None
            if path != "items.output" and path.startswith("items.output."):
                field_name = path[len("items.output.") :]

            op = graph_dict.get(node_id)
            if not isinstance(op, LLMChatOp):
                continue
            structural_outputs = op.structural_outputs
            if not isinstance(structural_outputs, list):
                continue

            template_entry: dict[str, Any] | None = None
            if field_name:
                template_entry = next(
                    (
                        item
                        for item in structural_outputs
                        if isinstance(item, dict) and item.get("name") == field_name
                    ),
                    None,
                )
            elif len(structural_outputs) == 1 and isinstance(
                structural_outputs[0], dict
            ):
                template_entry = structural_outputs[0]

            if not template_entry:
                continue

            constraint = {"name": label}
            for key, value in template_entry.items():
                if key == "name":
                    continue
                constraint[key] = value
            constraints.append(constraint)
            seen.add(label)

        return constraints
