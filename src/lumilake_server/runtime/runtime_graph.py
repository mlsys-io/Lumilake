import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Self

from lumilake import envs
from lumilake.log import Logger, LogLevel, init_child_logger
from pydantic import BaseModel, ConfigDict, Field

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
from lumilake_server.ops.llm_ops import ImageGenerationOp, LLMChatOp, LLMVisionOp
from lumilake_server.runtime.runtime_ops import RuntimeOp, RuntimeOpSchema
from lumilake_server.utils.graph import topological_sort


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


def make_node_prefix(name: str) -> str:
    safe = _sanitize_node_prefix(name)
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"


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
    dsl_to_runtime: dict[str, list[str]] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def serialize(self) -> dict[str, Any]:
        return RuntimeGraphSchema(
            nodes={
                nid: RuntimeOpSchema.model_validate(op.serialize())
                for nid, op in self.nodes.items()
            },
            node_order=self.node_order,
            output_node_map=self.output_node_map,
            dsl_to_runtime=self.dsl_to_runtime,
        ).model_dump(exclude_none=True)

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

        def remap(value: Any) -> Any:
            if isinstance(value, dict):
                updated: dict[str, Any] = {}
                for key, item in value.items():
                    if key == "node" and isinstance(item, str) and item in mapping:
                        updated[key] = mapping[item]
                    else:
                        updated[key] = remap(item)
                return updated
            if isinstance(value, list):
                return [remap(item) for item in value]
            if isinstance(value, tuple):
                return tuple(remap(item) for item in value)
            return value

        nodes: dict[str, RuntimeOp] = {}
        for old_id, op in self.nodes.items():
            new_id = mapping[old_id]
            nodes[new_id] = RuntimeOp(
                node_id=new_id,
                task_type=op.task_type,
                backend=op.backend,
                model=op.model,
                data_spec=remap(op.data_spec),
                model_spec=remap(op.model_spec),
                inference_spec=remap(op.inference_spec),
                dependencies=tuple(mapping.get(dep, dep) for dep in op.dependencies),
                output_spec=(
                    remap(op.output_spec) if op.output_spec is not None else None
                ),
                condition=(remap(op.condition) if op.condition is not None else None),
            )

        node_order = [mapping[node_id] for node_id in self.node_order]
        output_node_map = {
            mapping[node_id]: output_name
            for node_id, output_name in self.output_node_map.items()
        }
        dsl_to_runtime = {
            op_id: [mapping.get(node_id, node_id) for node_id in runtime_ids]
            for op_id, runtime_ids in self.dsl_to_runtime.items()
        }

        return type(self)(
            nodes=nodes,
            node_order=node_order,
            output_node_map=output_node_map,
            dsl_to_runtime=dsl_to_runtime,
        )


def merge_runtime_graphs(
    graphs: dict[str, RuntimeGraph],
) -> tuple[RuntimeGraph, dict[str, tuple[str, str]]]:
    nodes: dict[str, RuntimeOp] = {}
    node_order: list[str] = []
    output_node_map: dict[str, str] = {}
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

    return (
        RuntimeGraph(
            nodes=nodes,
            node_order=node_order,
            output_node_map=output_node_map,
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
        output_llmop_to_outputop: dict[str, str] = {}
        for op_id, op in graph_dict.items():
            if isinstance(op, LLMOp):
                llm_ops[op_id] = op
                visited_node_ids.add(op_id)
            elif isinstance(op, DataRetrievalOp):
                retrieval_ops[op_id] = op
                visited_node_ids.add(op_id)
            if isinstance(op, OutputOp):
                assert len(op.inputs) == 1, "OutputOp should have exactly one input"
                assert isinstance(
                    op.inputs[0], LLMOp
                ), "OutputOp input should be an LLMOp"
                visited_node_ids.add(op_id)
                output_llmop_to_outputop[op.inputs[0].id] = op.name

        if not llm_ops and not retrieval_ops:
            raise ValueError("No LLMOp found in compiled graph")

        nodes: dict[str, RuntimeOp] = {}
        node_order: list[str] = []
        output_node_map: dict[str, str] = {}
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

        for llm_op_id, llm_op in llm_ops.items():
            if task_type_override == "data_profile":
                continue

            runtime_ops: list[RuntimeOp] = []
            mapping: list[str] = []
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

            if llm_op_id in output_llmop_to_outputop:
                output_node_map[llm_op_id] = output_llmop_to_outputop[llm_op_id]

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

    def _build_inference_spec_from_config(self, config: Any) -> dict[str, Any]:
        spec: dict[str, Any] = {}
        if config.max_tokens:
            spec["max_tokens"] = config.max_tokens
        if config.temperature is not None:
            spec["temperature"] = config.temperature
        if config.top_p is not None:
            spec["top_p"] = config.top_p
        if config.chat_template_kwargs:
            spec["chat_template_kwargs"] = config.chat_template_kwargs
        return spec

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
    ) -> RuntimeOp:
        data_spec = self._attach_s3_cfg(data_spec)
        return RuntimeOp(
            node_id=name,
            task_type=task_type,
            backend=backend,
            model=model,
            data_spec=data_spec,
            model_spec=model_spec,
            inference_spec=inference_spec,
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
        if backend == "vllm":
            spec["vllm"] = backend_config or self._build_default_vllm_backend_config()
        elif backend == "transformers":
            spec["transformers"] = backend_config or {
                "mode": "visual-embedding",
                "device_map": "auto",
                "trust_remote_code": True,
            }
        elif backend == "diffusers":
            spec["diffusers"] = backend_config or {
                "dtype": "bf16",
                "use_safetensors": True,
            }
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

    def _build_s3_cfg_from_env(self) -> dict[str, Any] | None:
        if not envs.S3_URL:
            return None
        cfg: dict[str, Any] = {"connection_string": envs.S3_URL, "encoding": "utf-8"}
        cert_data = self._resolve_s3_cert_data({})
        if cert_data is not None:
            cfg["cert_data"] = cert_data
        return cfg

    def _resolve_s3_cert_data(self, spec: dict[str, Any]) -> str | None:
        """Return cert_data for an S3 spec — explicit on the spec wins, else env.

        Parsers stay environment-agnostic (don't read ``S3_CERT_FILE`` at
        parse time), so the same workflow compiles to identical specs across
        environments. Cert resolution lives here, at the runtime-builder
        layer that already touches the worker config.
        """
        cert_data = spec.get("cert_data")
        if isinstance(cert_data, str):
            return cert_data
        if not envs.S3_CERT_FILE:
            return None
        try:
            return Path(envs.S3_CERT_FILE).read_text(encoding="utf-8")
        except OSError as exc:
            self.logger.warning(
                "Failed to read S3 cert from %s: %s",
                envs.S3_CERT_FILE,
                exc,
            )
            return None

    def _attach_s3_cfg(self, data_spec: dict[str, Any]) -> dict[str, Any]:
        if data_spec.get("type") != "list" or "s3_cfg" in data_spec:
            return data_spec
        items = data_spec.get("items")
        if not isinstance(items, list):
            return data_spec
        has_s3 = any(
            isinstance(item, str) and item.startswith("s3://") for item in items
        )
        if not has_s3:
            return data_spec
        s3_cfg = self._build_s3_cfg_from_env()
        if not s3_cfg:
            raise ValueError("S3_URL is required for s3:// list inputs")
        updated = dict(data_spec)
        updated["s3_cfg"] = s3_cfg
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
                    columns.append({"label": label, "node": node_ref, "path": path})
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

        inference_spec = self._build_inference_spec_from_config(llm_op.config)
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
        spec_type = spec.get("type")
        if spec_type not in {"sql", "s3", "agent"}:
            raise ValueError(
                f"Unsupported DataRetrievalOp type for {op_id}: {spec_type}"
            )

        params = spec.get("params") or []
        if not isinstance(params, list):
            raise ValueError(f"DataRetrievalOp {op_id} params must be a list")

        connection_string: str | None = None
        template: str | None = None
        if spec_type in {"sql", "s3"}:
            connection_string = spec.get("connection_string")
            template = spec.get("template")
            if not isinstance(connection_string, str) or not isinstance(template, str):
                raise ValueError(
                    f"DataRetrievalOp {op_id} missing connection_string/template"
                )
        else:
            description = spec.get("description")
            schema_scope = spec.get("schema_scope")
            if not isinstance(description, str):
                raise ValueError(
                    f"DataRetrievalOp {op_id} (type=agent) requires description"
                )
            if schema_scope is not None and not isinstance(schema_scope, str):
                raise ValueError(
                    f"DataRetrievalOp {op_id} (type=agent) "
                    "schema_scope must be a string"
                )
            if not envs.LUMID_DATA_URL:
                raise ValueError(
                    f"DataRetrievalOp {op_id} (type=agent) requires the lumilake "
                    "server to be configured with LUMID_DATA_URL "
                    "(see .env.example's LUMID_DATA_URL)"
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
                if node not in seen:
                    seen.add(node)
                    dependencies.append(node)
            resolved_params.append(param)

        if spec_type in {"sql", "s3"}:
            template, resolved_params = _inline_single_value_list_params(
                template, resolved_params
            )

        data_spec: dict[str, Any] = {
            "type": spec_type,
            "params": resolved_params,
        }
        if spec_type in {"sql", "s3"}:
            data_spec["connection_string"] = connection_string
            data_spec["template"] = template
        if spec_type == "sql":
            table = spec.get("table")
            if isinstance(table, str) and table.strip():
                data_spec["table"] = table.strip()
            else:
                try:
                    data_spec["table"] = self._extract_table_from_sql_template(template)
                except ValueError:
                    pass
        if spec_type == "s3":
            data_spec["encoding"] = spec.get("encoding", "utf-8")
            cert_data = self._resolve_s3_cert_data(spec)
            if cert_data is not None:
                data_spec["cert_data"] = cert_data
        if spec_type == "agent":
            data_spec["description"] = template
            if spec.get("schema_scope"):
                data_spec["schema_scope"] = spec["schema_scope"]
            data_spec["lumid_data_url"] = envs.LUMID_DATA_URL
            if envs.LUMID_DATA_TOKEN:
                data_spec["lumid_data_token"] = envs.LUMID_DATA_TOKEN
            for optional in ("output_format", "max_steps", "model", "verify"):
                if optional in spec:
                    data_spec[optional] = spec[optional]

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

    @staticmethod
    def _resolve_profile_param(
        param: Any,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
    ) -> dict[str, Any] | None:
        try:
            upstream = graph_dict[param["node"]]
        except (KeyError, TypeError):
            return param if isinstance(param, dict) and "data" in param else None
        if not isinstance(upstream, InputOp):
            return param if isinstance(param, dict) and "data" in param else None
        values = inputs_dict.get(upstream.name) or []
        if not values:
            return None
        return {
            "label": param.get("label"),
            "data": {"type": "list", "items": list(values)},
        }

    def _build_data_profile_node_from_data_retrieval_op(
        self,
        op_id: str,
        op: DataRetrievalOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
    ) -> RuntimeOp | None:
        spec = op.data_spec or {}
        spec_type = spec.get("type")
        if spec_type not in {"sql", "s3"}:
            return None
        connection_string = spec.get("connection_string")
        template = spec.get("template")
        params = spec.get("params") or []
        if not isinstance(connection_string, str) or not isinstance(template, str):
            raise ValueError(
                f"DataRetrievalOp {op_id} missing connection_string/template"
            )
        if not isinstance(params, list):
            raise ValueError(f"DataRetrievalOp {op_id} params must be a list")
        if spec_type == "sql":
            params = [
                resolved
                for param in params
                if (
                    resolved := self._resolve_profile_param(
                        param, graph_dict, inputs_dict
                    )
                )
                is not None
            ]
            constraints = self._build_data_profile_constraints(
                spec.get("params") or [], graph_dict
            )
            if not constraints:
                constraints = self._build_data_profile_constraints(params, graph_dict)

            table = spec.get("table")
            if not isinstance(table, str):
                table = self._extract_table_from_sql_template(template)
            data_spec: dict[str, Any] = {
                "type": "sql",
                "connection_string": connection_string,
                "template": template,
                "params": params,
                "constraints": constraints,
                "num_test_queries": envs.LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES,
                "table": table,
            }
        else:
            data_spec = {
                "type": "s3",
                "connection_string": connection_string,
                "template": template,
                "params": params,
                "encoding": spec.get("encoding", "utf-8"),
            }
            cert_data = self._resolve_s3_cert_data(spec)
            if cert_data is not None:
                data_spec["cert_data"] = cert_data
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
        if isinstance(llm_op, ImageGenerationOp):
            visited_node_ids.add(llm_op_id)
            visited_node_ids.add(llm_op.content.id)
            inference_spec = self._build_inference_spec_from_config(llm_op.config)
            inference_spec.update(
                {
                    "num_inference_steps": 8,
                    "guidance_scale": 1.0,
                    "height": 1024,
                    "width": 1024,
                }
            )
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
                content_data_spec = {
                    "type": "list",
                    "node": content_op.id,
                    "path": "items.output",
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

        inference_spec = self._build_inference_spec_from_config(llm_op.config)
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
                    columns.append({"label": label, "node": node_ref, "path": path})
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
                    {
                        "label": label,
                        "node": node_ref,
                        "path": path,
                    }
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
            condition=llm_op.condition if isinstance(llm_op, LLMChatOp) else None,
        )

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
                if op.id not in upstream_llm_ids:
                    upstream_llm_ids.add(op.id)
                    if isinstance(op, LLMChatOp) and op.return_history:
                        columns[f"{op.id}_context"] = {
                            "node": op.id,
                            "path": "items.metadata.prompt",
                        }
                    columns[f"{op.id}_output"] = {
                        "node": op.id,
                        "path": "items.output",
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
                    retrieval_type = data_spec.get("type")
                    if retrieval_type == "sql":
                        path = "items.table"
                    elif retrieval_type == "s3":
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
                        for packed_msg in input_messages:
                            assert isinstance(
                                packed_msg, tuple
                            ), "MessageOp's messages must be context-free strings"
                            role, msg = packed_msg
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
            structural_outputs = getattr(op, "structural_outputs", None)
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
