import re
from collections.abc import Sequence
from typing import Any, cast

from lumilake import envs
from lumilake_server.common import (
    ApiConfig,
    retrieval_items_path,
)
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
from lumilake_server.ops.llm_ops import (
    ImageGenerationOp,
    LLMChatOp,
    LLMCompletionOp,
    LLMVisionOp,
)
from lumilake_server.ops.util_ops import RowLambdaFn, SubmittedFunction
from lumilake_server.runtime import python_step
from lumilake_server.runtime.flowmesh_client import (
    is_api_origin_trusted,
    resolve_api_credential,
)
from lumilake_server.runtime.runtime_graph.base import (
    _default_output_destination,
)
from lumilake_server.runtime.runtime_graph.graph import (
    _API_CREDENTIAL_PLACEHOLDER,
    Roles,
    RuntimeGraph,
    make_node_prefix,
)
from lumilake_server.runtime.runtime_graph.retrieval import (
    _RetrievalBuilder,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils import lambda_runtime


class RuntimeGraphBuilder(_RetrievalBuilder):
    def build(
        self,
        compiled_graph: CompiledGraph,
        task_type_override: str | None = None,
        node_prefix: str | None = None,
    ) -> RuntimeGraph:
        graph_dict = compiled_graph.graph.as_dict()
        inputs_dict = compiled_graph.inputs
        input_ops_by_id = {op.id: op for op in compiled_graph.graph.input_ops.values()}
        self._flattened_vlm_ids = {
            op_id
            for op_id, op in graph_dict.items()
            if isinstance(op, LLMVisionOp)
            and op.rowwise_template
            and isinstance(graph_dict.get(op.image_source), DataRetrievalOp)
        }

        visited_node_ids: set[str] = set()
        llm_ops: dict[str, LLMOp] = {}
        retrieval_ops: dict[str, DataRetrievalOp] = {}
        list_lambda_ops: dict[str, LambdaOp] = {}
        output_source_to_outputop: dict[str, list[tuple[str, str | None]]] = {}
        for op_id, op in graph_dict.items():
            if isinstance(op, LLMOp):
                llm_ops[op_id] = op
                visited_node_ids.add(op_id)
            elif isinstance(op, DataRetrievalOp):
                retrieval_ops[op_id] = op
                visited_node_ids.add(op_id)
            elif isinstance(op, LambdaOp) and op.takes_whole_columns:
                list_lambda_ops[op_id] = op
                visited_node_ids.add(op_id)
            if isinstance(op, OutputOp):
                assert len(op.inputs) == 1, "OutputOp should have exactly one input"
                source = op.inputs[0]
                if not isinstance(source, (LLMOp, DataRetrievalOp, LambdaOp)):
                    raise ValueError(
                        f"OutputOp '{op.name}' input must be an LLMOp, "
                        f"DataRetrievalOp or LambdaOp (got {type(source).__name__})"
                    )
                visited_node_ids.add(op_id)
                output_source_to_outputop.setdefault(source.id, []).append(
                    (op.name, op.path)
                )

        for source_id, outputs in output_source_to_outputop.items():
            if len(outputs) > 1:
                names = ", ".join(name for name, _ in outputs)
                raise ValueError(
                    f"Multiple OutputOps ({names}) target the same source node "
                    f"'{source_id}'. A runtime node produces a single output; "
                    "project multiple outputs from one source by adding an "
                    "intermediate LambdaOp instead."
                )

        has_lambda = any(isinstance(op, LambdaOp) for op in graph_dict.values())
        if not llm_ops and not retrieval_ops and not has_lambda:
            raise ValueError(
                "Graph must contain at least one LLMOp, DataRetrievalOp or LambdaOp"
            )

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
                    input_ops_by_id,
                )
                if runtime_op is None:
                    continue
            else:
                runtime_op = self._build_node_from_data_retrieval_op(
                    retrieval_op_id,
                    retrieval_op,
                    graph_dict,
                    inputs_dict,
                    input_ops_by_id,
                )
            nodes[runtime_op.node_id] = runtime_op
            node_order.append(runtime_op.node_id)
            dsl_to_runtime[retrieval_op_id] = [runtime_op.node_id]

            if retrieval_op_id in output_source_to_outputop:
                ((output_name, path_override),) = output_source_to_outputop[
                    retrieval_op_id
                ]
                output_node_map[runtime_op.node_id] = output_name
                mode = retrieval_op.data_spec["mode"]
                # Mode-derived defaults match the FlowMesh executor's item
                # shape; agent replays a SQL plan so it emits ``table`` too.
                default_path = retrieval_items_path(mode)
                output_paths[runtime_op.node_id] = path_override or default_path

        for lambda_op_id, lambda_op in list_lambda_ops.items():
            if task_type_override == "data_profile":
                continue
            for input_op in lambda_op.inputs:
                visited_node_ids.add(input_op.id)
            runtime_op = self._build_node_from_list_lambda_op(
                lambda_op_id,
                lambda_op,
                graph_dict,
                inputs_dict,
            )
            nodes[runtime_op.node_id] = runtime_op
            node_order.append(runtime_op.node_id)
            dsl_to_runtime[lambda_op_id] = [runtime_op.node_id]

            if lambda_op_id in output_source_to_outputop:
                ((output_name, path_override),) = output_source_to_outputop[
                    lambda_op_id
                ]
                output_node_map[runtime_op.node_id] = output_name
                output_paths[runtime_op.node_id] = path_override or "items.output"

        for llm_op_id, llm_op in llm_ops.items():
            if task_type_override == "data_profile":
                continue

            runtime_ops: list[RuntimeOp] = []
            mapping: list[str] = []
            output_node_ids: list[str] = [llm_op_id]
            if isinstance(llm_op, LLMVisionOp):
                runtime_ops, mapping = self._build_vlm_nodes_from_image_op(
                    llm_op_id,
                    llm_op,
                    graph_dict,
                    inputs_dict,
                    visited_node_ids,
                    dsl_to_runtime=dsl_to_runtime,
                    runtime_nodes=nodes,
                )
            elif task_type_override != "data_profile" and llm_op.config.api is not None:
                upstream_llm_ids, template_spec = self._infer_structural_messages(
                    llm_op_id,
                    graph_dict,
                    inputs_dict,
                    visited_node_ids,
                    dsl_to_runtime=dsl_to_runtime,
                    runtime_nodes=nodes,
                    api_mode=True,
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
                ((output_name, path_override),) = output_source_to_outputop[llm_op_id]
                for output_node_id in output_node_ids:
                    output_node_map[output_node_id] = output_name
                    if path_override:
                        output_paths[output_node_id] = path_override
                    elif self._is_api_task(llm_op):
                        output_paths[output_node_id] = self._upstream_output_path(
                            llm_op
                        )

        if task_type_override != "data_profile":
            # A LambdaOp is inlined into the LLM that consumes it (traced above).
            # One that is an output, that no LLM reached, that has an image, or
            # that a runtime node references runs on its own as a python step —
            # see runtime/python_step.py.
            referenced = self._referenced_row_lambdas(nodes, graph_dict)
            standalone = [
                op_id
                for op_id, op in graph_dict.items()
                if isinstance(op, LambdaOp)
                and not op.takes_whole_columns
                and (
                    op_id not in visited_node_ids
                    or op_id in output_source_to_outputop
                    or op.image is not None
                    or op_id in referenced
                )
            ]
            for lambda_id in standalone:
                self._build_python_step(
                    lambda_id,
                    graph_dict,
                    inputs_dict,
                    visited_node_ids,
                    nodes,
                    node_order,
                    dsl_to_runtime,
                )
                if lambda_id in output_source_to_outputop:
                    ((output_name, path_override),) = output_source_to_outputop[
                        lambda_id
                    ]
                    output_node_map[lambda_id] = output_name
                    if path_override:
                        output_paths[lambda_id] = path_override

        if task_type_override != "data_profile":
            self._check_inlined_lambda_refs(nodes, graph_dict)

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

    def _row_condition(
        self,
        condition: dict[str, str] | None,
        consumer_row_count: int,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None,
    ) -> dict[str, str] | None:
        """Validate a condition gates a whole task: fail closed when both the
        condition source and the consumer produce multiple rows, since a single
        gate cannot select which row's value to test."""
        if condition is None:
            return None
        node = condition.get("node")
        if node is None:
            return condition
        source = graph_dict.get(node)
        if source is None:
            return condition
        source_row_count = self._static_output_row_count(
            source, inputs_dict, graph_dict
        )
        if (
            source_row_count is not None
            and source_row_count > 1
            and consumer_row_count > 1
        ):
            raise ValueError(
                f"Condition node '{node}' produces {source_row_count} rows and"
                f" the consumer produces {consumer_row_count} rows; a condition"
                " can only gate a whole task, so this shape is not supported."
            )
        return condition

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
        if len(messages) != 1 or messages[0].role != "user":
            raise ValueError(
                f"LLMChatOp '{op.id}' return_history prior prompt must be a"
                " single user message; API mode cannot replay system messages"
                " or earlier turns with their roles, so this shape is not"
                " supported."
            )
        content = messages[0].content
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
                content_path = (
                    retrieval_items_path(content_op.data_spec["mode"])
                    if isinstance(content_op, DataRetrievalOp)
                    else self._upstream_output_path(content_op)
                )
                content_data_spec = {
                    "type": "list",
                    "node": content_op.id,
                    "path": content_path,
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

        if isinstance(llm_op, LLMCompletionOp):
            # Raw completions read their prompt directly and never build a
            # chat-message graph, including when the prompt is a list-mode op.
            upstream_llm_ids: list[str] = []
            template_spec: dict[str, Any] = {}
        else:
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

        if isinstance(llm_op, LLMCompletionOp):
            prompt_op = llm_op.prompt
            visited_node_ids.add(prompt_op.id)
            if isinstance(prompt_op, InputOp):
                prompt_values = inputs_dict.get(prompt_op.name)
                if prompt_values is None:
                    raise ValueError(
                        f"LLMCompletionOp '{llm_op_id}' prompt input "
                        f"{prompt_op.name!r} has no values supplied"
                    )
                data_spec = {"type": "list", "items": list(prompt_values)}
                row_dependencies: list[str] = []
            elif isinstance(prompt_op, DataOp):
                data_spec = {"type": "list", "items": list(prompt_op.data)}
                row_dependencies = []
            else:
                if not (
                    isinstance(prompt_op, LambdaOp)
                    or self._emits_one_item_per_row(prompt_op)
                ):
                    raise ValueError(
                        f"LLMCompletionOp '{llm_op_id}' prompt upstream"
                        f" '{prompt_op.id}' ({type(prompt_op).__name__}) does"
                        " not produce one prompt string per row"
                    )
                if isinstance(prompt_op, LambdaOp):
                    prompt_path = self._lambda_read_path(
                        f"LLMCompletionOp '{llm_op_id}'",
                        prompt_op.id,
                        "items.output",
                        prompt_op,
                    )
                else:
                    prompt_path = self._upstream_output_path(prompt_op)
                data_spec = {
                    "type": "list",
                    "node": prompt_op.id,
                    "path": prompt_path,
                }
                row_dependencies = [prompt_op.id]
            return self._create_runtime_op(
                name=llm_op_id,
                task_type=task_type,
                data_spec=data_spec,
                model_spec=self._build_model_spec(llm_op.config, backend),
                inference_spec={**inference_spec, "apply_chat_template": False},
                backend=backend,
                model=llm_op.config.model,
                dependencies=row_dependencies if row_dependencies else None,
                output_spec=output_spec,
                condition=None,
            )

        if isinstance(llm_op, LLMChatOp) and llm_op.rowwise_template:
            data_spec, row_dependencies = self._rowwise_data_spec(
                llm_op_id=llm_op_id,
                llm_op=llm_op,
                upstream_llm_ids=upstream_llm_ids,
                graph_dict=graph_dict,
                inputs_dict=inputs_dict,
                dsl_to_runtime=dsl_to_runtime,
            )
            return self._create_runtime_op(
                name=llm_op_id,
                task_type=task_type,
                data_spec=data_spec,
                model_spec=self._build_model_spec(llm_op.config, backend),
                inference_spec=inference_spec,
                backend=backend,
                model=llm_op.config.model,
                dependencies=row_dependencies if row_dependencies else None,
                output_spec=output_spec,
                condition=self._row_condition(
                    llm_op.condition,
                    self._static_output_row_count(llm_op, inputs_dict, graph_dict) or 1,
                    graph_dict,
                    inputs_dict,
                    dsl_to_runtime,
                ),
            )

        if isinstance(llm_op, LLMChatOp) and llm_op.aggregate_table:
            data_spec, aggregate_dependencies = self._aggregate_data_spec(
                llm_op_id=llm_op_id,
                llm_op=llm_op,
                template_spec=template_spec,
                upstream_llm_ids=upstream_llm_ids,
                graph_dict=graph_dict,
                inputs_dict=inputs_dict,
                dsl_to_runtime=dsl_to_runtime,
            )
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
                    self._static_output_row_count(llm_op, inputs_dict, graph_dict) or 1,
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
                self._static_output_row_count(llm_op, inputs_dict, graph_dict) or 1,
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
            headers["Authorization"] = _API_CREDENTIAL_PLACEHOLDER
        elif api_config.authorization:
            headers["Authorization"] = _API_CREDENTIAL_PLACEHOLDER
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
        """Dispatch an API LLM op to its row-wise, aggregate or plain builder."""
        if isinstance(llm_op, LLMChatOp) and llm_op.rowwise_template:
            return self._build_api_rowwise_op(
                llm_op_id=llm_op_id,
                llm_op=llm_op,
                api_config=api_config,
                upstream_llm_ids=upstream_llm_ids,
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
        return self._build_api_task(
            llm_op_id=llm_op_id,
            llm_op=llm_op,
            api_config=api_config,
            data_spec={"type": "graph_template", "template": template_spec},
            dependencies=merged_dependencies or None,
            output_spec=output_spec,
            condition=condition,
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
            dsl_to_runtime=dsl_to_runtime,
        )

    def _build_api_task(
        self,
        *,
        llm_op_id: str,
        llm_op: LLMOp,
        api_config: ApiConfig,
        data_spec: dict[str, Any],
        dependencies: list[str] | None,
        output_spec: dict[str, Any] | None,
        condition: dict[str, str] | None,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None,
    ) -> list[RuntimeOp]:
        """Build the single FlowMesh ``api`` task for an externally-hosted LLM:
        the data_spec matches the local path, and the executor substitutes each
        row's rendered messages into the ``{{prompt}}`` body placeholder."""
        model = llm_op.config.resolved_model()
        url = api_config.url or envs.LUMILAKE_API_DEFAULT_URL
        headers = self._api_request_headers(llm_op_id, api_config, url)
        body: dict[str, Any] = {"model": model, "messages": "{{prompt}}"}
        body.update(llm_op.config.inference_spec())
        if isinstance(llm_op, LLMChatOp) and llm_op.structural_outputs is not None:
            if isinstance(llm_op.structural_outputs, dict):
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structural_outputs",
                        "schema": llm_op.structural_outputs,
                        "strict": True,
                    },
                }
            else:
                raise ValueError(
                    "api mode takes structural_outputs as a JSON-schema object; "
                    f"got {type(llm_op.structural_outputs).__name__}"
                )
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
        if api_config.retries is not None:
            api_spec["retries"] = api_config.retries
        if api_config.concurrency is not None:
            api_spec["concurrency"] = api_config.concurrency
        return [
            self._create_runtime_op(
                name=llm_op_id,
                task_type="api",
                data_spec=data_spec,
                model_spec={},
                inference_spec={},
                api_spec=api_spec,
                backend="api",
                model=model,
                dependencies=dependencies,
                output_spec=output_spec,
                condition=self._row_condition(
                    condition,
                    self._static_output_row_count(llm_op, inputs_dict, graph_dict) or 1,
                    graph_dict,
                    inputs_dict,
                    dsl_to_runtime,
                ),
            )
        ]

    def _rowwise_data_spec(
        self,
        *,
        llm_op_id: str,
        llm_op: LLMChatOp,
        upstream_llm_ids: list[str],
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Build the dataframe data_spec for a rowwise LLMChatOp, shared by the
        local and API paths."""
        if llm_op.rowwise_template is None:
            raise ValueError(f"LLMChatOp {llm_op_id} has no rowwise_template")
        columns: list[dict[str, Any]] = []
        dependencies = list(upstream_llm_ids) if upstream_llm_ids else []
        for col in llm_op.rowwise_columns or []:
            label = col.get("label")
            data = col.get("data")
            node_ref = col.get("node")
            path = col.get("path")
            if isinstance(label, str) and isinstance(data, dict):
                columns.append({"label": label, "data": data})
                continue
            if isinstance(label, str) and isinstance(node_ref, str):
                if path is None and not isinstance(graph_dict.get(node_ref), InputOp):
                    raise ValueError(
                        f"Column '{label}' of consumer '{llm_op_id}'"
                        f" references node '{node_ref}' without a path;"
                        " only InputOp refs may omit it."
                    )
                columns.append(
                    self._node_ref_column(
                        consumer_id=llm_op_id,
                        label=label,
                        node_ref=node_ref,
                        path=path if isinstance(path, str) else "",
                        upstream=graph_dict.get(node_ref),
                        inputs_dict=inputs_dict,
                        graph_dict=graph_dict,
                        dsl_to_runtime=dsl_to_runtime,
                        kind="rowwise column",
                    )
                )
                # An InputOp ref is bound as a literal column (no runtime node),
                # so it must not become a dependency on a node that does not
                # exist.
                if node_ref not in dependencies and not isinstance(
                    graph_dict.get(node_ref), InputOp
                ):
                    dependencies.append(node_ref)
                continue
            raise ValueError(
                f"Column '{label}' of consumer '{llm_op_id}' has neither"
                " data nor node+path"
            )
        if not columns:
            raise ValueError(f"LLMChatOp {llm_op_id} has empty rowwise_columns")
        messages: list[dict[str, str]] = []
        for system_msg in llm_op.system_messages or []:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": llm_op.rowwise_template})
        return {
            "type": "dataframe",
            "columns": columns,
            "messages": messages,
        }, dependencies

    def _aggregate_data_spec(
        self,
        *,
        llm_op_id: str,
        llm_op: LLMChatOp,
        template_spec: dict[str, Any],
        upstream_llm_ids: list[str],
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Build the graph_template data_spec for an aggregate LLMChatOp,
        shared by the local and API paths."""
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
        table_columns: list[dict[str, Any]] = []
        dependencies = list(upstream_llm_ids) if upstream_llm_ids else []
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
            # An InputOp ref is bound as a literal column (no runtime node), so
            # it must not become a dependency on a node that does not exist.
            if node_ref not in dependencies and not isinstance(
                graph_dict.get(node_ref), InputOp
            ):
                dependencies.append(node_ref)
        for dep in self._collect_graph_template_dependencies(template_spec):
            if dep not in dependencies:
                dependencies.append(dep)

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
        return data_spec, dependencies

    def _build_api_rowwise_op(
        self,
        *,
        llm_op_id: str,
        llm_op: LLMChatOp,
        api_config: ApiConfig,
        upstream_llm_ids: list[str],
        output_spec: dict[str, Any] | None,
        condition: dict[str, str] | None,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        dsl_to_runtime: dict[str, list[str]] | None = None,
    ) -> list[RuntimeOp]:
        """Build the single API task for a rowwise LLMChatOp: rows are resolved
        by the executor from upstream results, so the op emits one runtime task
        whose data_spec matches the local rowwise path."""
        data_spec, dependencies = self._rowwise_data_spec(
            llm_op_id=llm_op_id,
            llm_op=llm_op,
            upstream_llm_ids=upstream_llm_ids,
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
            dsl_to_runtime=dsl_to_runtime,
        )
        return self._build_api_task(
            llm_op_id=llm_op_id,
            llm_op=llm_op,
            api_config=api_config,
            data_spec=data_spec,
            dependencies=dependencies or None,
            output_spec=output_spec,
            condition=condition,
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
            dsl_to_runtime=dsl_to_runtime,
        )

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
        """Build the single API task for an aggregate LLMChatOp: the data_spec
        matches the local aggregate path, and rows are resolved by the executor
        from upstream results."""
        data_spec, aggregate_dependencies = self._aggregate_data_spec(
            llm_op_id=llm_op_id,
            llm_op=llm_op,
            template_spec=template_spec,
            upstream_llm_ids=upstream_llm_ids,
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
            dsl_to_runtime=dsl_to_runtime,
        )
        return self._build_api_task(
            llm_op_id=llm_op_id,
            llm_op=llm_op,
            api_config=api_config,
            data_spec=data_spec,
            dependencies=aggregate_dependencies or None,
            output_spec=output_spec,
            condition=condition,
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
            dsl_to_runtime=dsl_to_runtime,
        )

    def _infer_structural_messages(
        self,
        llm_op_id: str,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited_node_ids: set[str],
        dsl_to_runtime: dict[str, list[str]] | None = None,
        runtime_nodes: dict[str, RuntimeOp] | None = None,
        api_mode: bool = False,
    ) -> tuple[list[str], dict[str, Any]]:
        target_llm_op = graph_dict[llm_op_id]
        assert isinstance(target_llm_op, LLMOp), "Target op must be an LLMOp"
        upstream_llm_ids: set[str] = set()

        columns: dict[str, dict[str, Any]] = {}
        steps: dict[
            str, tuple[str, Sequence[str | dict[str, str] | list[dict[str, str]]], bool]
        ] = {}

        ancestor_buffer: dict[str, list[str | tuple[Roles, str]]] = {}
        ancestor_row_counts: dict[str, int] = {}

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
                is_api_ancestor = self._is_api_task(op)
                if op.id not in upstream_llm_ids:
                    upstream_llm_ids.add(op.id)
                    if not is_api_ancestor:
                        row_count = self._static_output_row_count(
                            op, inputs_dict, graph_dict
                        )
                        if row_count is not None and row_count > 1:
                            if api_mode or not self._emits_one_item_per_row(op):
                                raise ValueError(
                                    f"LLMOp '{llm_op_id}' consumes '{op.id}', which"
                                    " produces multiple rows; a message column"
                                    " binds row-wise only from a plain local"
                                    " LLMChatOp into a local consumer, so wiring"
                                    " this node to the single unsuffixed output"
                                    " would silently drop every row but the first."
                                )
                            ancestor_row_counts[op.id] = row_count
                    if isinstance(op, LLMChatOp) and op.return_history:
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
                    if len(inputs_dict[op.name]) > 1:
                        ancestor_row_counts[op.id] = len(inputs_dict[op.name])
                ancestor_buffer[op.id] = [(Roles.USER, op.id)]

            elif isinstance(op, DataOp):
                if op.id not in columns:
                    columns[op.id] = {"data": {"type": "list", "items": op.data}}
                ancestor_buffer[op.id] = [(Roles.USER, op.id)]

            elif isinstance(op, DataRetrievalOp):
                if op.id not in columns:
                    path = retrieval_items_path(op.data_spec["mode"])
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
                if op.format_kwargs:
                    assert (
                        len(op.inputs) >= 1
                    ), "FormatOp should have at least one input"
                # A field-referenced input binds a column (value.items.value.<field>)
                # and runs as its own python step; it must not also be traced into
                # the message chain as an inline function step. But an input read
                # both by field and as a whole item still needs tracing for the
                # whole-item kwarg, so only skip it when every reference is a field.
                field_ref_ids = {op.format_kwargs[k].id for k in op.format_paths}
                whole_ref_ids = {
                    op.format_kwargs[k].id
                    for k in op.format_kwargs
                    if k not in op.format_paths
                }
                skip_trace_ids = field_ref_ids - whole_ref_ids
                message_labels = {
                    inp_op.id: _trace_ancestors(inp_op)
                    for inp_op in op.inputs
                    if inp_op.id not in skip_trace_ids
                }
                # A field-referenced InputOp is projected server-side into a
                # literal column and never materializes a runtime node; mark it
                # visited so the traversal does not treat it as orphaned.
                for field_ref_id in field_ref_ids:
                    if isinstance(graph_dict.get(field_ref_id), InputOp):
                        visited_node_ids.add(field_ref_id)
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
                format_kwargs = []
                for k, v in op.format_kwargs.items():
                    field_path = op.format_paths.get(k)
                    if field_path:
                        upstream = graph_dict.get(v.id)
                        if isinstance(upstream, InputOp):
                            values = inputs_dict.get(upstream.name)
                            if values is None:
                                raise ValueError(
                                    f"FormatOp '{op.id}' references InputOp"
                                    f" '{upstream.name}' with field path"
                                    f" '{field_path}' but no values were supplied"
                                    " for that input."
                                )
                            label = f"ref_{op.id}_{k}"
                            columns[label] = {
                                "label": label,
                                "data": {
                                    "type": "list",
                                    "items": self._project_input_field(
                                        values, field_path, upstream.name
                                    ),
                                },
                            }
                        else:
                            label = f"ref_{op.id}_{k}"
                            columns[label] = self._node_ref_column(
                                consumer_id=llm_op_id,
                                label=label,
                                node_ref=v.id,
                                path=field_path,
                                upstream=upstream,
                                inputs_dict=inputs_dict,
                                graph_dict=graph_dict,
                                dsl_to_runtime=dsl_to_runtime,
                                kind="prompt field",
                            )
                        format_kwargs.append({"label": k, "value": label})
                    else:
                        format_kwargs.append(
                            {"label": k, "value": _unwrap_msg(message_labels[v.id][0])}
                        )
                label = f"format_{op.id}"
                steps[label] = (template, format_kwargs, False)
                ancestor_buffer[op.id] = [(Roles.USER, label)]

            elif isinstance(op, LambdaOp):
                if op.image is not None:
                    raise ValueError(
                        f"LLMOp '{llm_op_id}' consumes image-bearing LambdaOp"
                        f" '{op.id}' through its message chain; a LambdaOp with"
                        " an image always runs as its own FlowMesh python task,"
                        " so read it through a node column (rowwise_columns /"
                        " aggregate_table) instead."
                    )
                if op.takes_whole_columns:
                    raise ValueError(
                        f"LLMOp '{llm_op_id}' consumes list-mode LambdaOp '{op.id}' "
                        "through its message chain; a list Lambda must be read "
                        "through a node column (rowwise_columns / aggregate_table)."
                    )
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
                # Submitted code is never called here; it stays a function step.
                can_evaluate = not isinstance(op.fn, SubmittedFunction)
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
                    # List-mode LambdaOps raise above, so this call is row-mode.
                    row_fn = cast(RowLambdaFn, op.fn)
                    columns[label] = {
                        "data": {
                            "type": "list",
                            "items": [str(row_fn(tuple(literal_args)))],
                        }
                    }
                    ancestor_buffer[op.id] = [(Roles.USER, label)]
                else:
                    if api_mode:
                        raise ValueError(
                            f"LLMChatOp '{llm_op_id}' API mode cannot render message"
                            f" step '{op.id}': a Lambda message transform is not"
                            " supported in API mode. The server does not execute"
                            " LambdaOp code, and an API request body cannot carry a"
                            " graph_template function step. Make the LambdaOp a"
                            " workflow output (it runs as an isolated FlowMesh python"
                            " task) or use a non-API LLMChatOp."
                        )
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
                    if isinstance(op.fn, SubmittedFunction):
                        # Inlined steps run in the FlowMesh worker's safe_eval
                        # namespace, stricter than a standalone python task.
                        try:
                            lambda_runtime.validate_inline_source(
                                op.code, op.fn.__name__
                            )
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid LambdaOp '{op.id}': {exc}"
                            ) from exc
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
        if len(set(ancestor_row_counts.values())) > 1:
            raise ValueError(
                f"LLMOp '{llm_op_id}' binds message columns of different row"
                f" counts ({ancestor_row_counts}); every multi-row column must"
                " carry the same number of rows."
            )

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

    def _build_vlm_nodes_from_image_op(
        self,
        llm_op_id: str,
        llm_op: LLMVisionOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited_node_ids: set[str],
        dsl_to_runtime: dict[str, list[str]] | None = None,
        runtime_nodes: dict[str, RuntimeOp] | None = None,
    ) -> tuple[list[RuntimeOp], list[str]]:
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
                if isinstance(label, str) and isinstance(node_ref, str):
                    if path is None and not isinstance(
                        graph_dict.get(node_ref), InputOp
                    ):
                        raise ValueError(
                            f"Column '{label}' of consumer '{llm_op_id}'"
                            f" references node '{node_ref}' without a path;"
                            " only InputOp refs may omit it."
                        )
                    columns.append(
                        self._node_ref_column(
                            consumer_id=llm_op_id,
                            label=label,
                            node_ref=node_ref,
                            path=path if isinstance(path, str) else "",
                            upstream=graph_dict.get(node_ref),
                            inputs_dict=inputs_dict,
                            graph_dict=graph_dict,
                            dsl_to_runtime=dsl_to_runtime,
                            kind="rowwise column",
                        )
                    )
                    continue
                raise ValueError(
                    f"Column '{label}' of consumer '{llm_op_id}' has neither"
                    " data nor node+path"
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

        if llm_op_id not in self._flattened_vlm_ids:
            template_spec["columns"] = list(template_spec.get("columns") or []) + [
                batch_column
            ]
            template_dependencies = self._collect_graph_template_dependencies(
                template_spec
            )
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
                model_spec=self._build_model_spec(
                    llm_op.config, "vllm", backend_config
                ),
                inference_spec=inference_spec,
                backend="vllm",
                model=llm_op.config.model,
                dependencies=deduped_deps,
            )
            return [embedding_node, vlm_node], [embedding_node_id, llm_op_id]

        # Flattened per-row VLM: one embedding + one inference over the flat
        # prompts, plus a flatten python step (renders one prompt per image)
        # and a regroup python step (back to one item per symbol). The image
        # source is the group boundary: it defines the per-symbol image counts,
        # and every other grouped column must match them.
        image_path = embedding_data_spec["path"]
        if not image_path.startswith("items."):
            raise ValueError(
                f"Per-row VLM '{llm_op_id}' image path '{image_path}' must"
                " start with 'items.'"
            )
        image_entry = {
            "kind": "stage",
            "node": embedding_data_spec["node"],
            "path": image_path[len("items.") :],
            "label": batch_column["label"],
            "grouped": True,
        }
        plan: list[dict[str, Any]] = [image_entry]
        for c in columns:
            if "data" in c:
                plan.append(
                    {
                        "kind": "literal",
                        "values": c["data"]["items"],
                        "label": c["label"],
                        "grouped": False,
                    }
                )
                continue
            up = graph_dict.get(c["node"])
            if up is None:
                raise ValueError(
                    f"Per-row VLM '{llm_op_id}' column '{c['label']}' reads node"
                    f" '{c['node']}', which is not reachable in the graph."
                )
            if isinstance(up, LambdaOp) and up.takes_whole_columns:
                raise ValueError(
                    f"Column '{c['label']}' of consumer '{llm_op_id}' references"
                    f" list-mode Lambda '{c['node']}'; a flattened per-row VLM"
                    " cannot read a whole-column Lambda."
                )
            if not c["path"].startswith("items."):
                raise ValueError(
                    f"Column '{c['label']}' of consumer '{llm_op_id}' reads"
                    f" '{c['path']}' from '{c['node']}'; only 'items.*' paths"
                    " are supported in a flattened per-row VLM."
                )
            plan.append(
                {
                    "kind": "stage",
                    "node": c["node"],
                    "path": c["path"][len("items.") :],
                    "label": c["label"],
                    "grouped": isinstance(up, DataRetrievalOp),
                }
            )

        flatten_id = f"{llm_op_id}_flatten"
        flatten_deps = list(
            dict.fromkeys(e["node"] for e in plan if e["kind"] == "stage")
        )
        flatten = self._create_runtime_op(
            name=flatten_id,
            task_type=python_step.TASK_TYPE,
            data_spec={
                "mode": "flatten",
                "plan": plan,
                "template": llm_op.rowwise_template,
                "timeout_s": python_step.DEFAULT_TIMEOUT_SECONDS,
            },
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=flatten_deps,
        )

        infer_id = f"{llm_op_id}_infer"
        infer_template: dict[str, Any] = {
            "name": "format",
            "columns": [
                {"label": "prompt", "node": flatten_id, "path": "value.items.output"},
                {
                    "label": batch_column["label"],
                    "node": flatten_id,
                    "path": "value.items.output",
                },
            ],
            "options": {
                "format": {
                    "messages": messages + [{"role": "user", "content": "{prompt}"}]
                }
            },
        }
        infer = self._create_runtime_op(
            name=infer_id,
            task_type="inference",
            data_spec={
                "type": "graph_template",
                "template": infer_template,
                "image_embedding": {
                    "node": embedding_node_id,
                    "path": "embedding_file",
                },
            },
            model_spec=self._build_model_spec(llm_op.config, "vllm", backend_config),
            inference_spec=inference_spec,
            backend="vllm",
            model=llm_op.config.model,
            dependencies=[flatten_id, embedding_node_id],
        )

        regroup_deps = list(dict.fromkeys([infer_id, flatten_id, image_entry["node"]]))
        regroup = self._create_runtime_op(
            name=llm_op_id,
            task_type=python_step.TASK_TYPE,
            data_spec={
                "mode": "regroup",
                "plan": [
                    {"kind": "stage", "node": infer_id, "label": "outputs"},
                    {
                        "kind": "stage",
                        "node": flatten_id,
                        "path": "row",
                        "label": "rows",
                    },
                    {**image_entry, "label": "groups"},
                ],
                "timeout_s": python_step.DEFAULT_TIMEOUT_SECONDS,
            },
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=regroup_deps,
        )

        return (
            [flatten, embedding_node, infer, regroup],
            [flatten_id, embedding_node_id, infer_id, llm_op_id],
        )

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
                llm_op.image_path
                if llm_op.image_path != "images"
                else retrieval_items_path(image_source_op.data_spec["mode"])
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
