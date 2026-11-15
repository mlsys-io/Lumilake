from lumilake_server.common import (
    retrieval_items_path,
)
from lumilake_server.graphs import CompiledGraph
from lumilake_server.ops import (
    DataRetrievalOp,
    LambdaOp,
    LLMOp,
    OutputOp,
)
from lumilake_server.ops.llm_ops import (
    LLMChatOp,
    LLMVisionOp,
)
from lumilake_server.runtime.runtime_graph.graph import (
    RuntimeGraph,
    make_node_prefix,
)
from lumilake_server.runtime.runtime_graph.vision import _VisionBuilder
from lumilake_server.runtime.runtime_ops import RuntimeOp


class RuntimeGraphBuilder(_VisionBuilder):
    def build(
        self,
        compiled_graph: CompiledGraph,
        task_type_override: str | None = None,
        node_prefix: str | None = None,
    ) -> RuntimeGraph:
        graph_dict = compiled_graph.graph.as_dict()
        inputs_dict = compiled_graph.inputs
        input_ops_by_id = {op.id: op for op in compiled_graph.graph.input_ops.values()}
        self._inputs_dict = inputs_dict
        self._graph_dict = graph_dict
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
