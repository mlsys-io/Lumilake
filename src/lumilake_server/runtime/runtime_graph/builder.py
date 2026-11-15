from lumilake_server.graphs import CompiledGraph
from lumilake_server.ops import (
    DataRetrievalOp,
    LambdaOp,
    LLMOp,
    Op,
    OutputOp,
)
from lumilake_server.ops.llm_ops import LLMVisionOp
from lumilake_server.runtime.runtime_graph.graph import (
    RuntimeGraph,
    make_node_prefix,
)
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
    Phase,
    lookup,
)
from lumilake_server.runtime.runtime_graph.vision import _VisionBuilder
from lumilake_server.runtime.runtime_ops import RuntimeOp


class RuntimeGraphBuilder(_VisionBuilder):
    def _lower_ops(self, ctx: LoweringContext) -> None:
        """Run every registered lowering, one phase at a time, recording the
        nodes, their emission order, the DSL-to-runtime map and the visited
        ops."""
        for phase in Phase:
            for op_id, op in ctx.graph_dict.items():
                lowering = lookup(op, phase)
                if lowering is None:
                    continue
                result = lowering(self, op_id, op, ctx)
                if result is None:
                    continue
                for node in result.nodes:
                    if node.node_id in ctx.nodes:
                        continue
                    ctx.nodes[node.node_id] = node
                    ctx.node_order.append(node.node_id)
                ctx.dsl_to_runtime[op_id] = list(result.dsl_to_runtime)
                ctx.visited.update(result.visited)
                ctx.visited.add(op_id)

    def _record_output(
        self,
        op_id: str,
        op: Op,
        result: LoweringResult,
        ctx: LoweringContext,
    ) -> None:
        """Record the path an OutputOp archives ``result``'s nodes at."""
        outputs = ctx.outputs_by_source.get(op_id)
        if outputs is None:
            return
        ((output_name, path_override),) = outputs
        for node_id in result.output_nodes:
            ctx.output_node_map[node_id] = output_name
            if path_override:
                ctx.output_paths[node_id] = path_override
            elif default_path := ctx.op_output(op).default_output_path:
                ctx.output_paths[node_id] = default_path

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
        output_source_to_outputop: dict[str, list[tuple[str, str | None]]] = {}
        for op_id, op in graph_dict.items():
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

        has_runtime_op = any(
            isinstance(op, (LLMOp, DataRetrievalOp, LambdaOp))
            for op in graph_dict.values()
        )
        if not has_runtime_op:
            raise ValueError(
                "Graph must contain at least one LLMOp, DataRetrievalOp or LambdaOp"
            )

        nodes: dict[str, RuntimeOp] = {}
        node_order: list[str] = []
        output_node_map: dict[str, str] = {}
        output_paths: dict[str, str] = {}
        dsl_to_runtime: dict[str, list[str]] = {}

        ctx = LoweringContext(
            builder=self,
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
            input_ops_by_id=input_ops_by_id,
            task_type_override=task_type_override,
            nodes=nodes,
            node_order=node_order,
            dsl_to_runtime=dsl_to_runtime,
            visited=visited_node_ids,
            outputs_by_source=output_source_to_outputop,
            output_node_map=output_node_map,
            output_paths=output_paths,
        )
        self._lower_ops(ctx)

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
