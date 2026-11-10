from lumilake_server.graphs import CompiledGraph
from lumilake_server.ops import LLMChatOp, LLMOp, LLMVisionOp, Op, OutputOp
from lumilake_server.runtime.runtime_graph.graph import (
    RuntimeGraph,
    make_node_prefix,
)
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    Phase,
    has_lowering,
    lookup,
    phase_hook,
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
        self._flattened_vlm_ids = self._flattened_vision_ids(graph_dict)

        visited: set[str] = set()
        self._reject_unsupported_output_enforcement(graph_dict)
        output_source_to_outputop = self._collect_outputs(graph_dict, visited)
        if not any(has_lowering(op) for op in graph_dict.values()):
            raise ValueError(
                "Graph must contain at least one LLMOp, DataRetrievalOp or LambdaOp"
            )

        nodes: dict[str, RuntimeOp] = {}
        output_node_map: dict[str, str] = {}
        output_paths: dict[str, str] = {}
        dsl_to_runtime: dict[str, list[str]] = {}

        ctx = LoweringContext(
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
            input_ops_by_id=input_ops_by_id,
            task_type_override=task_type_override,
            outputs_by_source=output_source_to_outputop,
            nodes=nodes,
            dsl_to_runtime=dsl_to_runtime,
            visited=visited,
        )
        node_order = self._lower_ops(ctx, output_node_map, output_paths)

        if task_type_override != "data_profile":
            unvisited_node_ids = set(graph_dict) - visited
            if unvisited_node_ids:
                raise ValueError(
                    f"Graph transformation failed: {len(unvisited_node_ids)} node(s)"
                    " were not visited during traversal, indicating orphaned or"
                    " unreachable ops. Unvisited node IDs:"
                    f" {sorted(unvisited_node_ids)}"
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

    @staticmethod
    def _reject_unsupported_output_enforcement(graph_dict: dict[str, Op]) -> None:
        """Only a text ``LLMChatOp`` validates its output; every other op drops
        the field, so reject it rather than silently ignoring it."""
        for op_id, op in graph_dict.items():
            if (
                isinstance(op, LLMOp)
                and (not isinstance(op, LLMChatOp) or isinstance(op, LLMVisionOp))
                and op.config.output_enforcement is not None
            ):
                raise ValueError(
                    f"{type(op).__name__} '{op_id}' does not support "
                    "config.output_enforcement; use it only on LLMChatOp."
                )

    def _collect_outputs(
        self,
        graph_dict: dict[str, Op],
        visited_node_ids: set[str],
    ) -> dict[str, list[tuple[str, str | None]]]:
        """Map each OutputOp's source to the outputs archiving from it, checking
        that the source lowers and that one source carries one output."""
        outputs_by_source: dict[str, list[tuple[str, str | None]]] = {}
        for op in graph_dict.values():
            if not isinstance(op, OutputOp):
                continue
            assert len(op.inputs) == 1, "OutputOp should have exactly one input"
            source = op.inputs[0]
            if not has_lowering(source):
                raise ValueError(
                    f"OutputOp '{op.name}' input must be an LLMOp, "
                    f"DataRetrievalOp or LambdaOp (got {type(source).__name__})"
                )
            visited_node_ids.add(op.id)
            outputs_by_source.setdefault(source.id, []).append((op.name, op.path))

        for source_id, outputs in outputs_by_source.items():
            if len(outputs) > 1:
                names = ", ".join(name for name, _ in outputs)
                raise ValueError(
                    f"Multiple OutputOps ({names}) target the same source node "
                    f"'{source_id}'. A runtime node produces a single output; "
                    "project multiple outputs from one source by adding an "
                    "intermediate LambdaOp instead."
                )
        return outputs_by_source

    def _lower_ops(
        self,
        ctx: LoweringContext,
        output_node_map: dict[str, str],
        output_paths: dict[str, str],
    ) -> list[str]:
        """Run every registered lowering, one phase at a time, adding each
        result's nodes to the graph in emission order and recording the output
        each op archives at as it lowers."""
        node_order: list[str] = []
        profile = ctx.task_type_override == "data_profile"
        for phase in Phase:
            if profile and phase is not Phase.RETRIEVAL:
                continue
            for op in ctx.graph_dict.values():
                lowering = lookup(op, phase)
                if lowering is None:
                    continue
                result = lowering(self, op, ctx)
                if result is None:
                    continue
                for node in result.nodes:
                    if node.node_id in ctx.nodes:
                        continue
                    ctx.nodes[node.node_id] = node
                    node_order.append(node.node_id)
                ctx.dsl_to_runtime.update(result.built)
                ctx.dsl_to_runtime[op.id] = [
                    node.node_id
                    for node in result.nodes
                    if node.node_id not in result.built
                ]
                ctx.visited.add(op.id)
                self._record_output(op, ctx, output_node_map, output_paths)
            after = phase_hook(phase)
            if after is not None:
                after(self, ctx)
        return node_order

    def _record_output(
        self,
        op: Op,
        ctx: LoweringContext,
        output_node_map: dict[str, str],
        output_paths: dict[str, str],
    ) -> None:
        """Record the output an op archives at, if an OutputOp reads it."""
        outputs = ctx.outputs_by_source.get(op.id)
        runtime_ids = ctx.dsl_to_runtime.get(op.id)
        if outputs is None or not runtime_ids:
            return
        ((output_name, path_override),) = outputs
        node_id = runtime_ids[-1]
        output_node_map[node_id] = output_name
        path = path_override or self._op_output(op).default_output_path
        if path is not None:
            output_paths[node_id] = path
