from lumilake_server.data_specs import (
    Column,
    ColumnPlanEntry,
    FormatConfig,
    GraphTemplate,
    GraphTemplateData,
    ListData,
    Message,
    NodeRead,
    PythonData,
    TemplateOptions,
    dump_data_spec,
)
from lumilake_server.ops import DataRetrievalOp, InputOp, LambdaOp, Op
from lumilake_server.ops.llm_ops import ImageGenerationOp, LLMVisionOp
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph.base import (
    _default_output_destination,
)
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
    Phase,
    register,
)
from lumilake_server.runtime.runtime_graph.retrieval import _RetrievalBuilder
from lumilake_server.runtime.runtime_ops import RuntimeOp


class _VisionBuilder(_RetrievalBuilder):
    @staticmethod
    def _flattened_vision_ids(graph_dict: dict[str, Op]) -> set[str]:
        """The per-row VLMs over a retrieval, flattened to one embedding, one
        inference and a regroup step; readers bind to the regrouped output."""
        return {
            op_id
            for op_id, op in graph_dict.items()
            if isinstance(op, LLMVisionOp)
            and op.rowwise_template
            and isinstance(graph_dict.get(op.image_source), DataRetrievalOp)
        }

    @register(LLMVisionOp, Phase.LLM)
    def _lower_vision(
        self, op: LLMVisionOp, ctx: LoweringContext
    ) -> LoweringResult | None:
        """Lower a VLM to its embedding node and inference node, or to the four
        nodes a flattened per-row VLM runs as."""
        ctx.visited.add(op.id)
        ctx.visited.add(op.image_source)
        runtime_ops = self._build_vlm_nodes_from_image_op(
            op.id,
            op,
            ctx.graph_dict,
            ctx.inputs_dict,
            ctx.visited,
            dsl_to_runtime=ctx.dsl_to_runtime,
            runtime_nodes=ctx.nodes,
        )
        return LoweringResult(nodes=runtime_ops)

    def _build_vlm_nodes_from_image_op(
        self,
        llm_op_id: str,
        llm_op: LLMVisionOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        visited_node_ids: set[str],
        dsl_to_runtime: dict[str, list[str]] | None = None,
        runtime_nodes: dict[str, RuntimeOp] | None = None,
    ) -> list[RuntimeOp]:
        if llm_op.rowwise_template:
            upstream_llm_ids, _ = self._infer_structural_messages(
                llm_op_id,
                graph_dict,
                inputs_dict,
                visited_node_ids,
                dsl_to_runtime=dsl_to_runtime,
                runtime_nodes=runtime_nodes,
            )
            columns: list[Column] = []
            for col in llm_op.rowwise_columns or []:
                label = col.get("label")
                data = col.get("data")
                node_ref = col.get("node")
                path = col.get("path")
                if isinstance(label, str) and isinstance(data, dict):
                    columns.append(
                        Column.model_validate({"label": label, "data": data})
                    )
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
                        )
                    )
                    continue
                raise ValueError(
                    f"Column '{label}' of consumer '{llm_op_id}' has neither"
                    " data nor node+path"
                )
            if not columns:
                raise ValueError(f"LLMVisionOp {llm_op_id} has empty rowwise_columns")

            template_spec = GraphTemplate(
                name="format",
                columns=columns,
                options=TemplateOptions(
                    format=FormatConfig(
                        messages=[
                            Message(role="system", content=system_msg)
                            for system_msg in llm_op.system_messages or []
                        ]
                        + [Message(role="user", content=llm_op.rowwise_template)]
                    )
                ),
            )
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
            template_spec.columns = [*template_spec.columns, batch_column]
            template_dependencies = self._collect_graph_template_dependencies(
                dump_data_spec(template_spec)
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
                data_spec=GraphTemplateData(
                    type="graph_template",
                    template=template_spec,
                    image_embedding=NodeRead(
                        node=embedding_node_id, path="embedding_file"
                    ),
                ),
                model_spec=self._build_model_spec(
                    llm_op.config, "vllm", backend_config
                ),
                inference_spec=inference_spec,
                backend="vllm",
                model=llm_op.config.model,
                dependencies=deduped_deps,
                source_op=llm_op,
            )
            return [embedding_node, vlm_node]

        # Flattened per-row VLM: one embedding + one inference over the flat
        # prompts, plus a flatten python step (renders one prompt per image)
        # and a regroup python step (back to one item per symbol). The image
        # source is the group boundary: it defines the per-symbol image counts,
        # and every other grouped column must match them.
        image_path = embedding_data_spec.path
        if image_path is None:
            raise ValueError(f"Per-row VLM '{llm_op_id}' image source has no path")
        if not image_path.startswith("items."):
            raise ValueError(
                f"Per-row VLM '{llm_op_id}' image path '{image_path}' must"
                " start with 'items.'"
            )
        image_entry = ColumnPlanEntry(
            kind="stage",
            node=embedding_data_spec.node,
            path=image_path[len("items.") :],
            label=batch_column.label,
            grouped=True,
        )
        plan: list[ColumnPlanEntry] = [image_entry]
        for c in columns:
            if isinstance(c.data, ListData):
                if not isinstance(c.data.items, list):
                    raise ValueError(
                        f"Per-row VLM '{llm_op_id}' column '{c.label}' carries no"
                        " literal values"
                    )
                plan.append(
                    ColumnPlanEntry(
                        kind="literal",
                        values=c.data.items,
                        label=c.label,
                        grouped=False,
                    )
                )
                continue
            if c.node is None:
                raise ValueError(
                    f"Per-row VLM '{llm_op_id}' column '{c.label}' reads no node"
                )
            up = graph_dict.get(c.node)
            if up is None:
                raise ValueError(
                    f"Per-row VLM '{llm_op_id}' column '{c.label}' reads node"
                    f" '{c.node}', which is not reachable in the graph."
                )
            if isinstance(up, LambdaOp) and up.takes_whole_columns:
                raise ValueError(
                    f"Column '{c.label}' of consumer '{llm_op_id}' references"
                    f" list-mode Lambda '{c.node}'; a flattened per-row VLM"
                    " cannot read a whole-column Lambda."
                )
            if c.path is None or not c.path.startswith("items."):
                raise ValueError(
                    f"Column '{c.label}' of consumer '{llm_op_id}' reads"
                    f" '{c.path}' from '{c.node}'; only 'items.*' paths"
                    " are supported in a flattened per-row VLM."
                )
            plan.append(
                ColumnPlanEntry(
                    kind="stage",
                    node=c.node,
                    path=c.path[len("items.") :],
                    label=c.label,
                    grouped=isinstance(up, DataRetrievalOp),
                )
            )

        flatten_id = f"{llm_op_id}_flatten"
        flatten_deps = list(
            dict.fromkeys(
                e.node for e in plan if e.kind == "stage" and e.node is not None
            )
        )
        flatten = self._create_runtime_op(
            name=flatten_id,
            task_type=python_step.TASK_TYPE,
            data_spec=PythonData(
                mode="flatten",
                plan=plan,
                template=llm_op.rowwise_template,
                timeout_s=python_step.DEFAULT_TIMEOUT_SECONDS,
            ),
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=flatten_deps,
        )

        infer_id = f"{llm_op_id}_infer"
        infer_messages = [
            *template_spec.options.format.messages,
            Message(role="user", content="{prompt}"),
        ]
        infer_template = GraphTemplate(
            name="format",
            columns=[
                Column(label="prompt", node=flatten_id, path="value.items.output"),
                Column(
                    label=batch_column.label,
                    node=flatten_id,
                    path="value.items.output",
                ),
            ],
            options=TemplateOptions(format=FormatConfig(messages=infer_messages)),
        )
        infer = self._create_runtime_op(
            name=infer_id,
            task_type="inference",
            data_spec=GraphTemplateData(
                type="graph_template",
                template=infer_template,
                image_embedding=NodeRead(node=embedding_node_id, path="embedding_file"),
            ),
            model_spec=self._build_model_spec(llm_op.config, "vllm", backend_config),
            inference_spec=inference_spec,
            backend="vllm",
            model=llm_op.config.model,
            dependencies=[flatten_id, embedding_node_id],
            source_op=llm_op,
        )

        regroup_deps = list(
            dict.fromkeys(
                dep
                for dep in (infer_id, flatten_id, image_entry.node)
                if dep is not None
            )
        )
        regroup = self._create_runtime_op(
            name=llm_op_id,
            task_type=python_step.TASK_TYPE,
            data_spec=PythonData(
                mode="regroup",
                plan=[
                    ColumnPlanEntry(kind="stage", node=infer_id, label="outputs"),
                    ColumnPlanEntry(
                        kind="stage", node=flatten_id, path="row", label="rows"
                    ),
                    image_entry.model_copy(update={"label": "groups"}),
                ],
                timeout_s=python_step.DEFAULT_TIMEOUT_SECONDS,
            ),
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=regroup_deps,
        )

        return [flatten, embedding_node, infer, regroup]

    def _resolve_vlm_image_source(
        self,
        llm_op: LLMVisionOp,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
    ) -> tuple[ListData, Column, list[str]]:
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
            embedding_data_spec = ListData(type="list", items=image_items)
            batch_column = Column(
                label=f"{image_source_op.id}_batch",
                data=ListData(items=["" for _ in range(len(image_items))]),
            )
            return embedding_data_spec, batch_column, []

        if isinstance(image_source_op, ImageGenerationOp):
            resolved_path = (
                "items.image" if llm_op.image_path == "images" else llm_op.image_path
            )
            embedding_data_spec = ListData(
                type="list", node=image_source_id, path=resolved_path
            )
            batch_column = Column(
                label=f"{image_source_id}_batch",
                node=image_source_id,
                path=resolved_path,
            )
            return embedding_data_spec, batch_column, [image_source_id]

        if isinstance(image_source_op, DataRetrievalOp):
            image_path = (
                llm_op.image_path
                if llm_op.image_path != "images"
                else self._op_output(image_source_op).path
            )
            embedding_data_spec = ListData(
                type="list", node=image_source_id, path=image_path
            )
            batch_column = Column(
                label=f"{image_source_id}_batch",
                node=image_source_id,
                path=image_path,
            )
            return embedding_data_spec, batch_column, [image_source_id]

        raise ValueError(
            "LLMVisionOp "
            f"{llm_op.id} image source must be an InputOp, ImageGenerationOp, "
            "or DataRetrievalOp "
            f"(got {type(image_source_op).__name__})."
        )
