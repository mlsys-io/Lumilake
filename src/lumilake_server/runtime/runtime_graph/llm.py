import json
import re
from collections.abc import Sequence
from typing import Any, cast

from lumilake import envs
from lumilake_server.common import ApiConfig
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
    RouterOp,
    RowShape,
)
from lumilake_server.ops.embedding_ops import EmbeddingOp
from lumilake_server.ops.llm_ops import (
    ImageGenerationOp,
    LLMChatOp,
    LLMCompletionOp,
)
from lumilake_server.ops.router_ops import RowGate
from lumilake_server.ops.util_ops import RowLambdaFn, SubmittedFunction
from lumilake_server.runtime import python_step
from lumilake_server.runtime.flowmesh_client import (
    is_api_origin_trusted,
    resolve_api_credential,
)
from lumilake_server.runtime.runtime_graph.base import (
    _default_output_destination,
)
from lumilake_server.runtime.runtime_graph.embedding import _EmbeddingBuilder
from lumilake_server.runtime.runtime_graph.graph import (
    _API_CREDENTIAL_PLACEHOLDER,
    Roles,
)
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
    Phase,
    register,
)
from lumilake_server.runtime.runtime_graph.outputs import (
    is_api_task,
    is_row_gated,
    output_row_count,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils import lambda_runtime


def _declared_node_refs(op: LLMChatOp) -> set[str]:
    """Every node id an ``LLMChatOp`` names outside its inputs."""
    refs: set[str] = set()
    for column in (*(op.rowwise_columns or []), *(op.aggregate_table or [])):
        node_ref = column.get("node")
        if isinstance(node_ref, str):
            refs.add(node_ref)
    if op.rowwise_messages_column is not None:
        refs.add(op.rowwise_messages_column.node)
    if op.condition is not None:
        refs.add(op.condition["node"])
    return refs


class _LLMBuilder(_EmbeddingBuilder):

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
        source_row_count = output_row_count(source, inputs_dict, graph_dict)
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
                    kind="content",
                )
                content_path = self._op_output(content_op).path
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
        output_enforcement = llm_op.config.output_enforcement_spec()

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
                    or self._op_output(prompt_op).one_item_per_row
                ):
                    raise ValueError(
                        f"LLMCompletionOp '{llm_op_id}' prompt upstream"
                        f" '{prompt_op.id}' ({type(prompt_op).__name__}) does"
                        " not produce one prompt string per row"
                    )
                prompt_path = self._op_output(prompt_op).path
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
                output_enforcement=output_enforcement,
                backend=backend,
                model=llm_op.config.model,
                dependencies=row_dependencies if row_dependencies else None,
                output_spec=output_spec,
                condition=None,
                skip_when_all_inputs_empty=bool(row_dependencies),
            )

        if isinstance(llm_op, LLMChatOp) and llm_op.row_shape is RowShape.GROUPED:
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
                output_enforcement=output_enforcement,
                backend=backend,
                model=llm_op.config.model,
                dependencies=row_dependencies if row_dependencies else None,
                output_spec=output_spec,
                condition=self._row_condition(
                    llm_op.condition,
                    output_row_count(llm_op, inputs_dict, graph_dict) or 1,
                    graph_dict,
                    inputs_dict,
                    dsl_to_runtime,
                ),
                skip_when_all_inputs_empty=True,
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
                output_enforcement=output_enforcement,
                backend=backend,
                model=llm_op.config.model,
                dependencies=aggregate_dependencies if aggregate_dependencies else None,
                output_spec=output_spec,
                condition=self._row_condition(
                    llm_op.condition,
                    output_row_count(llm_op, inputs_dict, graph_dict) or 1,
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
            output_enforcement=output_enforcement,
            backend=backend,
            model=llm_op.config.model,
            dependencies=default_dependencies,
            output_spec=output_spec,
            condition=self._row_condition(
                llm_op.condition if isinstance(llm_op, LLMChatOp) else None,
                output_row_count(llm_op, inputs_dict, graph_dict) or 1,
                graph_dict,
                inputs_dict,
                dsl_to_runtime,
            ),
        )

    def _row_gate(
        self,
        *,
        consumer: str,
        row_gate: RowGate | None,
        graph_dict: dict[str, Op],
    ) -> RowGate | None:
        """Validate a per-row branch gate against the router it names."""
        if row_gate is None:
            return None
        router = graph_dict.get(row_gate.node)
        if not isinstance(router, RouterOp):
            raise ValueError(
                f"LLMChatOp '{consumer}' row_gate node '{row_gate.node}' must be a"
                " RouterOp"
            )
        if row_gate.branch not in router.branches:
            raise ValueError(
                f"LLMChatOp '{consumer}' row_gate branch {row_gate.branch!r} is not"
                f" declared by RouterOp '{row_gate.node}'"
            )
        return row_gate

    @staticmethod
    def _check_row_gate_consumers(graph_dict: dict[str, Op]) -> None:
        """Reject every reader of a row-gated LLM other than an ``OutputOp`` or
        a render step.

        A row gate compiles its LLM into status, selection, gated call and
        merge stages, so the op has no single runtime node a consumer could
        bind to. A fused round reads each leaf through a render step, and a
        list-mode Lambda reads the merge stage's item list, which holds one
        item per input row. References declared on a column
        (``rowwise_columns``, ``aggregate_table``), on a messages column or in
        a ``condition`` read the op without listing it as an input, so they are
        checked as well.
        """
        for op_id, op in graph_dict.items():
            if not isinstance(op, LLMChatOp) or op.row_gate is None:
                continue
            for consumer_id, consumer in graph_dict.items():
                if consumer_id == op_id:
                    continue
                reads_op = any(inp is op for inp in consumer.inputs) or (
                    isinstance(consumer, LLMChatOp)
                    and op_id in _declared_node_refs(consumer)
                )
                if not reads_op:
                    continue
                if isinstance(consumer, OutputOp):
                    continue
                if (
                    isinstance(consumer, FormatOp)
                    and consumer.template == "{ref0}"
                    and len(consumer.inputs) == 1
                ):
                    continue
                if isinstance(consumer, LambdaOp) and consumer.takes_whole_columns:
                    # A list-mode Lambda reads the merge stage, one item per
                    # input row, so it sees the routed answer or ``None``.
                    continue
                raise ValueError(
                    f"Op '{consumer_id}' reads the row-gated LLMChatOp '{op_id}',"
                    " which compiles to status, selection, gated call and merge"
                    " stages; a row gate keeps one item per input row, so read it"
                    " through an OutputOp, a list-mode Lambda, or a fused round's"
                    " render step."
                )

    def _row_gate_item_path(
        self,
        *,
        llm_op_id: str,
        label: str,
        path: str,
    ) -> str:
        """Map a gated column's declared read to the python step's item path.

        The selector reads the upstream's item list first, so a declared
        ``items.output.<field>`` read walks ``output.<field>`` inside each
        item. The walker decodes a JSON string before stepping into it, so a
        row-mode Lambda's textified ``output`` reads the same value as its raw
        ``value``.
        """
        if not path.startswith("items.output."):
            raise ValueError(
                f"LLMChatOp '{llm_op_id}' row gate column {label!r} reads"
                f" {path!r}; only 'items.output' (the whole output) or a"
                " field under it is supported."
            )
        return path[len("items.") :]

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
        skip_when_all_inputs_empty: bool = False,
    ) -> list[RuntimeOp]:
        """Build the single FlowMesh ``api`` task for an externally-hosted LLM:
        the data_spec matches the local path, and the executor substitutes each
        row's rendered messages into the ``{{prompt}}`` body placeholder."""
        model = llm_op.config.resolved_model()
        url = api_config.url or envs.LUMILAKE_API_DEFAULT_URL
        if llm_op.config.num_beams is not None:
            raise ValueError(
                f"LLMChatOp '{llm_op_id}' num_beams is available only for local "
                "vLLM generation, not API mode."
            )
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
        api_output_enforcement = llm_op.config.output_enforcement_spec()
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
                output_enforcement=api_output_enforcement,
                backend="api",
                model=model,
                dependencies=dependencies,
                output_spec=output_spec,
                condition=self._row_condition(
                    condition,
                    output_row_count(llm_op, inputs_dict, graph_dict) or 1,
                    graph_dict,
                    inputs_dict,
                    dsl_to_runtime,
                ),
                skip_when_all_inputs_empty=skip_when_all_inputs_empty,
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
        """Build the dataframe data_spec for a rowwise LLMChatOp with a
        ``rowwise_template``, shared by the local and API paths."""
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
                upstream = self._graph_node(node_ref)
                if path is None and not isinstance(upstream, InputOp):
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
                        upstream=upstream,
                        inputs_dict=inputs_dict,
                    )
                )
                # An InputOp ref is bound as a literal column (no runtime node),
                # so it must not become a dependency on a node that does not
                # exist.
                if node_ref not in dependencies and not isinstance(upstream, InputOp):
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
        assert (
            llm_op.rowwise_template is not None
        ), "a rowwise op without a messages column declares rowwise_template"
        messages.append({"role": "user", "content": llm_op.rowwise_template})
        return {
            "type": "dataframe",
            "columns": columns,
            "messages": messages,
        }, dependencies

    def _lower_gated_llm(self, op: LLMChatOp, ctx: LoweringContext) -> LoweringResult:
        """Validate a row gate and lower its chat op to its four stages."""
        row_gate = self._row_gate(
            consumer=op.id, row_gate=op.row_gate, graph_dict=ctx.graph_dict
        )
        assert row_gate is not None, "the caller checked the gate is present"
        if op.rowwise_messages_column is not None:
            raise ValueError(
                f"LLMChatOp '{op.id}' combines row_gate with"
                " rowwise_messages_column; the gated path builds its prompts"
                " from rowwise_template, so a messages column would be ignored."
                " Use rowwise_template, or drop the row gate."
            )
        if op.row_shape is not RowShape.GROUPED:
            raise ValueError(f"LLMChatOp '{op.id}' row_gate requires rowwise_template")
        if op.condition is not None:
            raise ValueError(
                f"LLMChatOp '{op.id}' combines row_gate with condition; the"
                " gated call runs on the rows the selector kept, so a condition"
                " would be ignored. Route on the condition's field instead, or"
                " drop the row gate."
            )
        return LoweringResult(
            nodes=self._build_row_gated_llm_ops(llm_op=op, row_gate=row_gate, ctx=ctx)
        )

    @register(RouterOp, Phase.LLM)
    def _lower_router(
        self, op: RouterOp, ctx: LoweringContext
    ) -> LoweringResult | None:
        """A router emits no node of its own; its gate lowers it into the status
        and selection steps that read its rules."""
        ctx.visited.add(op.id)
        return None

    def _build_row_gated_llm_ops(
        self,
        *,
        llm_op: LLMChatOp,
        row_gate: RowGate,
        ctx: LoweringContext,
    ) -> list[RuntimeOp]:
        """Build status, selection, gated LLM, and merge stages for a row gate.

        Every upstream read goes through ``OpOutput``; a gated op keeps only the
        gate's own rules: its source must compile to one node, and no API item
        can fill a status or a column, because an API item's text sits under
        ``json.choices``.
        """
        llm_op_id = llm_op.id
        graph_dict = ctx.graph_dict
        router_id = row_gate.node
        router = graph_dict[router_id]
        assert isinstance(router, RouterOp), "the gate was validated by the caller"
        source_op = router.source
        if isinstance(source_op, InputOp):
            values = ctx.inputs_dict.get(source_op.name)
            if values is None:
                raise ValueError(
                    f"RouterOp '{router_id}' source input '{source_op.name}' "
                    "has no values supplied"
                )
            source_columns: python_step.ColumnPlan = [
                {"kind": "literal", "label": "source", "values": values}
            ]
            status_dependencies: tuple[str, ...] = ()
        else:
            self._lower_for_read(source_op, ctx)
            output = self._op_output(source_op)
            if source_op.id not in ctx.dsl_to_runtime:
                raise ValueError(
                    f"RouterOp '{router_id}' source '{source_op.id}' must compile "
                    "to one runtime node"
                )
            if is_api_task(source_op) and not output.needs_flatten:
                raise ValueError(
                    f"RouterOp '{router_id}' reads API upstream"
                    f" '{source_op.id}'; an API item carries its text under"
                    " 'json.choices[0].message.content', so only a local op can"
                    " source a router."
                )
            plan_path = output.step_read(router.input_path, f"RouterOp '{router_id}'")
            source_columns = [
                {
                    "label": "source",
                    **self._python_plan_entry(
                        output.node, path=plan_path, flatten=output.needs_flatten
                    ),
                }
            ]
            status_dependencies = (output.node,)
        status_id = f"{llm_op_id}__row_status"
        select_id = f"{llm_op_id}__row_select"
        selected_id = f"{llm_op_id}__selected"
        merge_id = f"{llm_op_id}__rows"
        status = self._python_step_op(
            status_id,
            "row_status",
            source_columns,
            status_dependencies,
            rules=[rule.to_dict() for rule in router.rules],
            default_branch=router.default_branch,
        )
        # Mark the message chain as visited even though the gated model reads
        # its rows from the filtered selector below.
        self._infer_structural_messages(
            llm_op_id,
            graph_dict,
            ctx.inputs_dict,
            ctx.visited,
            dsl_to_runtime=ctx.dsl_to_runtime,
            api_mode=llm_op.config.api is not None,
        )
        labels: list[str] = []
        # The selector and the merge both read the status step's raw row, not
        # its JSON text.
        select_plan: python_step.ColumnPlan = [
            {"kind": "stage", "node": status_id, "path": "value"}
        ]
        select_dependencies: list[str] = [status_id]
        for column in llm_op.rowwise_columns or []:
            label = column.get("label")
            if not isinstance(label, str) or not label:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' row gate requires labeled columns"
                )
            if label in labels:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' row gate has duplicate label {label!r}"
                )
            labels.append(label)
            data = column.get("data")
            node_ref = column.get("node")
            if isinstance(data, dict):
                items = data.get("items")
                if not isinstance(items, list):
                    raise ValueError(
                        f"LLMChatOp '{llm_op_id}' row gate column {label!r} "
                        "requires a list data payload"
                    )
                select_plan.append({"kind": "literal", "values": items})
                continue
            if not isinstance(node_ref, str):
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' row gate column {label!r} "
                    "requires data or node"
                )
            upstream = self._graph_node(node_ref)
            if isinstance(upstream, InputOp):
                values = ctx.inputs_dict.get(upstream.name)
                if values is None:
                    raise ValueError(
                        f"LLMChatOp '{llm_op_id}' row gate column {label!r} "
                        f"references input {upstream.name!r} with no values"
                    )
                select_plan.append({"kind": "literal", "values": values})
                continue
            if upstream is None:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' row gate column {label!r} "
                    f"references unknown node {node_ref!r}"
                )
            self._lower_for_read(upstream, ctx)
            output = self._op_output(upstream)
            if is_api_task(upstream) and not output.needs_flatten:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' row gate column {label!r} reads API"
                    f" upstream '{node_ref}'; an API item carries its text under"
                    " 'json.choices[0].message.content', so only a local op can"
                    " fill a gated column."
                )
            column_path = column.get("path") or ""
            if not isinstance(column_path, str):
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' row gate column {label!r} path "
                    "must be a string"
                )
            select_plan.append(
                self._python_plan_entry(
                    output.node,
                    path=output.step_read(
                        column_path,
                        f"LLMChatOp '{llm_op_id}' row gate column {label!r}",
                    ),
                    flatten=output.needs_flatten,
                )
            )
            if output.node not in select_dependencies:
                select_dependencies.append(output.node)
        selector = self._python_step_op(
            select_id,
            "row_select",
            select_plan,
            tuple(select_dependencies),
            labels=labels,
            branch=row_gate.branch,
        )
        # The selector emits one flat ``{label: value}`` row per kept row, so
        # a gated column reads its label straight off each selected item.
        base_data_spec: dict[str, Any] = {
            "type": "dataframe",
            "columns": [
                {
                    "label": label,
                    "node": select_id,
                    "path": f"value.items.{label}",
                }
                for label in labels
            ],
            "messages": [
                *[
                    {"role": "system", "content": message}
                    for message in llm_op.system_messages or []
                ],
                {"role": "user", "content": llm_op.rowwise_template},
            ],
        }
        api_config = llm_op.config.api
        if api_config is not None:
            selected = self._build_api_task(
                llm_op_id=selected_id,
                llm_op=llm_op,
                api_config=api_config,
                data_spec=base_data_spec,
                dependencies=[select_id],
                output_spec=None,
                condition=None,
                graph_dict=graph_dict,
                inputs_dict=ctx.inputs_dict,
                dsl_to_runtime=ctx.dsl_to_runtime,
            )[0]
        else:
            inference_spec = llm_op.config.inference_spec()
            if llm_op.structural_outputs:
                inference_spec["templates"] = llm_op.structural_outputs
            selected = RuntimeOp(
                node_id=selected_id,
                task_type=ctx.task_type_override or "inference",
                backend="vllm",
                model=llm_op.config.model,
                data_spec=base_data_spec,
                model_spec=self._build_model_spec(llm_op.config, "vllm"),
                inference_spec=inference_spec,
                dependencies=(select_id,),
                condition=None,
            )
        merge = self._python_step_op(
            merge_id,
            "row_skip_merge",
            [
                {"kind": "stage", "node": status_id, "path": "value"},
                {
                    "kind": "stage",
                    "node": selected_id,
                    "flatten": True,
                    "allow_empty": True,
                },
            ],
            (status_id, selector.node_id, selected_id),
            branch=row_gate.branch,
        )
        ctx.visited.update({router_id, llm_op_id})
        return [status, selector, selected, merge]

    def _build_rowwise_messages_op(
        self,
        *,
        llm_op: LLMChatOp,
        ctx: LoweringContext,
    ) -> RuntimeOp:
        """Build one LLM op whose input is a list of complete message arrays.

        The declared read is ``items.output``-rooted like any other column, so
        it maps through ``OpOutput.read`` to the envelope the upstream emits.
        """
        llm_op_id = llm_op.id
        graph_dict = ctx.graph_dict
        inputs_dict = ctx.inputs_dict
        dsl_to_runtime = ctx.dsl_to_runtime
        self._infer_structural_messages(
            llm_op_id,
            graph_dict,
            inputs_dict,
            ctx.visited,
            dsl_to_runtime=dsl_to_runtime,
            api_mode=llm_op.config.api is not None,
        )
        column = llm_op.rowwise_messages_column
        if column is None:
            raise ValueError(f"LLMChatOp '{llm_op_id}' has no message column")
        node_ref = column.node
        upstream = self._graph_node(node_ref)
        if isinstance(upstream, InputOp):
            ctx.visited.add(node_ref)
            values = inputs_dict.get(upstream.name)
            if values is None:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' rowwise_messages_column references "
                    f"input {upstream.name!r} with no values"
                )
            messages = []
            for value in values:
                try:
                    decoded = json.loads(value)
                except ValueError as exc:
                    raise ValueError(
                        f"LLMChatOp '{llm_op_id}' rowwise_messages_column input "
                        f"{upstream.name!r} values must be JSON message arrays"
                    ) from exc
                messages.append(decoded)
            data_spec: dict[str, Any] = {"type": "list", "items": messages}
            dependencies: list[str] = []
        else:
            if upstream is None:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' rowwise_messages_column references "
                    f"unknown node {node_ref!r}"
                )
            if is_row_gated(upstream):
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' rowwise_messages_column reads row-gated"
                    f" '{node_ref}', which lowers to status, selection, call and"
                    " merge stages and emits no message arrays; point the column at"
                    " the node that builds them."
                )
            self._lower_for_read(upstream, ctx)
            output = self._op_output(upstream)
            if output.needs_flatten:
                raise ValueError(
                    f"LLMChatOp '{llm_op_id}' rowwise_messages_column reads grouped"
                    f" upstream '{node_ref}'; a messages column needs one serialized"
                    " array per row."
                )
            ctx.visited.add(node_ref)
            data_spec = {
                "type": "list",
                "node": output.node,
                "path": output.read(
                    column.path,
                    f"LLMChatOp '{llm_op_id}' rowwise_messages_column",
                ),
            }
            dependencies = [output.node]
        if llm_op.config.api is None:
            return self._create_runtime_op(
                name=llm_op_id,
                task_type="inference",
                data_spec=data_spec,
                model_spec=self._build_model_spec(llm_op.config, "vllm"),
                inference_spec=llm_op.config.inference_spec(),
                backend="vllm",
                model=llm_op.config.model,
                dependencies=dependencies or None,
            )
        api_config = llm_op.config.api
        return self._build_api_task(
            llm_op_id=llm_op_id,
            llm_op=llm_op,
            api_config=api_config,
            data_spec=data_spec,
            dependencies=dependencies or None,
            output_spec=None,
            condition=None,
            graph_dict=graph_dict,
            inputs_dict=inputs_dict,
            dsl_to_runtime=dsl_to_runtime,
        )[0]

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
                    upstream=self._graph_node(node_ref),
                    inputs_dict=inputs_dict,
                )
            )
            # An InputOp ref is bound as a literal column (no runtime node), so
            # it must not become a dependency on a node that does not exist.
            if node_ref not in dependencies and not isinstance(
                self._graph_node(node_ref), InputOp
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
            skip_when_all_inputs_empty=True,
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

    @register(LLMOp, Phase.LLM)
    def _lower_llm(self, op: LLMOp, ctx: LoweringContext) -> LoweringResult | None:
        """Lower an LLM op: a row-gated chat op to its four stages, a messages
        column to a list-fed task, and any other op through the API task when it
        targets an external endpoint and the local inference task otherwise."""
        if isinstance(op, LLMChatOp) and op.row_gate is not None:
            return self._lower_gated_llm(op, ctx)
        if isinstance(op, LLMChatOp) and op.rowwise_messages_column is not None:
            return LoweringResult(
                nodes=[self._build_rowwise_messages_op(llm_op=op, ctx=ctx)]
            )
        if op.config.api is not None:
            upstream_llm_ids, template_spec = self._infer_structural_messages(
                op.id,
                ctx.graph_dict,
                ctx.inputs_dict,
                ctx.visited,
                dsl_to_runtime=ctx.dsl_to_runtime,
                runtime_nodes=ctx.nodes,
                api_mode=True,
            )
            runtime_ops = self._build_api_llm_op(
                llm_op_id=op.id,
                llm_op=op,
                api_config=op.config.api,
                template_spec=template_spec,
                upstream_llm_ids=upstream_llm_ids,
                output_spec=None,
                condition=op.condition if isinstance(op, LLMChatOp) else None,
                graph_dict=ctx.graph_dict,
                inputs_dict=ctx.inputs_dict,
                dsl_to_runtime=ctx.dsl_to_runtime,
            )
        else:
            runtime_ops = [
                self._build_node_from_llm_op(
                    op.id,
                    op,
                    ctx.graph_dict,
                    ctx.inputs_dict,
                    ctx.visited,
                    task_type_override=ctx.task_type_override,
                    dsl_to_runtime=ctx.dsl_to_runtime,
                    runtime_nodes=ctx.nodes,
                )
            ]
        return LoweringResult(nodes=runtime_ops)

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
                is_api_ancestor = is_api_task(op)
                output = self._op_output(op)
                if op.id not in upstream_llm_ids:
                    upstream_llm_ids.add(op.id)
                    if not is_api_ancestor:
                        row_count = output.row_count
                        if row_count is not None and row_count > 1:
                            if api_mode or not output.one_item_per_row:
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
                        "path": output.path,
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
                    columns[op.id] = {"node": op.id, "path": self._op_output(op).path}
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
