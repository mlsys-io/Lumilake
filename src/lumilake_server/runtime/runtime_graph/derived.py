import re
from dataclasses import replace
from typing import Any

from lumilake_server.common import GenerationConfig
from lumilake_server.data_specs import (
    PYTHON_TASK_TYPE,
    ColumnPlanEntry,
    DataSpecModel,
    ListData,
    PythonData,
)
from lumilake_server.ops import DataOp, InputOp, LambdaOp, Op
from lumilake_server.ops.classification_ops import (
    ClassificationOp,
    literal_pairs,
)
from lumilake_server.ops.http_fetch_ops import HttpFetchOp
from lumilake_server.ops.join_ops import JoinOp
from lumilake_server.runtime import python_step
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
    Phase,
    register,
)
from lumilake_server.runtime.runtime_graph.outputs import (
    OpOutput,
    item_relative_path,
)
from lumilake_server.runtime.runtime_graph.vision import _VisionBuilder
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _replace_query_placeholder(value: Any) -> Any:
    """Adapt the typed op's ``{query}`` template to FlowMesh's prompt slot."""
    if isinstance(value, str):
        return value.replace("{query}", "{{prompt}}")
    if isinstance(value, dict):
        return {key: _replace_query_placeholder(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_query_placeholder(item) for item in value]
    return value


class _DerivedBuilder(_VisionBuilder):
    def _read_source(
        self, consumer: str, op_ref: Op, ctx: LoweringContext
    ) -> list[str] | OpOutput:
        """The literal values, or how an upstream op's values are read."""
        ctx.visited.add(op_ref.id)
        if isinstance(op_ref, InputOp):
            values = ctx.inputs_dict.get(op_ref.name)
            if not values:
                raise ValueError(
                    f"{consumer} {op_ref.name!r} input has no values supplied"
                )
            return values
        if isinstance(op_ref, DataOp):
            if not op_ref.data:
                raise ValueError(f"{consumer} has empty input")
            return op_ref.data
        runtime_ids = ctx.dsl_to_runtime.get(op_ref.id)
        if not runtime_ids and isinstance(op_ref, LambdaOp):
            # A row-mode Lambda a derived op reads runs on its own; the derived
            # phase precedes the row-Lambda one, so build it here.
            steps = self._build_python_step(op_ref, ctx)
            runtime_ids = []
            for node in steps:
                ctx.nodes.setdefault(node.node_id, node)
                runtime_ids.append(node.node_id)
            ctx.dsl_to_runtime[op_ref.id] = runtime_ids
        if isinstance(op_ref, JoinOp):
            # A join's runtime node is its op id; the join phase builds it.
            runtime_ids = [op_ref.id]
        if not runtime_ids:
            raise ValueError(f"{consumer} input {op_ref.id!r} has no runtime node")
        return replace(self._op_output(op_ref), node=runtime_ids[-1])

    @staticmethod
    def _stage_plan(values: list[str] | OpOutput) -> ColumnPlanEntry:
        if not isinstance(values, OpOutput):
            return ColumnPlanEntry(kind="literal", values=values or [])
        path = None
        if not values.needs_flatten and values.path:
            path = re.sub(r"\[(\d+)\]", r".\1", item_relative_path(values.path))
        return ColumnPlanEntry(
            kind="stage",
            node=values.node,
            flatten=True if values.needs_flatten else None,
            path=path,
        )

    def _flatten_step(self, op_id: str, upstream: OpOutput) -> RuntimeOp:
        """A python step that flattens an upstream's grouped output to one value
        per row, so a fetch issues one request per row, not per group."""
        node_id = f"{op_id}__flatten"
        return self._create_runtime_op(
            name=node_id,
            task_type=PYTHON_TASK_TYPE,
            data_spec=PythonData(
                mode="flatten_step",
                # No path: the worker's flatten branch expands both shapes —
                # a local group's ``output`` list and API ``rows``.
                plan=[ColumnPlanEntry(kind="stage", node=upstream.node, flatten=True)],
                timeout_s=python_step.DEFAULT_TIMEOUT_SECONDS,
            ),
            model_spec={},
            inference_spec={},
            backend=python_step.BACKEND,
            model="",
            dependencies=[upstream.node],
        )

    def _fetch_data_spec(
        self, op_id: str, query_op: Op, ctx: LoweringContext
    ) -> tuple[DataSpecModel, list[str], list[RuntimeOp]]:
        """The list data spec a fetch reads: literal queries, one upstream row
        per request, or a flatten step's items when the upstream groups rows."""
        consumer = f"HttpFetchOp '{op_id}'"
        values = self._read_source(consumer, query_op, ctx)
        if isinstance(values, list):
            if not values:
                raise ValueError(f"{consumer} requires at least one query")
            return ListData(items=values), [], []
        if values.needs_flatten:
            # A grouped upstream yields one item per table, so only a python
            # step can expand it to one value per row.
            step = self._flatten_step(op_id, values)
            return (
                ListData(items=f"${{{step.node_id}.value.items}}"),
                [step.node_id],
                [step],
            )
        return (
            ListData(node=values.node, path=values.path),
            [values.node],
            [],
        )

    @register(HttpFetchOp, Phase.DERIVED)
    def _lower_http_fetch(
        self, op: HttpFetchOp, ctx: LoweringContext
    ) -> LoweringResult | None:
        """Lower a fetch to one FlowMesh ``api`` request per query row."""
        op_id = op.id
        data_spec, dependencies, extra_ops = self._fetch_data_spec(op_id, op.query, ctx)
        api_spec: dict[str, Any] = {
            "method": op.method,
            "url": op.url.replace("{query}", "{{prompt}}"),
            "headers": op.config.headers,
            "timeout_sec": op.config.timeout_sec,
            "retries": op.config.retries,
            "concurrency": op.config.max_concurrency,
            "request_delay_sec": op.config.request_delay_sec,
            "response": {
                "parse_json": True,
                "return_body": True,
                "raise_for_status": True,
                "fail_open": op.config.fail_open,
            },
        }
        if op.params is not None:
            api_spec["params"] = _replace_query_placeholder(op.params)
        if op.body is not None:
            api_spec["json"] = _replace_query_placeholder(op.body)
        node = self._create_runtime_op(
            name=op_id,
            task_type="api",
            data_spec=data_spec,
            model_spec={},
            inference_spec={},
            api_spec=api_spec,
            backend="api",
            model="http-fetch",
            dependencies=dependencies or None,
            source_op=op,
        )
        return LoweringResult(nodes=[*extra_ops, node])

    @register(ClassificationOp, Phase.DERIVED)
    def _lower_classification(
        self, op: ClassificationOp, ctx: LoweringContext
    ) -> LoweringResult | None:
        """Lower a classification op to its ``classification`` node, through the
        ``pairs`` step when an input comes from an upstream op."""
        op_id = op.id
        query_values = self._read_source(f"ClassificationOp '{op_id}'", op.query, ctx)
        candidate_values = self._read_source(
            f"ClassificationOp '{op_id}'", op.candidates, ctx
        )
        extra_ops: list[RuntimeOp] = []
        if isinstance(candidate_values, list) and not candidate_values:
            raise ValueError(f"ClassificationOp '{op_id}' has empty candidate list")
        if isinstance(query_values, list) and isinstance(candidate_values, list):
            pairs = literal_pairs(query_values, candidate_values, op_id=op_id)
            data_spec = ListData(
                items=[
                    {"text": pair.query, "text_pair": pair.candidate} for pair in pairs
                ],
            )
            dependencies: list[str] = []
        else:
            pairs_node_id = f"{op_id}__pairs"
            pairs_data_spec = PythonData(
                mode="pairs",
                plan=[
                    self._stage_plan(query_values),
                    self._stage_plan(candidate_values),
                ],
                timeout_s=python_step.DEFAULT_TIMEOUT_SECONDS,
            )
            pairs_dependencies: list[str] = []
            for values in (query_values, candidate_values):
                if (
                    isinstance(values, OpOutput)
                    and values.node not in pairs_dependencies
                ):
                    pairs_dependencies.append(values.node)
            extra_ops.append(
                self._create_runtime_op(
                    name=pairs_node_id,
                    task_type=PYTHON_TASK_TYPE,
                    data_spec=pairs_data_spec,
                    model_spec={},
                    inference_spec={},
                    backend=python_step.BACKEND,
                    model="",
                    dependencies=pairs_dependencies or None,
                )
            )
            data_spec = ListData(items=f"${{{pairs_node_id}.value.items}}")
            dependencies = [pairs_node_id]

        backend = op.config.backend
        backend_config = (
            {"convert": "classify", **op.config.vllm}
            if backend == "vllm"
            else {"device_map": "auto"}
        )
        model_config = GenerationConfig(model=op.config.model, api=None)
        node = self._create_runtime_op(
            name=op_id,
            task_type="classification",
            data_spec=data_spec,
            model_spec=self._build_model_spec(model_config, backend, backend_config),
            inference_spec={},
            backend=backend,
            model=op.config.model,
            dependencies=dependencies or None,
            source_op=op,
        )
        return LoweringResult(nodes=[*extra_ops, node])
