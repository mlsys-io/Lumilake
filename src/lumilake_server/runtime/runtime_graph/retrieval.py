import re
from dataclasses import dataclass, field
from typing import Any

import sqlparse
from sqlparse.sql import TokenList
from sqlparse.tokens import Keyword

from lumilake import envs
from lumilake_server.ops import DataRetrievalOp, InputOp, LambdaOp, LLMChatOp, LLMOp, Op
from lumilake_server.runtime.runtime_graph.llm import _LLMBuilder
from lumilake_server.runtime.runtime_graph.lowering import (
    LoweringContext,
    LoweringResult,
    Phase,
    register,
)
from lumilake_server.runtime.runtime_ops import RuntimeOp
from lumilake_server.utils.data_profile_offload import (
    _build_sample_data_profile_queries,
    _type_default_sample,
)
from lumilake_server.utils.lumid_data_client import (
    retrieve_sample as lumid_retrieve_sample,
)


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


class _RetrievalBuilder(_LLMBuilder):
    def _resolve_param_upstream(
        self,
        node_id: str,
        graph_dict: dict[str, Op],
        input_ops_by_id: dict[str, InputOp],
    ) -> Op | None:
        """Resolve a param's upstream node, falling back to the graph's
        input-ops map for InputOps referenced only via
        ``data_spec.params[*].node``."""
        upstream = graph_dict.get(node_id)
        if upstream is None:
            upstream = input_ops_by_id.get(node_id)
        return upstream

    @register(DataRetrievalOp, Phase.RETRIEVAL)
    def _lower_data_retrieval(
        self, op: DataRetrievalOp, ctx: LoweringContext
    ) -> LoweringResult | None:
        """Lower one DataRetrievalOp, or to its data-profile node in profile
        mode."""
        # Retrieval nodes can consume InputOp values that are not represented as
        # runtime dependencies. Mark those upstream op ids as visited so strict
        # traversal validation does not flag legitimate input-only branches.
        ctx.visited.update(input_op.id for input_op in op.inputs)
        spec = op.data_spec or {}
        params = spec.get("params")
        if isinstance(params, list):
            for param in params:
                if not isinstance(param, dict):
                    continue
                node = param.get("node")
                if isinstance(node, str) and node in ctx.graph_dict:
                    ctx.visited.add(node)

        if ctx.task_type_override == "data_profile":
            profile = self._data_profile_node(op, ctx)
            return None if profile is None else LoweringResult(nodes=[profile])
        graph_dict = ctx.graph_dict
        inputs_dict = ctx.inputs_dict
        input_ops_by_id = ctx.input_ops_by_id
        op_id = op.id
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
        param_row_counts: dict[str, int] = {}
        for param in params:
            if not isinstance(param, dict):
                resolved_params.append(param)
                continue
            node = param.get("node")
            if isinstance(node, str):
                upstream = self._resolve_param_upstream(
                    node, graph_dict, input_ops_by_id
                )
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
                    if len(values) > 1:
                        param_row_counts[str(param.get("label"))] = len(values)
                    resolved_params.append(literal_param)
                    continue
                if isinstance(upstream, LLMOp):
                    output = self._op_output(upstream)
                    row_count = output.row_count
                    if row_count is not None and row_count > 1:
                        if not output.one_item_per_row:
                            raise ValueError(
                                f"DataRetrievalOp '{op_id}' template param "
                                f"{param.get('label')!r} references '{node}', which"
                                " produces multiple rows but not one output item"
                                " per row; only a plain local LLMChatOp or an"
                                " LLMCompletionOp binds row-wise to a retrieval"
                                " param, so this would silently drop every row"
                                " but the first."
                            )
                        param_row_counts[str(param.get("label"))] = row_count
                if isinstance(upstream, LambdaOp):
                    param = {
                        **param,
                        "path": self._op_output(upstream).read(
                            param.get("path", ""),
                            (
                                f"DataRetrievalOp '{op_id}' template param "
                                f"{param.get('label')!r}"
                            ),
                        ),
                    }
                if node not in seen:
                    seen.add(node)
                    dependencies.append(node)
            resolved_params.append(param)

        if len(set(param_row_counts.values())) > 1:
            raise ValueError(
                f"DataRetrievalOp '{op_id}' binds row-wise params of different"
                f" lengths ({param_row_counts}); every multi-row param must"
                " carry the same number of rows."
            )

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
        if isinstance(spec.get("key"), dict):
            data_spec["key"] = spec["key"]

        node = self._create_runtime_op(
            name=op_id,
            task_type="data_retrieval",
            data_spec=data_spec,
            model_spec={},
            inference_spec={},
            backend="data_retrieval",
            model="data_retrieval",
            dependencies=dependencies if dependencies else None,
        )
        return LoweringResult(nodes=[node])

    def _resolve_profile_param(
        self,
        owner_node_id: str,
        param: Any,
        graph_dict: dict[str, Op],
        inputs_dict: dict[str, list[str]],
        input_ops_by_id: dict[str, InputOp],
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
        upstream = self._resolve_param_upstream(node_id, graph_dict, input_ops_by_id)
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
                input_ops_by_id=input_ops_by_id,
                visited=visited,
            )
            return {
                "label": label,
                "data": {"type": "list", "items": [sample]},
            }
        if isinstance(upstream, LambdaOp):
            sample_value = upstream.sample_value
            if sample_value is None:
                raise ValueError(
                    "Data profile preflight cannot sample upstream LambdaOp "
                    f"'{node_id}' for placeholder '{label}' at node "
                    f"'{owner_node_id}': user code is not executed during "
                    "preflight. Set 'sample_value' on the LambdaOp (or the "
                    "Code node's 'sample-value' note) to supply a "
                    "representative value, or set "
                    "LUMILAKE_DISABLE_DATA_PROFILE=1."
                )
            sample = self._project_sample_value(sample_value, param.get("path"))
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

    def _data_profile_node(
        self, op: DataRetrievalOp, ctx: LoweringContext
    ) -> RuntimeOp | None:
        op_id = op.id
        graph_dict = ctx.graph_dict
        inputs_dict = ctx.inputs_dict
        input_ops_by_id = ctx.input_ops_by_id
        visited: set[str] | None = None
        spec = op.data_spec or {}
        mode = spec["mode"]
        if not op.profilable:
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
                        input_ops_by_id,
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
                        input_ops_by_id,
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
        input_ops_by_id: dict[str, InputOp],
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
            input_ops_by_id=input_ops_by_id,
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
        column = _RetrievalBuilder._column_from_path(path)
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
        input_ops_by_id: dict[str, InputOp],
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
                    input_ops_by_id,
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
                max(1, envs.LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES),
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
        disallowed = _RetrievalBuilder._DISALLOWED_SAMPLE_KEYWORDS

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
        _RetrievalBuilder._assert_sample_query_is_readonly(query)
        if re.search(r"\bFOR\s+(?:UPDATE|SHARE)\b", query, re.IGNORECASE):
            raise ValueError(
                "Data profile preflight cannot append LIMIT to a sample SQL "
                "query that contains a FOR UPDATE / FOR SHARE clause. Add a "
                "'sample_value' to the upstream data_spec to avoid issuing a "
                "live sample query."
            )
        inner = query.rstrip().rstrip(";")
        return f"SELECT * FROM ({inner}) AS _lumilake_sample LIMIT {n}"

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
