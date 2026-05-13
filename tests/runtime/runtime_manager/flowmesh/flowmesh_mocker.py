import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FlowmeshMockNode:
    name: str
    depends_on: tuple[str, ...]
    spec: Mapping[str, Any]


class FlowmeshMockShapeError(RuntimeError):
    pass


class FlowmeshJobMocker:
    """Offline mock executor for FlowMesh jobs.

    The mocker walks graph nodes in topological order, synthesizes node outputs,
    and validates graph-template shape invariants (notably list-length alignment
    for template formatting steps).
    """

    def run(self, task_spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        nodes = self._parse_nodes(task_spec)
        order = self._topological_order(nodes)
        context: dict[str, dict[str, Any]] = {}
        for node_id in order:
            node = nodes[node_id]
            try:
                context[node_id] = self._mock_node_output(node=node, context=context)
            except Exception as exc:
                raise FlowmeshMockShapeError(
                    f"Mock execution failed at node '{node_id}': {exc}"
                ) from exc
        return context

    def _parse_nodes(self, task_spec: Mapping[str, Any]) -> dict[str, FlowmeshMockNode]:
        if "spec" not in task_spec or not isinstance(task_spec["spec"], Mapping):
            raise ValueError("Task spec missing 'spec' mapping")
        spec_payload = task_spec["spec"]
        if "graph" not in spec_payload or not isinstance(
            spec_payload["graph"], Mapping
        ):
            raise ValueError("Task spec missing 'spec.graph' mapping")
        graph_payload = spec_payload["graph"]
        if "nodes" not in graph_payload or not isinstance(graph_payload["nodes"], list):
            raise ValueError("Task spec missing 'spec.graph.nodes' list")

        nodes: dict[str, FlowmeshMockNode] = {}
        for raw_node in graph_payload["nodes"]:
            if not isinstance(raw_node, Mapping):
                raise ValueError("Graph node entry must be a mapping")
            if "name" not in raw_node or not isinstance(raw_node["name"], str):
                raise ValueError("Graph node missing string 'name'")
            node_name = raw_node["name"]
            if node_name in nodes:
                raise ValueError(f"Duplicate graph node name: {node_name}")

            raw_deps = raw_node["dependsOn"] if "dependsOn" in raw_node else []
            if raw_deps is None:
                dep_list: list[str] = []
            elif isinstance(raw_deps, list):
                dep_list = [dep for dep in raw_deps if isinstance(dep, str)]
            else:
                raise ValueError(
                    f"Graph node '{node_name}' has non-list dependsOn:"
                    f" {type(raw_deps).__name__}"
                )

            if "spec" not in raw_node or not isinstance(raw_node["spec"], Mapping):
                raise ValueError(f"Graph node '{node_name}' missing spec mapping")
            node_spec = raw_node["spec"]
            nodes[node_name] = FlowmeshMockNode(
                name=node_name,
                depends_on=tuple(dep_list),
                spec=node_spec,
            )
        return nodes

    def _topological_order(self, nodes: Mapping[str, FlowmeshMockNode]) -> list[str]:
        node_names = list(nodes.keys())
        node_set = set(node_names)
        in_degree: dict[str, int] = {name: 0 for name in node_names}
        children: dict[str, list[str]] = {name: [] for name in node_names}

        for node_id, node in nodes.items():
            for dep in node.depends_on:
                if dep not in node_set:
                    raise ValueError(
                        f"Graph node '{node_id}' depends on unknown node '{dep}'"
                    )
                in_degree[node_id] += 1
                children[dep].append(node_id)

        queue: deque[str] = deque(name for name in node_names if in_degree[name] == 0)
        ordered: list[str] = []
        while queue:
            cur = queue.popleft()
            ordered.append(cur)
            for child in children[cur]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(node_names):
            raise ValueError("Cycle detected in FlowMesh graph")
        return ordered

    def _mock_node_output(
        self,
        *,
        node: FlowmeshMockNode,
        context: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        if "data" not in node.spec or not isinstance(node.spec["data"], Mapping):
            count = self._infer_row_count(node=node, context=context)
            return {
                "items": [{"output": f"{node.name}:mock:{idx}"} for idx in range(count)]
            }

        data_spec = node.spec["data"]
        if "type" in data_spec and data_spec["type"] == "graph_template":
            prompts = self._build_prompts_from_graph_template(
                data_cfg=data_spec,
                context=context,
            )
            return {
                "items": [
                    {"output": self._coerce_to_string(prompt)} for prompt in prompts
                ]
            }

        if "type" in data_spec and data_spec["type"] == "list":
            values = self._materialize_list_data(data_spec=data_spec, context=context)
            return {"items": [self._to_item(value) for value in values]}

        if "type" in data_spec and data_spec["type"] == "value":
            value = data_spec["value"] if "value" in data_spec else None
            return {"items": [self._to_item(value)]}

        if "type" in data_spec and data_spec["type"] == "sql":
            count = self._infer_row_count(node=node, context=context)
            return {
                "items": [
                    {
                        "table": _WildcardMapping(f"{node.name}:row:{idx}"),
                        "output": f"{node.name}:sql:{idx}",
                    }
                    for idx in range(count)
                ]
            }

        if "type" in data_spec and data_spec["type"] == "s3":
            count = self._infer_row_count(node=node, context=context)
            return {
                "items": [
                    {
                        "content": f"{node.name}:content:{idx}",
                        "output": f"{node.name}:s3:{idx}",
                    }
                    for idx in range(count)
                ]
            }

        count = self._infer_row_count(node=node, context=context)
        return {
            "items": [{"output": f"{node.name}:mock:{idx}"} for idx in range(count)]
        }

    def _materialize_list_data(
        self,
        *,
        data_spec: Mapping[str, Any],
        context: Mapping[str, Mapping[str, Any]],
    ) -> list[Any]:
        if "items" in data_spec and isinstance(data_spec["items"], list):
            values: list[Any] = []
            for item in data_spec["items"]:
                materialized = self._materialize_value(value=item, context=context)
                if isinstance(materialized, list):
                    values.extend(materialized)
                else:
                    values.append(materialized)
            return values

        if (
            "node" in data_spec
            and isinstance(data_spec["node"], str)
            and "path" in data_spec
            and isinstance(data_spec["path"], str)
        ):
            resolved = self._evaluate_ref(
                node_id=data_spec["node"],
                path=data_spec["path"],
                context=context,
            )
            if isinstance(resolved, list):
                return list(resolved)
            return [resolved]
        return []

    def _materialize_value(
        self,
        *,
        value: Any,
        context: Mapping[str, Mapping[str, Any]],
    ) -> Any:
        if (
            isinstance(value, Mapping)
            and "node" in value
            and isinstance(value["node"], str)
            and "path" in value
            and isinstance(value["path"], str)
        ):
            return self._evaluate_ref(
                node_id=value["node"],
                path=value["path"],
                context=context,
            )
        return value

    @staticmethod
    def _to_item(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            if any(key in value for key in ("output", "table", "content", "metadata")):
                return dict(value)
            return {"output": json.dumps(value, default=str, ensure_ascii=False)}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return {"output": value}
        return {"output": json.dumps(value, default=str, ensure_ascii=False)}

    def _infer_row_count(
        self,
        *,
        node: FlowmeshMockNode,
        context: Mapping[str, Mapping[str, Any]],
    ) -> int:
        count = 1
        for dep in node.depends_on:
            if dep not in context:
                continue
            dep_result = context[dep]
            if "items" in dep_result and isinstance(dep_result["items"], list):
                count = max(count, len(dep_result["items"]))
        return max(1, count)

    def _build_prompts_from_graph_template(
        self,
        *,
        data_cfg: Mapping[str, Any],
        context: Mapping[str, Mapping[str, Any]],
    ) -> list[str | Sequence[dict[str, str]]]:
        if "template" not in data_cfg or not isinstance(data_cfg["template"], Mapping):
            raise ValueError("graph_template data missing template mapping")
        template_cfg = data_cfg["template"]
        columns_cfg = template_cfg["columns"] if "columns" in template_cfg else []
        columns = self._resolve_columns(columns_cfg=columns_cfg, context=context)
        template_name = template_cfg["name"] if "name" in template_cfg else "format"
        options = template_cfg["options"] if "options" in template_cfg else {}

        if template_name == "format":
            if not isinstance(options, Mapping):
                raise ValueError("format template options must be mapping")
            rendered_messages = self._render_structural_messages(
                columns=columns,
                options=options,
            )
            rendered: list[str | Sequence[dict[str, str]]] = []
            rendered.extend(rendered_messages)
            return rendered
        if "text" in template_cfg and isinstance(template_cfg["text"], str):
            return [self._render_inline_text(template_cfg["text"], columns)]
        return [f"mock:{template_name}"]

    def _resolve_columns(
        self,
        *,
        columns_cfg: Any,
        context: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(columns_cfg, list):
            raise ValueError("graph_template.template.columns must be a list")

        columns: list[dict[str, Any]] = []
        for idx, raw in enumerate(columns_cfg):
            if not isinstance(raw, Mapping):
                raise ValueError("graph_template column must be a mapping")
            label_raw = raw["label"] if "label" in raw else f"Column {idx + 1}"
            if not isinstance(label_raw, str):
                raise ValueError("graph_template column label must be a string")
            label = label_raw

            value: Any = None
            if "expr" in raw and isinstance(raw["expr"], str) and raw["expr"].strip():
                value = self._evaluate_expr(raw["expr"].strip(), context=context)
            elif (
                "node" in raw
                and isinstance(raw["node"], str)
                and "path" in raw
                and isinstance(raw["path"], str)
            ):
                expr = f"{raw['node']}.{raw['path']}"
                value = self._evaluate_expr(expr, context=context)
            elif "data" in raw and isinstance(raw["data"], Mapping):
                data_payload = raw["data"]
                value = self._resolve_column_data(data=data_payload, context=context)
            else:
                raise ValueError(
                    f"Column '{label}' is missing expr/node+path/data definition"
                )
            columns.append({"label": label, "value": value})
        return columns

    def _resolve_column_data(
        self,
        *,
        data: Mapping[str, Any],
        context: Mapping[str, Mapping[str, Any]],
    ) -> Any:
        if "type" not in data or not isinstance(data["type"], str):
            raise ValueError("Column data payload must include string 'type'")
        data_type = data["type"]
        if data_type == "list":
            if "items" in data and isinstance(data["items"], list):
                values: list[Any] = []
                for item in data["items"]:
                    materialized = self._materialize_value(value=item, context=context)
                    values.append(materialized)
                return values
            if (
                "node" in data
                and isinstance(data["node"], str)
                and "path" in data
                and isinstance(data["path"], str)
            ):
                return self._evaluate_ref(
                    node_id=data["node"],
                    path=data["path"],
                    context=context,
                )
            raise ValueError("Column data.type == 'list' missing items or node/path")
        if data_type == "value":
            return data["value"] if "value" in data else None
        if data_type == "dataframe":
            if "columns" not in data or not isinstance(data["columns"], list):
                raise ValueError(
                    "Column data.type == 'dataframe' requires a columns list"
                )
            nested_columns = self._resolve_columns(
                columns_cfg=data["columns"],
                context=context,
            )
            return self._build_grouped_tables(columns=nested_columns)
        if data_type == "graph_template":
            if "template" not in data or not isinstance(data["template"], Mapping):
                raise ValueError(
                    "Column data.type == 'graph_template' requires a template mapping"
                )
            nested_data = {"type": "graph_template", "template": data["template"]}
            return self._build_prompts_from_graph_template(
                data_cfg=nested_data,
                context=context,
            )
        raise ValueError(f"Unsupported column data type: {data_type}")

    def _build_grouped_tables(
        self,
        *,
        columns: Sequence[Mapping[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        if not columns:
            raise ValueError("dataframe data type requires at least one nested column")

        grouped_columns: dict[str, list[list[Any]]] = {}
        for column in columns:
            if "label" not in column or not isinstance(column["label"], str):
                raise ValueError("Nested dataframe column missing string label")
            label = column["label"]
            value = column["value"] if "value" in column else None

            if (
                isinstance(value, list)
                and value
                and all(isinstance(item, list) for item in value)
            ):
                groups = value
            elif isinstance(value, list):
                groups = [value]
            else:
                groups = [[value]]
            grouped_columns[label] = groups

        group_count = max(len(groups) for groups in grouped_columns.values())
        for label, groups in list(grouped_columns.items()):
            if len(groups) == 1 and group_count > 1:
                grouped_columns[label] = groups * group_count
                continue
            if len(groups) != group_count:
                raise ValueError(
                    "dataframe column values must resolve to the same number of groups"
                )

        output_groups: list[list[dict[str, Any]]] = []
        for group_idx in range(group_count):
            max_rows = 1
            raw_values: dict[str, list[Any]] = {}
            for label, groups in grouped_columns.items():
                values = groups[group_idx]
                if not isinstance(values, list):
                    values = [values]
                if values:
                    max_rows = max(max_rows, len(values))
                raw_values[label] = values

            normalized: dict[str, list[Any]] = {}
            for label, values in raw_values.items():
                if len(values) == 1 and max_rows > 1:
                    normalized[label] = [values[0] for _ in range(max_rows)]
                    continue
                if len(values) != max_rows:
                    raise ValueError(
                        "dataframe column values must resolve to the same number of"
                        " rows per group"
                    )
                normalized[label] = values

            rows = [
                {label: values[row_idx] for label, values in normalized.items()}
                for row_idx in range(max_rows)
            ]
            output_groups.append(rows)
        return output_groups

    def _render_structural_messages(
        self,
        *,
        columns: Sequence[Mapping[str, Any]],
        options: Mapping[str, Any],
    ) -> list[list[dict[str, str]]]:
        if "format" not in options or not isinstance(options["format"], Mapping):
            raise ValueError("graph_template format renderer requires options.format")
        format_options = options["format"]

        formatted_prompts: dict[str, list[Any]] = {}
        for column in columns:
            if "label" not in column or not isinstance(column["label"], str):
                raise ValueError("graph_template column is missing label")
            label = column["label"]
            value = column["value"] if "value" in column else None
            formatted_prompts[label] = value if isinstance(value, list) else [value]

        steps = format_options["steps"] if "steps" in format_options else []
        if steps and not isinstance(steps, list):
            raise ValueError("options.format.steps must be a list")
        for step in steps:
            if not isinstance(step, Mapping):
                raise ValueError("format step must be a mapping")
            if "label" not in step or not isinstance(step["label"], str):
                raise ValueError("format step missing string label")
            step_label = step["label"]
            if "template" in step and isinstance(step["template"], str):
                args = step["arguments"] if "arguments" in step else []
                if args and not isinstance(args, list):
                    raise ValueError("format step arguments must be a list")
                format_kwargs: dict[str, str] = {}
                for arg in args if isinstance(args, list) else []:
                    if not isinstance(arg, Mapping):
                        raise ValueError("format step argument must be a mapping")
                    if (
                        "label" not in arg
                        or not isinstance(arg["label"], str)
                        or "value" not in arg
                        or not isinstance(arg["value"], str)
                    ):
                        raise ValueError(
                            "format step argument requires string label/value"
                        )
                    format_kwargs[arg["label"]] = arg["value"]
                formatted_prompts[step_label] = self._render_template(
                    columns=formatted_prompts,
                    template=step["template"],
                    format_kwargs=format_kwargs,
                )
                continue
            if "function" in step and isinstance(step["function"], str):
                args = step["arguments"] if "arguments" in step else []
                formatted_prompts[step_label] = self._render_function_step(
                    columns=formatted_prompts,
                    step_label=step_label,
                    fn_args=args,
                )
                continue
            raise ValueError("Each format step must include a string template")

        messages_cfg = (
            format_options["messages"] if "messages" in format_options else None
        )
        if not isinstance(messages_cfg, list):
            raise ValueError("options.format.messages must be a list")
        return self._aggregate_messages(
            columns=formatted_prompts, messages=messages_cfg
        )

    def _render_template(
        self,
        *,
        columns: Mapping[str, Sequence[Any]],
        template: str,
        format_kwargs: Mapping[str, str],
    ) -> list[str]:
        list_lengths = [len(values) for values in columns.values() if len(values) > 1]
        if list_lengths:
            num_rows = list_lengths[0]
            if any(length != num_rows for length in list_lengths):
                raise AssertionError(
                    "All list-type format arguments must have the same length."
                )
        else:
            num_rows = 1

        materialized: dict[str, list[str]] = {}
        for key, col_id in format_kwargs.items():
            if col_id not in columns:
                raise ValueError(f"Column '{col_id}' not found for template formatting")
            source = columns[col_id]
            values = list(source) if len(source) == num_rows else [source[0]] * num_rows
            materialized[key] = [self._coerce_to_string(value) for value in values]

        prompts: list[str] = []
        for idx in range(num_rows):
            row_args = {key: values[idx] for key, values in materialized.items()}
            prompts.append(template.format(**row_args))
        return prompts

    def _aggregate_messages(
        self,
        *,
        columns: Mapping[str, Sequence[Any]],
        messages: Sequence[Any],
    ) -> list[list[dict[str, str]]]:
        list_lengths = [len(values) for values in columns.values() if len(values) > 1]
        if list_lengths:
            num_rows = list_lengths[0]
            if any(length != num_rows for length in list_lengths):
                raise AssertionError(
                    "All list-type format arguments must have the same length."
                )
        else:
            num_rows = 1
        normalized = {
            label: list(values) if len(values) == num_rows else [values[0]] * num_rows
            for label, values in columns.items()
        }

        output: list[list[dict[str, str]]] = [[] for _ in range(num_rows)]
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError("Each message must be a mapping")
            if "content" not in message or not isinstance(message["content"], str):
                raise ValueError("Each message must include string 'content'")
            content_template = message["content"]

            if content_template in normalized:
                row_contents = [
                    self._coerce_to_string(value)
                    for value in normalized[content_template]
                ]
            else:
                row_contents = []
                for row_idx in range(num_rows):
                    row_mapping = {
                        label: self._coerce_to_string(values[row_idx])
                        for label, values in normalized.items()
                    }
                    row_contents.append(content_template.format(**row_mapping))

            role = message["role"] if "role" in message else "user"
            if not isinstance(role, str):
                raise ValueError("Message role must be a string")
            for row_idx, row_content in enumerate(row_contents):
                output[row_idx].append({"role": role, "content": row_content})
        return output

    def _render_function_step(
        self,
        *,
        columns: Mapping[str, Sequence[Any]],
        step_label: str,
        fn_args: Any,
    ) -> list[str]:
        list_lengths = [len(values) for values in columns.values() if len(values) > 1]
        if list_lengths:
            num_rows = list_lengths[0]
            if any(length != num_rows for length in list_lengths):
                raise AssertionError(
                    "All list-type function arguments must have the same length."
                )
        else:
            num_rows = 1

        if fn_args is None:
            arg_specs: list[Any] = []
        elif isinstance(fn_args, list):
            arg_specs = list(fn_args)
        else:
            raise ValueError("function step arguments must be a list")

        materialized_args: list[list[str]] = []
        for arg in arg_specs:
            if isinstance(arg, str) and arg in columns:
                source = columns[arg]
                values = (
                    list(source) if len(source) == num_rows else [source[0]] * num_rows
                )
                materialized_args.append(
                    [self._coerce_to_string(value) for value in values]
                )
                continue
            materialized_args.append(
                [self._coerce_to_string(arg) for _ in range(num_rows)]
            )

        prompts: list[str] = []
        for row_idx in range(num_rows):
            row_args = [arg_values[row_idx] for arg_values in materialized_args]
            prompts.append(f"<function:{step_label}>({', '.join(row_args)})")
        return prompts

    def _evaluate_ref(
        self,
        *,
        node_id: str,
        path: str,
        context: Mapping[str, Mapping[str, Any]],
    ) -> Any:
        expr = f"{node_id}.{path}"
        return self._evaluate_expr(expr=expr, context=context)

    def _evaluate_expr(
        self,
        expr: str,
        *,
        context: Mapping[str, Mapping[str, Any]],
    ) -> Any:
        parts = expr.split(".")
        if not parts:
            raise ValueError("Empty expression")
        root = parts[0]
        if root not in context:
            raise ValueError(f"Expression root not found in upstream context: {root}")
        value: Any = context[root]

        for token in parts[1:]:
            if not token:
                continue
            attr, indexes = self._split_indexes(token)
            if attr:
                if isinstance(value, Mapping):
                    if attr not in value:
                        raise ValueError(
                            f"Expression attribute '{attr}' not found in '{expr}'"
                        )
                    value = value[attr]
                elif isinstance(value, list):
                    resolved_list: list[Any] = []
                    for item in value:
                        if not isinstance(item, Mapping):
                            raise ValueError(
                                f"Expression attribute '{attr}' expects mapping items"
                                f" in '{expr}'"
                            )
                        if attr not in item:
                            raise ValueError(
                                f"Expression attribute '{attr}' not found in list item"
                                f" for '{expr}'"
                            )
                        resolved_list.append(item[attr])
                    value = resolved_list
                else:
                    raise ValueError(
                        f"Expression attribute '{attr}' cannot be applied to"
                        f" {type(value).__name__} in '{expr}'"
                    )
            for idx in indexes:
                if isinstance(value, list) and -len(value) <= idx < len(value):
                    value = value[idx]
                elif isinstance(value, list) and all(
                    isinstance(item, list) and -len(item) <= idx < len(item)
                    for item in value
                ):
                    value = [item[idx] for item in value]
                else:
                    raise ValueError(
                        f"Expression index [{idx}] out of bounds for '{expr}'"
                    )
        return value

    @staticmethod
    def _split_indexes(token: str) -> tuple[str, list[int]]:
        segments = token.split("[")
        attr = segments[0]
        indexes: list[int] = []
        for segment in segments[1:]:
            raw_idx = segment.rstrip("]").strip()
            if not raw_idx:
                continue
            try:
                indexes.append(int(raw_idx))
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer list index '{raw_idx}' in token '{token}'"
                ) from exc
        return attr, indexes

    @staticmethod
    def _render_inline_text(text: str, columns: Sequence[Mapping[str, Any]]) -> str:
        mapping: dict[str, str] = {}
        for idx, column in enumerate(columns):
            label = column["label"] if "label" in column else f"col{idx}_label"
            value = column["value"] if "value" in column else ""
            mapping[f"col{idx}_label"] = str(label)
            mapping[f"col{idx}_value"] = FlowmeshJobMocker._coerce_to_string(value)
        return text.format_map(_MissingKeySafeDict(mapping))

    @staticmethod
    def _coerce_to_string(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)


class _MissingKeySafeDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class _WildcardMapping(Mapping[str, str]):
    def __init__(self, seed: str) -> None:
        self._seed = seed

    def __getitem__(self, key: str) -> str:
        return f"{self._seed}:{key}"

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str)
