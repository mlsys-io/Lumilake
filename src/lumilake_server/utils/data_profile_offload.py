import random
import re
from collections.abc import Mapping, Sequence
from string import Formatter
from typing import TYPE_CHECKING, Any

from lumilake import envs
from pydantic import BaseModel, ConfigDict, Field

from lumilake_server.data_profile_models import (
    DataProfileCostEstimate,
    DataProfileResultRow,
    DataProfileResultsPayload,
)
from lumilake_server.graphs import CompiledGraph

if TYPE_CHECKING:
    from lumilake_server.runtime.request import WorkflowSliceMeta
    from lumilake_server.runtime.runtime_graph import RuntimeGraph

RuntimeGraphBuilder = None

_SQL_TABLE_PATTERN = re.compile(
    r"(?:from|join)\s+([^\s,;]+)",
    re.IGNORECASE,
)


class DataProfileTaskNode(BaseModel):
    node_id: str
    raw_node_id: str
    data_spec: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class DataProfileTaskPayload(BaseModel):
    task_key: str
    request_id: str
    public_graph_name: str
    template_hash: str
    node_order: list[str]
    nodes: dict[str, DataProfileTaskNode]

    model_config = ConfigDict(extra="forbid")


class DataProfileTaskSpec(BaseModel):
    task_key: str
    payload: DataProfileTaskPayload

    model_config = ConfigDict(extra="forbid")


class DataProfileTaskResult(BaseModel):
    data_profile_results: dict[str, list[DataProfileResultRow]] = Field(
        default_factory=dict
    )

    model_config = ConfigDict(extra="forbid")


# Process-lifetime cache: jobs.py writes, collect_data_profile pops by task_key.
data_profile_registry: dict[str, dict[str, Any]] = {}


def build_request_data_profile_tasks(
    *,
    request_id: str,
    graphs: dict[str, CompiledGraph],
    workflow_slices: dict[str, "WorkflowSliceMeta"],
) -> list[DataProfileTaskSpec]:
    grouped: dict[str, list[tuple[str, WorkflowSliceMeta, CompiledGraph]]] = {}
    for graph_name, compiled in graphs.items():
        slice_meta = workflow_slices.get(graph_name)
        if slice_meta is None:
            continue
        task_key = (
            f"request::{request_id}::{slice_meta.public_graph_name}::"
            f"{slice_meta.template_hash}"
        )
        grouped.setdefault(task_key, []).append((graph_name, slice_meta, compiled))

    global RuntimeGraphBuilder
    if RuntimeGraphBuilder is None:
        from lumilake_server.runtime.runtime_graph import (
            RuntimeGraphBuilder as _Builder,
        )

        RuntimeGraphBuilder = _Builder
    from lumilake_server.runtime.runtime_graph import merge_runtime_graphs

    runtime_builder = RuntimeGraphBuilder()
    tasks: list[DataProfileTaskSpec] = []
    for task_key, entries in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(
            entries,
            key=lambda item: (item[1].slice_index, item[1].slice_start),
        )
        first_meta = ordered[0][1]
        runtime_graphs_by_name = {
            item_graph_name: runtime_builder.build(
                item_compiled,
                task_type_override="data_profile",
                node_prefix=item_graph_name,
            )
            for item_graph_name, _item_slice_meta, item_compiled in ordered
        }
        if len(runtime_graphs_by_name) == 1:
            runtime_graph = next(iter(runtime_graphs_by_name.values()))
        else:
            runtime_graph, _ = merge_runtime_graphs(runtime_graphs_by_name)
        runtime_to_raw = _runtime_to_raw_node_map(runtime_graph)

        node_order: list[str] = []
        nodes: dict[str, DataProfileTaskNode] = {}
        for node_id in runtime_graph.node_order:
            op = runtime_graph.nodes[node_id]
            data_spec = op.data_spec if isinstance(op.data_spec, dict) else {}
            if data_spec.get("type") != "sql":
                continue
            raw_node_id = runtime_to_raw.get(node_id, node_id)
            nodes[node_id] = DataProfileTaskNode(
                node_id=node_id,
                raw_node_id=raw_node_id,
                data_spec=data_spec,
            )
            node_order.append(node_id)
        if not node_order:
            continue

        payload = DataProfileTaskPayload(
            task_key=task_key,
            request_id=request_id,
            public_graph_name=first_meta.public_graph_name,
            template_hash=first_meta.template_hash,
            node_order=node_order,
            nodes=nodes,
        )
        tasks.append(DataProfileTaskSpec(task_key=task_key, payload=payload))
    return tasks


def run_data_profile_task(
    payload: DataProfileTaskPayload,
) -> DataProfileTaskResult:
    try:
        import psycopg
    except Exception as exc:
        raise RuntimeError(
            "DataProfile worker requires psycopg to be installed"
        ) from exc

    result_data: dict[str, list[DataProfileResultRow]] = {}
    for node_id in payload.node_order:
        node = payload.nodes.get(node_id)
        if node is None:
            continue
        data_spec = node.data_spec if isinstance(node.data_spec, dict) else {}
        if data_spec.get("type") != "sql":
            continue
        connection_string = data_spec.get("connection_string")
        template = data_spec.get("template")
        if not isinstance(connection_string, str) or not isinstance(template, str):
            continue
        table = data_spec.get("table")
        if not isinstance(table, str) or not table.strip():
            table = _extract_table_from_sql(template)
        else:
            table = _normalize_table_name(table)
        if not table:
            continue
        num_test_queries_raw = data_spec.get(
            "num_test_queries", envs.LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES
        )
        try:
            num_test_queries = max(1, int(num_test_queries_raw))
        except (TypeError, ValueError):
            num_test_queries = max(1, int(envs.LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES))
        queries = _build_sample_data_profile_queries(
            template=template,
            params=data_spec.get("params"),
            constraints=data_spec.get("constraints"),
            num_samples=num_test_queries,
            node_id=node_id,
        )
        cost_estimates = _estimate_plan_variants(
            psycopg=psycopg,
            connection_string=connection_string,
            queries=queries,
        )
        if not cost_estimates:
            raise RuntimeError(
                "Data profile SQL node produced no valid cost estimates: "
                f"node='{node_id}' query_count={len(queries)}"
            )
        query_name = f"{node_id}_query"
        cache_key = data_profile_key_for_node_query(node_id, query_name)
        result_data[cache_key] = [
            DataProfileResultRow(
                node_id=node_id,
                raw_node_id=node.raw_node_id,
                query_name=query_name,
                connection_string=connection_string.strip(),
                table=table,
                cost_estimates=cost_estimates,
            )
        ]
    return DataProfileTaskResult(data_profile_results=result_data)


def normalize_data_profile_result(
    payload: object,
) -> dict[str, list[DataProfileResultRow]]:
    if isinstance(payload, dict) and "data_profile_results" in payload:
        parsed = DataProfileResultsPayload.model_validate(payload)
    else:
        parsed = DataProfileResultsPayload.model_validate(
            {"data_profile_results": payload}
        )
    return parsed.data_profile_results


def project_data_profile_results_to_runtime_graph(
    *,
    data_profile_results: Mapping[str, Sequence[DataProfileResultRow]],
    target_graph: "RuntimeGraph",
) -> dict[str, list[DataProfileResultRow]]:
    source_by_raw: dict[str, list[DataProfileResultRow]] = {}
    for rows in data_profile_results.values():
        for row in rows:
            source_by_raw.setdefault(row.raw_node_id, []).append(row)

    runtime_to_raw = _runtime_to_raw_node_map(target_graph)
    projected: dict[str, list[DataProfileResultRow]] = {}
    for node_id in target_graph.node_order:
        op = target_graph.nodes[node_id]
        data_spec = op.data_spec if isinstance(op.data_spec, dict) else {}
        if data_spec.get("type") != "sql":
            continue
        raw_node_id = runtime_to_raw.get(node_id)
        if not raw_node_id:
            continue
        matched = source_by_raw.get(raw_node_id)
        if not matched:
            continue
        query_name = f"{node_id}_query"
        cache_key = data_profile_key_for_node_query(node_id, query_name)
        projected[cache_key] = [
            row.model_copy(
                update={
                    "node_id": node_id,
                    "query_name": query_name,
                }
            )
            for row in matched
        ]
    return projected


def data_profile_key_for_node_query(node_id: str, query_name: str) -> str:
    return f"data_profile::{node_id}::{query_name}"


def _runtime_to_raw_node_map(graph: "RuntimeGraph") -> dict[str, str]:
    runtime_to_raw: dict[str, str] = {}
    for raw_node_id, runtime_ids in graph.dsl_to_runtime.items():
        for runtime_id in runtime_ids:
            runtime_to_raw[runtime_id] = raw_node_id
    return runtime_to_raw


def _normalize_table_name(table: str) -> str:
    raw = table.strip().strip(";")
    if not raw:
        return ""
    parts: list[str] = []
    for piece in raw.split("."):
        token = piece.strip()
        if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
            token = token[1:-1]
        if token:
            parts.append(token)
    return ".".join(parts)


def _extract_table_from_sql(template: str) -> str | None:
    match = _SQL_TABLE_PATTERN.search(template)
    if not match:
        return None
    table = str(match.group(1)).strip().strip(";")
    normalized = _normalize_table_name(table)
    return normalized or None


def _collect_data_profile_param_candidates(
    params: Any,
    constraints: Any,
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    if isinstance(params, list):
        for item in params:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if not isinstance(label, str) or not label:
                continue
            data = item.get("data")
            if isinstance(data, dict) and data.get("type") == "list":
                items = data.get("items")
                if isinstance(items, list) and items:
                    result[label] = list(items)
    if isinstance(constraints, list):
        for item in constraints:
            if not isinstance(item, dict):
                continue
            label = item.get("name")
            if not isinstance(label, str) or not label or label in result:
                continue
            variants: list[Any] = []
            candidates = item.get("candidates")
            if isinstance(candidates, list) and candidates:
                variants = list(candidates)
            else:
                min_value = item.get("min")
                max_value = item.get("max")
                if min_value is not None:
                    variants.append(min_value)
                if max_value is not None and max_value != min_value:
                    variants.append(max_value)
                if not variants and item.get("type") == "bool":
                    variants = [True, False]
            if variants:
                result[label] = variants
    return result


def _build_sample_data_profile_queries(
    *,
    template: str,
    params: Any,
    constraints: Any,
    num_samples: int,
    node_id: str | None = None,
) -> list[str]:
    # LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES is interpreted as the maximum
    # number of SQL statements executed per SQL runtime node.
    candidates = _collect_data_profile_param_candidates(params, constraints)
    fields: list[str] = []
    formatter = Formatter()
    for _literal, field_name, _fmt_spec, _conversion in formatter.parse(template):
        if field_name:
            fields.append(field_name)
    if not fields:
        return [template]
    resolved_candidates: dict[str, list[str]] = {}
    for field in sorted(set(fields)):
        values = candidates.get(field)
        if not values:
            raise ValueError(
                "Data profile SQL template has unresolved placeholder: "
                f"node='{node_id}' field='{field}' template='{template}'"
            )
        resolved_candidates[field] = [str(value) for value in values]

    fixed_row = {
        label: values[0]
        for label, values in resolved_candidates.items()
        if len(values) == 1
    }
    variable_candidates = {
        label: values
        for label, values in resolved_candidates.items()
        if len(values) > 1
    }

    def render_query(row: Mapping[str, str]) -> str:
        try:
            return template.format(**row)
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise ValueError(
                "Data profile SQL template has unresolved placeholder during render: "
                f"node='{node_id}' field='{missing}' template='{template}'"
            ) from exc

    if not variable_candidates:
        return [render_query(fixed_row)]

    rng = random.Random(0)
    queries: list[str] = []
    sample_count = max(1, int(num_samples))
    for _ in range(sample_count):
        row = dict(fixed_row)
        for label in sorted(variable_candidates):
            values = variable_candidates[label]
            row[label] = values[rng.randrange(len(values))]
        queries.append(render_query(row))

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        deduped.append(query)
    if not deduped:
        raise RuntimeError(
            "Data profile SQL sampling produced zero queries after dedupe: "
            f"node='{node_id}'"
        )
    return deduped


def _local_data_profile_variants() -> tuple[tuple[str, str, dict[str, str]], ...]:
    named: dict[str, tuple[str, str, dict[str, str]]] = {
        "default": ("default", "planner default", {}),
        "prefer_index": (
            "prefer_index",
            "disable sequential scan to prefer index scans",
            {"enable_seqscan": "off"},
        ),
        "prefer_seq": (
            "prefer_seq",
            "disable index and bitmap scans to force seq scan",
            {"enable_indexscan": "off", "enable_bitmapscan": "off"},
        ),
        "prefer_nestloop": (
            "prefer_nestloop",
            "disable hash/merge joins to bias nested loop",
            {"enable_hashjoin": "off", "enable_mergejoin": "off"},
        ),
    }
    raw = envs.LUMILAKE_LOCAL_DATA_PROFILE_PLAN_VARIANTS
    ordered: list[tuple[str, str, dict[str, str]]] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        variant = named.get(key)
        if variant is None:
            continue
        ordered.append(variant)
    if not ordered:
        return (named["default"],)
    deduped: list[tuple[str, str, dict[str, str]]] = []
    seen: set[str] = set()
    for variant in ordered:
        if variant[0] in seen:
            continue
        seen.add(variant[0])
        deduped.append(variant)
    return tuple(deduped)


def _coerce_footprints(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    footprints: dict[str, int] = {}
    for key, raw_weight in value.items():
        if not isinstance(key, str):
            continue
        try:
            weight = int(raw_weight)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        footprints[key] = weight
    return footprints


def _aggregate_plan_footprints(
    node: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    def safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    relations: dict[str, int] = {}
    indexes: dict[str, int] = {}
    relation_name = node.get("Relation Name")
    index_name = node.get("Index Name")
    footprint = safe_int(node.get("Shared Hit Blocks")) + safe_int(
        node.get("Shared Read Blocks")
    )
    if footprint <= 0:
        footprint = 1
    if isinstance(relation_name, str) and relation_name:
        relations[relation_name] = relations.get(relation_name, 0) + footprint
    if isinstance(index_name, str) and index_name:
        indexes[index_name] = indexes.get(index_name, 0) + footprint
    children = node.get("Plans")
    if isinstance(children, list):
        for child in children:
            if not isinstance(child, Mapping):
                continue
            child_relations, child_indexes = _aggregate_plan_footprints(child)
            for name, weight in child_relations.items():
                relations[name] = relations.get(name, 0) + weight
            for name, weight in child_indexes.items():
                indexes[name] = indexes.get(name, 0) + weight
    return relations, indexes


def _extract_plan_footprints(explain_json: Any) -> dict[str, int]:
    if not isinstance(explain_json, list) or not explain_json:
        return {}
    top = explain_json[0]
    if not isinstance(top, Mapping):
        return {}
    plan = top.get("Plan")
    if not isinstance(plan, Mapping):
        return {}
    relations, indexes = _aggregate_plan_footprints(plan)
    merged: dict[str, int] = {}
    for name, weight in relations.items():
        merged[name] = merged.get(name, 0) + weight
    for name, weight in indexes.items():
        merged[name] = merged.get(name, 0) + weight
    return merged


def _estimate_single_query_variant(
    *,
    conn: Any,
    query: str,
    settings: Mapping[str, str],
) -> dict[str, Any]:
    try:
        with conn.transaction():
            with conn.cursor() as cursor:
                for setting_name, setting_value in settings.items():
                    cursor.execute(f"SET LOCAL {setting_name} = {setting_value}")
                cursor.execute(f"EXPLAIN (FORMAT JSON, VERBOSE) {query}")
                row = cursor.fetchone()
        if not row:
            return {"ok": False}
        plan_json = row[0]
        if not isinstance(plan_json, list) or not plan_json:
            return {"ok": False}
        plan = plan_json[0]
        plan_node = plan.get("Plan") if isinstance(plan, dict) else None
        estimated_cost = None
        estimated_rows = None
        if isinstance(plan_node, dict):
            estimated_cost = plan_node.get("Total Cost")
            estimated_rows = plan_node.get("Plan Rows")
        footprints = _extract_plan_footprints(plan_json)
        return {
            "ok": True,
            "estimated_cost": (
                float(estimated_cost)
                if isinstance(estimated_cost, (int, float))
                else None
            ),
            "estimated_rows": (
                int(estimated_rows)
                if isinstance(estimated_rows, (int, float))
                else None
            ),
            "footprints": footprints,
            "explain_json": plan_json,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _estimate_plan_variants(
    *,
    psycopg: Any,
    connection_string: str,
    queries: Sequence[str],
) -> list[DataProfileCostEstimate]:
    variants = _local_data_profile_variants()
    estimates: list[DataProfileCostEstimate] = []
    with psycopg.connect(connection_string) as conn:
        for plan_id, description, settings in variants:
            sample_raw_costs: list[float] = []
            sample_rows: list[int] = []
            representative_footprints: dict[str, int] = {}
            for query in queries:
                result = _estimate_single_query_variant(
                    conn=conn,
                    query=query,
                    settings=settings,
                )
                if not result.get("ok"):
                    continue
                estimated_cost = result.get("estimated_cost")
                estimated_rows = result.get("estimated_rows")
                if isinstance(estimated_cost, (int, float)):
                    raw_cost = float(estimated_cost)
                    sample_raw_costs.append(raw_cost)
                if isinstance(estimated_rows, (int, float)):
                    sample_rows.append(int(estimated_rows))
                footprints = result.get("footprints")
                if not representative_footprints:
                    representative_footprints = _coerce_footprints(footprints)
            if not sample_raw_costs:
                continue
            avg_raw_cost = sum(sample_raw_costs) / len(sample_raw_costs)
            avg_rows = (
                round(sum(sample_rows) / len(sample_rows)) if sample_rows else None
            )
            estimates.append(
                DataProfileCostEstimate(
                    plan_id=plan_id,
                    description=description,
                    raw_cost=avg_raw_cost,
                    estimated_rows=avg_rows,
                    footprints=representative_footprints,
                )
            )
    estimates.sort(key=lambda item: (item.raw_cost, item.plan_id))
    return estimates
