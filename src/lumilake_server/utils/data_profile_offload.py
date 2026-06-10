import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from lumilake_server.utils.lumid_data_client import profile as lumid_profile

if TYPE_CHECKING:
    from lumilake_server.runtime.request import WorkflowSliceMeta
    from lumilake_server.runtime.runtime_graph import RuntimeGraph

RuntimeGraphBuilder = None


@dataclass(slots=True)
class _MergeGroupWorkflow:
    """Fields read by ``LumilakeServer._merge_group_compiled_graph`` and its
    helpers. Lets ``build_request_data_profile_tasks`` synthesize merge
    inputs from ``(graph, slice_meta)`` entries without pulling in the full
    ``RequestWorkflow`` type."""

    request_id: str
    public_graph_name: str
    template_hash: str
    slice_index: int
    slice_start: int
    slice_length: int
    total_length: int
    workflow_id: str
    varying_input_keys: tuple[str, ...]
    dsl_graph: CompiledGraph


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
    from lumilake_server.runtime.server import LumilakeServer

    runtime_builder = RuntimeGraphBuilder()
    tasks: list[DataProfileTaskSpec] = []
    for task_key, entries in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(
            entries,
            key=lambda item: (item[1].slice_index, item[1].slice_start),
        )
        first_meta = ordered[0][1]
        merge_inputs = [
            _MergeGroupWorkflow(
                request_id=request_id,
                public_graph_name=meta.public_graph_name,
                template_hash=meta.template_hash,
                slice_index=meta.slice_index,
                slice_start=meta.slice_start,
                slice_length=meta.slice_length,
                total_length=meta.total_length,
                workflow_id=graph_name,
                varying_input_keys=meta.varying_input_keys,
                dsl_graph=compiled,
            )
            for graph_name, meta, compiled in ordered
        ]
        merged_compiled = LumilakeServer._merge_group_compiled_graph(merge_inputs)
        runtime_graph = runtime_builder.build(
            merged_compiled,
            task_type_override="data_profile",
            node_prefix=task_key,
        )
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
    result_data: dict[str, list[DataProfileResultRow]] = {}
    for node_id in payload.node_order:
        node = payload.nodes.get(node_id)
        if node is None:
            continue
        data_spec = node.data_spec if isinstance(node.data_spec, dict) else {}
        if data_spec.get("type") != "sql":
            continue
        template = data_spec.get("template")
        if not isinstance(template, str):
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


_TYPE_DEFAULT_SAMPLES: dict[str, Any] = {
    "string": "",
    "str": "",
    "text": "",
    "int": 0,
    "integer": 0,
    "number": 0,
    "float": 0,
    "bool": False,
    "boolean": False,
    "datetime": "1970-01-01",
    "date": "1970-01-01",
    "timestamp": "1970-01-01",
    "array": "[]",
    "list": "[]",
    "object": "{}",
    "dict": "{}",
    "json": "{}",
}


def _type_default_sample(type_name: str | None) -> Any:
    if not isinstance(type_name, str):
        return ""
    return _TYPE_DEFAULT_SAMPLES.get(type_name.lower(), "")


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
            if not variants:
                variants = [_type_default_sample(item.get("type"))]
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
                f"node='{node_id}' param='{field}' template='{template}'. "
                "The upstream op did not supply a sample value; add "
                "'sample_value' to the upstream data_spec, attach "
                "'structural_outputs' to the LLM, or set "
                "LUMILAKE_DISABLE_DATA_PROFILE=1."
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
                "Data profile SQL template has unresolved placeholder "
                f"during render: node='{node_id}' param='{missing}' "
                f"template='{template}'"
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


def _estimate_plan_variants(
    *,
    queries: Sequence[str],
) -> list[DataProfileCostEstimate]:
    """Call lumid-data-app POST /profile for each sample query and each plan
    variant, then average the per-variant costs and rows across queries.

    Each ``/profile`` response variant contains a ``footprints`` field
    (``dict[str, int]``). The first non-empty footprint map for a variant
    is used as the representative footprint.
    """
    variants = _local_data_profile_variants()
    plans_payload = [
        {"plan_id": plan_id, "settings": settings}
        for plan_id, _description, settings in variants
    ]
    description_by_id = {
        plan_id: description for plan_id, description, _settings in variants
    }
    per_variant_costs: dict[str, list[float]] = {v[0]: [] for v in variants}
    per_variant_rows: dict[str, list[int]] = {v[0]: [] for v in variants}
    per_variant_footprints: dict[str, dict[str, int]] = {v[0]: {} for v in variants}

    for query in queries:
        response_variants = lumid_profile(query, plans_payload)
        for item in response_variants:
            if not isinstance(item, Mapping):
                continue
            plan_id = item.get("plan_id")
            if not isinstance(plan_id, str) or plan_id not in per_variant_costs:
                continue
            raw_cost = item.get("raw_cost")
            estimated_rows = item.get("estimated_rows")
            footprints_raw = item.get("footprints")
            if isinstance(raw_cost, (int, float)):
                per_variant_costs[plan_id].append(float(raw_cost))
            if isinstance(estimated_rows, (int, float)):
                per_variant_rows[plan_id].append(int(estimated_rows))
            if not per_variant_footprints[plan_id]:
                coerced = _coerce_footprints(footprints_raw)
                if coerced:
                    per_variant_footprints[plan_id] = coerced

    estimates: list[DataProfileCostEstimate] = []
    for plan_id, _description, _settings in variants:
        costs = per_variant_costs[plan_id]
        if not costs:
            continue
        avg_raw_cost = sum(costs) / len(costs)
        rows = per_variant_rows[plan_id]
        avg_rows = round(sum(rows) / len(rows)) if rows else None
        estimates.append(
            DataProfileCostEstimate(
                plan_id=plan_id,
                description=description_by_id[plan_id],
                raw_cost=avg_raw_cost,
                estimated_rows=avg_rows,
                footprints=per_variant_footprints[plan_id],
            )
        )
    estimates.sort(key=lambda item: (item.raw_cost, item.plan_id))
    return estimates
