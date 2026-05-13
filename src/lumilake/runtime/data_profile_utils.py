import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from string import Formatter
from typing import Any

import yaml
from minio import Minio
from pydantic import BaseModel, ConfigDict, Field

from lumilake import envs
from lumilake.data_profile_models import DataProfileCostEstimate, DataProfileResultRow
from lumilake.runtime.protocol import RequestCancelledError
from lumilake.runtime.runtime_graph import RuntimeGraph
from lumilake.utils.data_profile_offload import (
    data_profile_registry,
    normalize_data_profile_result,
    project_data_profile_results_to_runtime_graph,
)
from lumilake.utils.parsing import split_bucket_prefix
from lumilake.utils.s3 import create_minio_client


class DataProfileSource(BaseModel):
    task_key: str = Field(min_length=1)
    org_id: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


CancellationCallback = Callable[[], bool | Awaitable[bool]]


def data_profile_key_for_node_query(node_id: str, query_name: str) -> str:
    return f"data_profile::{node_id}::{query_name}"


def normalize_table_name(table: str) -> str:
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


def coerce_data_profile_footprints(value: Any) -> dict[str, int]:
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


async def _is_cancelled(callback: CancellationCallback | None) -> bool:
    if callback is None:
        return False
    result = callback()
    if inspect.isawaitable(result):
        return bool(await result)
    return bool(result)


def _has_sql_data_profile_nodes(graph: RuntimeGraph) -> bool:
    for node_id in graph.node_order:
        if graph.nodes[node_id].data_spec.get("type") == "sql":
            return True
    return False


def _has_s3_data_profile_nodes(graph: RuntimeGraph) -> bool:
    for node_id in graph.node_order:
        if graph.nodes[node_id].data_spec.get("type") == "s3":
            return True
    return False


def _runtime_to_raw_node_map(graph: RuntimeGraph) -> dict[str, str]:
    runtime_to_raw: dict[str, str] = {}
    for raw_node_id, runtime_ids in graph.dsl_to_runtime.items():
        for runtime_id in runtime_ids:
            runtime_to_raw[runtime_id] = raw_node_id
    return runtime_to_raw


def _normalize_s3_template_path(template: str) -> str:
    """Normalize an S3 retrieval template to a leading-slash-stripped path."""
    cleaned = template.strip()
    if not cleaned:
        raise ValueError("S3 retrieval template is empty")
    if cleaned.startswith("s3://"):
        raise ValueError(
            "S3 retrieval template must be a relative path, not an s3:// URI"
        )
    normalized = cleaned.lstrip("/")
    if normalized.endswith("/"):
        raise ValueError(f"S3 retrieval template must resolve to a file: {normalized}")
    return normalized


def _normalize_snapshot_s3_path(path: str) -> str:
    cleaned = path.strip()
    if not cleaned:
        raise ValueError("S3 snapshot file path is required")
    if cleaned.endswith("/"):
        raise ValueError("S3 snapshot must contain file names, not folder prefixes")
    return cleaned.lstrip("/")


def _best_sql_estimated_rows(
    node_id: str,
    projected_sql: Mapping[str, Sequence[DataProfileResultRow]],
) -> int | None:
    key = data_profile_key_for_node_query(node_id, f"{node_id}_query")
    rows_raw = projected_sql.get(key, ())
    if not rows_raw:
        return None
    best_score = float("inf")
    best_rows: int | None = None
    for row in rows_raw:
        for estimate in row.cost_estimates:
            score = estimate.raw_cost
            if score < best_score and estimate.estimated_rows is not None:
                best_score = score
                best_rows = estimate.estimated_rows
    return best_rows


def _extract_template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    formatter = Formatter()
    for _literal, field_name, _fmt_spec, _conversion in formatter.parse(template):
        if field_name:
            fields.add(field_name)
    return fields


def _collect_s3_params(
    data_spec: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    literal_values: dict[str, list[str]] = {}
    node_refs: dict[str, str] = {}
    params = data_spec.get("params")
    if not isinstance(params, list):
        return literal_values, node_refs
    for item in params:
        if not isinstance(item, Mapping):
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label:
            continue
        data = item.get("data")
        if isinstance(data, Mapping) and data.get("type") == "list":
            values = data.get("items")
            if isinstance(values, list) and values:
                literal_values[label] = [str(value) for value in values]
            continue
        node_id = item.get("node")
        if isinstance(node_id, str) and node_id:
            node_refs[label] = node_id
    return literal_values, node_refs


def _template_list_length(
    *,
    template_fields: set[str],
    literal_values: Mapping[str, Sequence[str]],
) -> int:
    if not template_fields:
        return 1
    missing = sorted(field for field in template_fields if field not in literal_values)
    if missing:
        raise ValueError(
            "S3 retrieval template cannot be resolved from literal lists; missing: "
            + ", ".join(missing)
        )
    max_len = max(len(literal_values[field]) for field in template_fields)
    if max_len <= 0:
        raise ValueError("S3 retrieval literal params are empty")
    for field in template_fields:
        size = len(literal_values[field])
        if size not in {1, max_len}:
            raise ValueError(
                "S3 retrieval literal params must have length 1 or shared max length"
            )
    return max_len


def _folder_prefix_for_path(path: str) -> str:
    """Return the folder prefix (with trailing slash) of a snapshot file path."""
    normalized = _normalize_snapshot_s3_path(path)
    folder = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
    return f"{folder}/" if folder else ""


def _normalize_s3_folder_prefix(path: str) -> str:
    """Normalize a folder prefix: strip leading slash, ensure trailing slash."""
    cleaned = path.strip().lstrip("/")
    if not cleaned:
        return ""
    return cleaned.rstrip("/") + "/"


def _resolve_folders_from_template(
    *,
    template_path: str,
    literal_values: Mapping[str, Sequence[str]],
    node_fields: set[str],
) -> list[str]:
    folder_template = _folder_prefix_for_path(template_path)
    folder_fields = _extract_template_fields(folder_template)
    dynamic_folder_fields = sorted(
        field for field in folder_fields if field in node_fields
    )
    if dynamic_folder_fields:
        raise RuntimeError(
            "S3 data profile requires a static folder prefix; SQL-driven placeholders "
            "inside folder path are not supported: " + ", ".join(dynamic_folder_fields)
        )
    if not folder_fields:
        return [folder_template]
    folder_count = _template_list_length(
        template_fields=folder_fields,
        literal_values=literal_values,
    )
    folders: list[str] = []
    for idx in range(folder_count):
        values: dict[str, str] = {}
        for field in sorted(folder_fields):
            variants = literal_values[field]
            values[field] = variants[idx] if len(variants) > 1 else variants[0]
        folders.append(_normalize_s3_folder_prefix(folder_template.format(**values)))
    deduped = sorted(set(folders))
    return deduped


def _average_size_for_folders(
    *,
    folders: Sequence[str],
    listing_sizes: Mapping[str, int | None],
    listing_folders: Sequence[str],
    graph_key: str,
    node_id: str,
    org_id: str,
) -> float:
    folder_samples: dict[str, list[int]] = {folder: [] for folder in folders}
    folder_entry_counts: dict[str, int] = {folder: 0 for folder in folders}
    known_folders = set(listing_folders)
    for file_path, size_bytes in listing_sizes.items():
        file_folder = _folder_prefix_for_path(file_path)
        known_folders.add(file_folder)
        if file_folder in folder_entry_counts:
            folder_entry_counts[file_folder] += 1
        if size_bytes is None:
            continue
        if file_folder in folder_samples:
            folder_samples[file_folder].append(size_bytes)
    for folder in folders:
        if folder not in known_folders:
            raise RuntimeError(
                f"S3 data profile folder missing from snapshot for graph='{graph_key}'"
                f" node='{node_id}' folder='{folder}' org_id='{org_id}'"
            )
    for folder, entry_count in folder_entry_counts.items():
        if entry_count <= 0:
            raise RuntimeError(
                f"S3 data profile folder is empty in snapshot for graph='{graph_key}'"
                f" node='{node_id}' folder='{folder}' org_id='{org_id}'"
            )
    folder_avgs: list[float] = []
    for folder, sizes in folder_samples.items():
        if not sizes:
            raise RuntimeError(
                "S3 data profile folder snapshot size unavailable for"
                f" graph='{graph_key}' node='{node_id}' folder='{folder}'"
                f" org_id='{org_id}'"
            )
        folder_avgs.append(sum(sizes) / len(sizes))
    return sum(folder_avgs) / len(folder_avgs)


def _select_s3_source(
    *,
    graph_key: str,
    selected_source: DataProfileSource | None,
    candidates: Sequence[DataProfileSource],
) -> DataProfileSource:
    if selected_source is not None:
        return selected_source
    unique_org_ids = sorted({source.org_id for source in candidates})
    if len(unique_org_ids) != 1:
        raise RuntimeError(
            "Failed to resolve S3 data profile source org_id for graph "
            f"'{graph_key}': candidates={[(c.task_key, c.org_id) for c in candidates]}"
        )
    return DataProfileSource(task_key=graph_key, org_id=unique_org_ids[0])


def _derive_s3_profile_for_graph(
    *,
    graph: RuntimeGraph,
    graph_key: str,
    org_id: str,
    projected_sql: Mapping[str, Sequence[DataProfileResultRow]],
    listing_sizes: Mapping[str, int | None],
    listing_folders: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    runtime_to_raw = _runtime_to_raw_node_map(graph)
    derived: dict[str, list[dict[str, Any]]] = {}
    per_file = envs.LUMILAKE_S3_PROFILE_COST_PER_FILE
    per_mib = envs.LUMILAKE_S3_PROFILE_COST_PER_MIB

    for node_id in graph.node_order:
        data_spec = graph.nodes[node_id].data_spec
        if data_spec.get("type") != "s3":
            continue
        template = data_spec.get("template")
        if not isinstance(template, str):
            raise RuntimeError(
                f"S3 data profile requires template for node '{node_id}'"
            )
        template_path = _normalize_s3_template_path(template)
        template_fields = _extract_template_fields(template_path)
        literal_values, node_refs = _collect_s3_params(data_spec)

        node_fields = {field for field in template_fields if field in node_refs}
        literal_fields = {field for field in template_fields if field in literal_values}
        missing_fields = sorted(template_fields - node_fields - literal_fields)
        if missing_fields:
            raise RuntimeError(
                "S3 data profile cannot resolve template placeholders for "
                f"graph '{graph_key}' node '{node_id}': {missing_fields}"
            )

        folders = _resolve_folders_from_template(
            template_path=template_path,
            literal_values=literal_values,
            node_fields=node_fields,
        )
        avg_file_size_bytes = _average_size_for_folders(
            folders=folders,
            listing_sizes=listing_sizes,
            listing_folders=listing_folders,
            graph_key=graph_key,
            node_id=node_id,
            org_id=org_id,
        )

        if node_fields:
            # Template references classify into one of two buckets:
            #  - SQL runtime nodes: file count = estimated_rows from the
            #    upstream SQL profile.
            #  - DSL-only ops (InputOp and siblings) that never materialize
            #    as runtime nodes: file count defaults to 1 per supplied
            #    input value (conservative lower bound).
            # A placeholder pointing at a non-SQL *runtime* node is neither.
            # That's almost always a graph-wiring bug (e.g. the retrieval
            # template references an S3 / LLM node whose output shape
            # doesn't line up with file-count semantics), so raise rather
            # than silently fall back to 1.
            sql_nodes: set[str] = set()
            input_driven_fields: set[str] = set()
            invalid_refs: dict[str, str] = {}
            for field in node_fields:
                ref_node_id = node_refs[field]
                ref_node = graph.nodes.get(ref_node_id)
                if ref_node is None:
                    # DSL-layer op that isn't part of the runtime graph —
                    # almost always an InputOp consumed as a retrieval key.
                    input_driven_fields.add(field)
                    continue
                ref_spec = ref_node.data_spec
                if ref_spec.get("type") == "sql":
                    sql_nodes.add(ref_node_id)
                else:
                    # Concrete runtime node whose spec isn't SQL — modeling
                    # error. Record the offender for a clear error message.
                    invalid_refs[field] = (
                        f"{ref_node_id} (type={ref_spec.get('type')!r})"
                    )
            if invalid_refs:
                pairs = ", ".join(
                    f"{field}->{ref}" for field, ref in sorted(invalid_refs.items())
                )
                raise RuntimeError(
                    "S3 data profile template placeholder points at a "
                    "non-SQL runtime node — file count is undefined. "
                    f"graph='{graph_key}' node='{node_id}' refs=[{pairs}]. "
                    "Only SQL-typed runtime nodes or DSL-only InputOp refs "
                    "can source template placeholders."
                )
            if sql_nodes and input_driven_fields:
                # Mixing sources would leave the file count ambiguous —
                # one placeholder says "as many files as a SQL projection
                # returns" and another says "one per input value". Reject
                # rather than guess; agents should pick one source shape.
                raise RuntimeError(
                    "S3 data profile cannot mix SQL-driven and input-driven "
                    "template placeholders on the same retrieval node. "
                    f"graph='{graph_key}' node='{node_id}' "
                    f"sql_nodes={sorted(sql_nodes)} "
                    f"input_driven={sorted(input_driven_fields)}"
                )
            if sql_nodes:
                if len(sql_nodes) != 1:
                    raise RuntimeError(
                        "S3 data profile requires a single SQL source for "
                        "templated file count. "
                        f"graph='{graph_key}' node='{node_id}' "
                        f"sql_nodes={sorted(sql_nodes)}"
                    )
                sql_node_id = next(iter(sql_nodes))
                estimated_rows = _best_sql_estimated_rows(sql_node_id, projected_sql)
                if estimated_rows is None:
                    raise RuntimeError(
                        "S3 data profile missing SQL estimated_rows for "
                        f"graph='{graph_key}' node='{node_id}' "
                        f"sql_node='{sql_node_id}'"
                    )
                file_count = estimated_rows
            else:
                # Input-driven: one fetch per supplied input value. The
                # profiler doesn't see the inputs map, so use 1 as a lower
                # bound — callers that care about input-count-driven cost
                # should override the estimate downstream.
                file_count = 1
        else:
            file_count = _template_list_length(
                template_fields=template_fields,
                literal_values=literal_values,
            )

        total_size_bytes = round(file_count * avg_file_size_bytes)
        raw_cost = file_count * per_file + (total_size_bytes / 1048576.0) * per_mib
        query_name = f"{node_id}_query"
        cache_key = data_profile_key_for_node_query(node_id, query_name)
        connection_string = data_spec.get("connection_string")
        if not isinstance(connection_string, str):
            connection_string = ""
        row = DataProfileResultRow(
            node_id=node_id,
            raw_node_id=runtime_to_raw.get(node_id, node_id),
            query_name=query_name,
            connection_string=connection_string.strip(),
            table=template_path,
            cost_estimates=[
                DataProfileCostEstimate(
                    plan_id="s3_profile",
                    description="snapshot-based s3 retrieval estimate",
                    raw_cost=raw_cost,
                    estimated_files=file_count,
                    total_size_bytes=total_size_bytes,
                    avg_file_size_bytes=avg_file_size_bytes,
                    footprints={},
                )
            ],
        )
        derived[cache_key] = [row.model_dump(mode="json")]
    return derived


def dump_data_profile_yaml(
    data_profile: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(data_profile, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _compute_minio_listing() -> tuple[dict[str, int | None], list[str]]:
    """Return ``(file_sizes, folder_paths)`` for the compute MinIO data area.

    ``file_sizes`` maps relative paths under ``S3_USER_DATA_PREFIX`` to byte size;
    ``folder_paths`` is the set of folder prefixes implied.
    """
    endpoint = envs.S3_ENDPOINT
    access_key = envs.S3_ACCESS_KEY
    connection_value = envs.S3_CONNECTION_VALUE
    data_prefix_raw = envs.S3_USER_DATA_PREFIX
    if not (endpoint and access_key and connection_value and data_prefix_raw):
        return {}, []
    client: Minio = create_minio_client(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=connection_value,
        cert_file=envs.S3_CERT_FILE,
    )
    sizes: dict[str, int | None] = {}
    folders: set[str] = set()
    bucket, key_prefix = split_bucket_prefix(data_prefix_raw)
    key_prefix_norm = key_prefix.rstrip("/")
    scan_prefix = f"{key_prefix_norm}/" if key_prefix_norm else ""
    for obj in client.list_objects(bucket, prefix=scan_prefix, recursive=True):
        if obj.is_dir or not obj.object_name:
            continue
        name = obj.object_name
        rel = name[len(scan_prefix) :] if scan_prefix else name
        if not rel:
            continue
        sizes[rel] = obj.size
        parts = rel.split("/")
        for idx in range(1, len(parts)):
            folders.add(f"{'/'.join(parts[:idx])}/")
    return sizes, sorted(folders)


async def collect_data_profile(
    *,
    request_id: str,
    data_profile_graphs: Mapping[str, RuntimeGraph],
    data_profile_sources: Mapping[str, Sequence[DataProfileSource]] | None = None,
    cancellation_callback: CancellationCallback | None = None,
    output_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if await _is_cancelled(cancellation_callback):
        raise RequestCancelledError(request_id)

    result_data: dict[str, list[dict[str, Any]]] = {}
    s3_listing_cache: tuple[dict[str, int | None], list[str]] | None = None

    for graph_key in sorted(data_profile_graphs.keys()):
        graph = data_profile_graphs[graph_key]
        if not graph.nodes:
            continue
        has_sql = _has_sql_data_profile_nodes(graph)
        has_s3 = _has_s3_data_profile_nodes(graph)
        if not has_sql and not has_s3:
            continue

        configured_sources = (
            data_profile_sources.get(graph_key) if data_profile_sources else None
        )
        candidates: list[DataProfileSource] = (
            list(configured_sources) if configured_sources else []
        )
        if not candidates:
            candidates = [DataProfileSource(task_key=graph_key, org_id="default")]

        selected: DataProfileSource | None = None
        completed_payload: dict[str, Any] | None = None
        if has_sql:
            # ``routes/jobs.py`` runs the profile inline at submit time and
            # stashes the result in ``data_profile_registry`` keyed by
            # task_key. A missing entry here means the profile step failed
            # during parse.
            for source in candidates:
                if payload := data_profile_registry.pop(source.task_key, None):
                    completed_payload = payload
                    selected = source
                    break
            if completed_payload is None:
                keys = ", ".join(
                    f"{source.task_key}@{source.org_id}" for source in candidates
                )
                raise RuntimeError(
                    "Data profile result not found for graph "
                    f"'{graph_key}' (candidates: {keys}); check submission logs"
                )

        projected_sql: dict[str, list[DataProfileResultRow]] = {}
        if completed_payload is not None:
            normalized = normalize_data_profile_result(completed_payload)
            projected_sql = project_data_profile_results_to_runtime_graph(
                data_profile_results=normalized,
                target_graph=graph,
            )
            result_data.update(
                {
                    key: [row.model_dump(mode="json") for row in rows]
                    for key, rows in projected_sql.items()
                }
            )

        s3_entries = 0
        if has_s3:
            s3_source = _select_s3_source(
                graph_key=graph_key,
                selected_source=selected,
                candidates=candidates,
            )
            if s3_listing_cache is None:
                s3_listing_cache = await asyncio.to_thread(_compute_minio_listing)
            sizes, folders = s3_listing_cache
            derived_s3 = _derive_s3_profile_for_graph(
                graph=graph,
                graph_key=graph_key,
                org_id=s3_source.org_id,
                projected_sql=projected_sql,
                listing_sizes=sizes,
                listing_folders=folders,
            )
            s3_entries = len(derived_s3)
            result_data.update(derived_s3)

        if logger is not None:
            logger.info(
                "Data profile resolved for graph '%s' via task_key=%s org_id=%s"
                " sql_entries=%d s3_entries=%d",
                graph_key,
                selected.task_key if selected else "<none>",
                selected.org_id if selected else "<none>",
                len(projected_sql),
                s3_entries,
            )

    if output_path is not None:
        await asyncio.to_thread(dump_data_profile_yaml, result_data, output_path)
    return result_data
