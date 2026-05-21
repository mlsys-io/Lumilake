"""Multi-batch data-profile tests.

A single Lumilake request can be split across multiple scheduler batches
(``num_slices > batch_size``). Each batch invokes
``collect_data_profile`` on its own merged runtime graph, and each
lookup must find the SQL profile that ``routes/jobs.py`` runs once at
submit time.

These tests load the ``trading-agent.json`` n8n template (q2
multi-modal-analysis workload). Execution is stubbed:

* ``psycopg`` is monkey-patched to a no-op module so the data profile
  task never touches a database.
* ``_estimate_plan_variants`` is replaced with a fixture returning
  fixed cost estimates so every SQL probe yields a deterministic row
  count.
* ``_compute_minio_listing`` is replaced with a fixture that returns
  the exact folder + sizes the S3 derivation step expects.

The live code path exercised here is the DSL→runtime ID plumbing across
``build_request_data_profile_tasks``,
``LumilakeServer._merge_group_compiled_graph``,
``RuntimeGraphBuilder.build``, ``data_profile_registry``,
``project_data_profile_results_to_runtime_graph``, and
``collect_data_profile``.
"""

import asyncio
import copy
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from lumilake_server.data_profile_models import DataProfileCostEstimate
from lumilake_server.graphs.graph import CompiledGraph, Graph
from lumilake_server.parser.n8n import parse_n8n_payload
from lumilake_server.runtime import data_profile_utils
from lumilake_server.runtime.data_profile_utils import (
    DataProfileSource,
    collect_data_profile,
)
from lumilake_server.runtime.request import WorkflowSliceMeta
from lumilake_server.runtime.runtime_graph import RuntimeGraphBuilder
from lumilake_server.runtime.server import LumilakeServer
from lumilake_server.utils import data_profile_offload

REAL_TEMPLATE = Path(
    "/home/szy/lumilake/lumilake-revision-exp/workloads/templates/n8n/trading-agent.json"
)


@pytest.fixture(autouse=True)
def _clean_registry():
    data_profile_offload.data_profile_registry.clear()
    yield
    data_profile_offload.data_profile_registry.clear()


@pytest.fixture(autouse=True)
def _stub_psycopg(monkeypatch):
    """``run_data_profile_task`` imports ``psycopg`` at call time; the
    test never hits a real database, so swap in a no-op stub."""
    monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace())


@pytest.fixture(autouse=True)
def _stub_plan_estimates(monkeypatch):
    """Every SQL probe returns one deterministic estimate."""

    def _fake_estimates(**_kwargs: Any) -> list[DataProfileCostEstimate]:
        return [
            DataProfileCostEstimate(
                plan_id="default",
                description="planner default",
                raw_cost=1.0,
                estimated_rows=7,
                footprints={},
            )
        ]

    monkeypatch.setattr(
        data_profile_offload,
        "_estimate_plan_variants",
        _fake_estimates,
    )


@pytest.fixture(autouse=True)
def _stub_minio_listing(monkeypatch):
    """The S3 derivation step probes MinIO to check folder existence.

    The trading-agent template's S3 retrieval node fans out under a
    ``news/`` static folder prefix, so the listing only needs that.
    """
    # trading-agent.json's S3 retrieval template resolves to
    # ``unstructured/news-images/{symbol}-{date}.jpg`` under the bucket,
    # so the static folder prefix is ``unstructured/news-images/``.
    listing_sizes = {
        f"unstructured/news-images/{name}.jpg": 4096
        for name in ("NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "GOOGL")
    }
    listing_folders = ["unstructured/news-images/"]
    monkeypatch.setattr(
        data_profile_utils,
        "_compute_minio_listing",
        lambda: (listing_sizes, listing_folders),
    )


def _load_template() -> dict[str, Any]:
    if not REAL_TEMPLATE.is_file():
        pytest.skip(f"Production template not at {REAL_TEMPLATE}")
    return json.loads(REAL_TEMPLATE.read_text())


def _parse_per_slice(
    workflow_json: dict[str, Any],
    slice_inputs: list[dict[str, list[str]]],
    public_name: str,
) -> tuple[dict[str, CompiledGraph], dict[str, WorkflowSliceMeta]]:
    """Parse the workflow once per slice.

    Slice graphs are named ``{public}__slice_{idx+1}`` and ``scope`` is
    set to the public name so DSL ids stay stable across slices — the
    same wiring the submit path uses.
    """
    graphs: dict[str, CompiledGraph] = {}
    slices: dict[str, WorkflowSliceMeta] = {}
    template_hash = "th-fixture"
    total_length = sum(max(len(v) for v in batch.values()) for batch in slice_inputs)
    slice_start = 0
    varying = next(iter(slice_inputs[0].keys()))
    for batch_idx, batch_inputs in enumerate(slice_inputs):
        graph_name = f"{public_name}__slice_{batch_idx + 1}"
        payload = {
            "graphs": [
                {
                    "workflow": copy.deepcopy(workflow_json),
                    "inputs": batch_inputs,
                    "name": graph_name,
                    "scope": public_name,
                }
            ]
        }
        specs = parse_n8n_payload(payload)
        graph_dict = specs[graph_name]
        # parse_n8n_payload returns the serialized graph dict; build the
        # CompiledGraph from it the same way the server's parse_query does.
        compiled = Graph.from_json(graph_dict["graph"]).compile(**graph_dict["inputs"])
        graphs[graph_name] = compiled
        slice_len = max(len(v) for v in batch_inputs.values())
        slices[graph_name] = WorkflowSliceMeta(
            public_graph_name=public_name,
            slice_index=batch_idx,
            slice_start=slice_start,
            slice_length=slice_len,
            total_length=total_length,
            template_hash=template_hash,
            varying_input_keys=(varying,),
        )
        slice_start += slice_len
    return graphs, slices


def _run_inline_data_profile(
    request_id: str,
    graphs: dict[str, CompiledGraph],
    workflow_slices: dict[str, WorkflowSliceMeta],
) -> str:
    """Run one inline data-profile pass for the request.

    Returns the (single) task_key populated into ``data_profile_registry``.
    """
    tasks = data_profile_offload.build_request_data_profile_tasks(
        request_id=request_id,
        graphs=graphs,
        workflow_slices=workflow_slices,
    )
    assert len(tasks) == 1, f"expected exactly one DP task, got {len(tasks)}"
    task = tasks[0]
    result = data_profile_offload.run_data_profile_task(task.payload)
    data_profile_offload.data_profile_registry[task.task_key] = result.model_dump(
        mode="json"
    )
    return task.task_key


def _build_batch_runtime_graph(batch_graphs: dict[str, CompiledGraph], group_key: str):
    """Merge a batch's compiled graphs and build the runtime graph."""
    workflows = []
    for graph_name, compiled in batch_graphs.items():
        slice_meta = _BATCH_SLICE_INFO[graph_name]
        workflows.append(
            types.SimpleNamespace(
                request_id=slice_meta["request_id"],
                public_graph_name=slice_meta["public_graph_name"],
                slice_index=slice_meta["slice_index"],
                slice_start=slice_meta["slice_start"],
                slice_length=slice_meta["slice_length"],
                total_length=slice_meta["total_length"],
                template_hash=slice_meta["template_hash"],
                varying_input_keys=slice_meta["varying_input_keys"],
                workflow_id=graph_name,
                dsl_graph=compiled,
            )
        )
    merged = LumilakeServer._merge_group_compiled_graph(workflows)
    builder = RuntimeGraphBuilder()
    return builder.build(merged, node_prefix=group_key)


_BATCH_SLICE_INFO: dict[str, dict[str, Any]] = {}


def _record_slice_info(
    workflow_slices: dict[str, WorkflowSliceMeta], request_id: str
) -> None:
    """Stash slice meta for ``_build_batch_runtime_graph`` to pick up."""
    _BATCH_SLICE_INFO.clear()
    for graph_name, meta in workflow_slices.items():
        _BATCH_SLICE_INFO[graph_name] = {
            "request_id": request_id,
            "public_graph_name": meta.public_graph_name,
            "slice_index": meta.slice_index,
            "slice_start": meta.slice_start,
            "slice_length": meta.slice_length,
            "total_length": meta.total_length,
            "template_hash": meta.template_hash,
            "varying_input_keys": meta.varying_input_keys,
        }


def _run_collect(batch_graph, task_key: str) -> dict[str, Any]:
    sources = [DataProfileSource(task_key=task_key, org_id="default")]
    return asyncio.run(
        collect_data_profile(
            request_id="req-mb-test",
            data_profile_graphs={task_key: batch_graph},
            data_profile_sources={task_key: sources},
        )
    )


def test_collect_data_profile_batch_2_succeeds_with_stable_parser_scope() -> None:
    """4-input job (slices 1..4), batch size 2 → batches {1,2} and {3,4}.

    Batch 2 reuses the data-profile payload populated at submit time.
    With ``parser_scope = public_graph_name``, every slice parses with
    the same scope, so DSL ids are stable across slices. The DP-task
    merge and the batch-2 merge end up in the same DSL id space, and
    projection-by-raw-node-id finds the SQL row counts the S3 derivation
    needs.
    """
    workflow_json = _load_template()
    request_id = "req-mb-test"

    slice_inputs = [
        {"Stock": ["NVDA"]},
        {"Stock": ["AAPL"]},
        {"Stock": ["MSFT"]},
        {"Stock": ["GOOG"]},
    ]
    graphs, slice_metas = _parse_per_slice(workflow_json, slice_inputs, "g0")
    _record_slice_info(slice_metas, request_id)

    task_key = _run_inline_data_profile(request_id, graphs, slice_metas)
    payload = data_profile_offload.data_profile_registry[task_key]
    sample_raw_ids = {
        row["raw_node_id"]
        for rows in payload["data_profile_results"].values()
        for row in rows
    }

    batch_2_graphs = {name: graphs[name] for name in ("g0__slice_3", "g0__slice_4")}
    batch_2_runtime = _build_batch_runtime_graph(batch_2_graphs, task_key)
    batch_2_dsl_ids = set(batch_2_runtime.dsl_to_runtime.keys())

    # Invariant: payload-row DSL ids and batch-2 graph DSL ids must
    # overlap on retrieval ops so projection can re-key the rows for
    # batch-2's runtime graph.
    retrieval_payload_ids = {
        rid for rid in sample_raw_ids if rid.startswith("retrieval_")
    }
    retrieval_batch_ids = {
        rid for rid in batch_2_dsl_ids if rid.startswith("retrieval_")
    }
    assert retrieval_payload_ids, "expected payload to record retrieval rows"
    assert retrieval_batch_ids, "expected batch-2 graph to carry retrieval DSL ids"
    assert retrieval_payload_ids & retrieval_batch_ids, (
        "DP-payload retrieval ids must overlap with batch-2 graph "
        "retrieval ids; disjoint sets indicate the parser scope is "
        "drifting between slices."
    )

    # Success path: collect_data_profile completes; the projected SQL
    # rows carry estimated_rows for batch-2's S3 derivation to consume.
    result = _run_collect(batch_2_runtime, task_key)
    assert result, "expected at least one projected data-profile entry for batch 2"
