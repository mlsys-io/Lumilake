"""The template-driven dynamic job loop.

A template-driven run renders a declared body per round instead of asking a
planner for one, so its loop dispatches every round from ``driver.template``
and stops when the previous round's export holds no rows.
"""

import asyncio
import uuid
from typing import Any

from lumid_hooks import PrincipalContext

from lumilake.log import init_child_logger, set_trace_id
from lumilake_server.dynamic.driver import (
    STOP,
    DriverProtocolError,
    build_round,
    compute_observation,
    result_outputs,
    round_output_location,
    walk_archived_items,
)
from lumilake_server.dynamic.spec import DynamicSpec
from lumilake_server.dynamic.template import (
    DriverTemplate,
    archive_result_items,
    run_round,
)
from lumilake_server.routes import jobs as job_routes
from lumilake_server.routes.jobs import (
    _IO_LOCATION_ADAPTER,
    TERMINAL_JOB_STATUSES,
    JobRecord,
    _admit_subgraph,
    _extract_archived_outputs,
    _extract_leaf_outputs,
    _fire_parent_terminal_hooks,
    _flatten_leaf_outputs,
    _now,
    _submit_dynamic_child,
)
from lumilake_server.runtime.protocol import (
    HardwareRequirements,
    LumilakeResponse,
    Priority,
)
from lumilake_server.schemas.io import IOLocation
from lumilake_server.schemas.progress import ProgressDetails, ProgressStep

logger = init_child_logger("TemplateJobs")


async def run_template_job(
    parent_job_id: str,
    spec: DynamicSpec,
    record: JobRecord,
    symbols: list[str],
    output_location: IOLocation,
    priority: Priority,
    principal: PrincipalContext,
    runtime_token: str | None,
    trace_id: str,
    template: DriverTemplate,
    optimizer_type: str | None = None,
    hardware_requirements: HardwareRequirements | None = None,
) -> None:
    """Run a template-driven dynamic job, one child job per round.

    Round 1 renders ``first`` and every later round ``next``; each round runs
    as one whole-set slice. A round whose predecessor exported no rows stops
    the loop, and the parent fails if any child round does not complete.
    """
    set_trace_id(trace_id)
    try:
        async with job_routes.jobs_lock:
            terminal_before_start = record.status in TERMINAL_JOB_STATUSES
            if not terminal_before_start:
                record.status = "running"
                record.started_at = _now()
        if terminal_before_start:
            return
        await asyncio.to_thread(job_routes._job_storage.save, record)

        run_namespace = f"run-{uuid.uuid4().hex}"
        max_rounds = spec.driver.max_rounds
        max_nodes = spec.driver.max_nodes_per_round
        node_registry: dict[str, dict[str, Any]] = {}
        observations: list[str] = []
        plans: list[dict[str, Any]] = []
        round_results: list[dict[str, list[str]]] = []
        all_results: dict[str, list[dict[str, Any]]] = {}

        async with job_routes.jobs_lock:
            record.progress.execution = ProgressStep(
                completed=False,
                details=ProgressDetails(
                    succeeded=0,
                    failed=0,
                    pending=max_rounds,
                    dispatched=0,
                ),
            )
        await asyncio.to_thread(job_routes._job_storage.save, record)

        round_index = 1
        while round_index <= max_rounds:
            topology = list(node_registry.keys())
            try:
                template_round = run_round(template, round_index, node_registry)
                if template_round.stopped(all_results):
                    plans.append({"next": STOP, "source": "template"})
                    break
                current_subgraph = _admit_subgraph(
                    template_round.plan,
                    node_registry,
                    max_nodes,
                    spec.library,
                    all_results,
                )
                plans.append(template_round.record())
                round_build = build_round(
                    current_subgraph,
                    node_registry=node_registry,
                    round_index=round_index,
                    goal=spec.goal,
                    observations=observations,
                    topology=topology,
                    preview_width=spec.driver.preview_width,
                    model=spec.driver.model,
                    max_tokens=spec.driver.max_tokens,
                    temperature=spec.driver.temperature,
                    threshold=spec.driver.threshold,
                    library=spec.library,
                    results=all_results,
                    export=template_round.plan.export,
                    chat_template_kwargs=spec.driver.chat_template_kwargs,
                    max_model_len=spec.driver.max_model_len,
                    gpu_memory_utilization=spec.driver.gpu_memory_utilization,
                    dtype=spec.driver.dtype,
                    extra_engine_kwargs=spec.driver.extra_engine_kwargs,
                    include_proposer=False,
                    rows=1,
                )
            except (DriverProtocolError, ValueError) as exc:
                async with job_routes.jobs_lock:
                    if record.status in TERMINAL_JOB_STATUSES:
                        return
                    record.status = "failed"
                    record.error = f"dynamic round {round_index} failed to build: {exc}"
                    record.finished_at = _now()
                await asyncio.to_thread(job_routes._job_storage.save, record)
                return
            try:
                child_job_id = await _submit_dynamic_child(
                    parent_job_id=parent_job_id,
                    graph=round_build.graph,
                    symbols=symbols,
                    forwarded_inputs=round_build.forwarded_inputs,
                    output_location=_IO_LOCATION_ADAPTER.validate_python(
                        round_output_location(
                            output_location.model_dump(), run_namespace, round_index
                        )
                    ),
                    priority=priority,
                    principal=principal,
                    runtime_token=runtime_token,
                    trace_id=trace_id,
                    round_index=round_index,
                    optimizer_type=optimizer_type,
                    hardware_requirements=hardware_requirements,
                    job_timeout=spec.driver.job_timeout,
                    one_slice=True,
                )
            except TimeoutError as exc:
                async with job_routes.jobs_lock:
                    if record.status in TERMINAL_JOB_STATUSES:
                        return
                    record.status = "failed"
                    record.error = str(exc)
                    record.finished_at = _now()
                await asyncio.to_thread(job_routes._job_storage.save, record)
                return
            if child_job_id is None:
                return
            async with job_routes.jobs_lock:
                child = job_routes.jobs[child_job_id]
                child_status = child.status
            if child_status != "completed":
                async with job_routes.jobs_lock:
                    if record.status in TERMINAL_JOB_STATUSES:
                        return
                    if child_status == "cancelled":
                        record.status = "cancelled"
                        record.error = (
                            f"dynamic round {round_index} ({child_job_id}) "
                            "was cancelled"
                        )
                    else:
                        record.status = "failed"
                        record.error = (
                            f"dynamic round {round_index} ({child_job_id}) "
                            f"terminated with status {child_status!r}"
                        )
                    record.finished_at = _now()
                await asyncio.to_thread(job_routes._job_storage.save, record)
                return
            result = child.result
            if result is None:
                async with job_routes.jobs_lock:
                    if record.status in TERMINAL_JOB_STATUSES:
                        return
                    record.status = "failed"
                    record.error = f"dynamic round {round_index} produced no result"
                    record.finished_at = _now()
                await asyncio.to_thread(job_routes._job_storage.save, record)
                return
            outputs = result_outputs({"result": {"outputs": result.outputs}})
            leaf_outputs = _extract_leaf_outputs(outputs, round_build.leaf_output_names)
            export_outputs = _extract_archived_outputs(
                outputs, round_build.export_output_names, "export_"
            )
            internal_configs: dict[str, dict[str, Any]] = {}
            for name, user_id in zip(
                round_build.leaf_output_names, round_build.leaf_user_ids
            ):
                internal_configs[name[len("leaf_") :]] = node_registry[user_id]
            for name, user_id in zip(
                round_build.export_output_names, round_build.export_user_ids
            ):
                internal_configs[name[len("export_") :]] = node_registry[user_id]
            walked_leaves = walk_archived_items(leaf_outputs, internal_configs)
            round_results.append(_flatten_leaf_outputs(walked_leaves))
            observations.append(
                compute_observation(walked_leaves, spec.driver.preview_width)
            )
            leaf_internal_to_user = {
                name[len("leaf_") :]: user
                for name, user in zip(
                    round_build.leaf_output_names, round_build.leaf_user_ids
                )
            }
            for internal_id, values in leaf_outputs.items():
                user_id = leaf_internal_to_user[internal_id]
                all_results[user_id] = archive_result_items(
                    _flatten_leaf_outputs({internal_id: values})[internal_id]
                )
            export_internal_to_user = {
                name[len("export_") :]: user
                for name, user in zip(
                    round_build.export_output_names, round_build.export_user_ids
                )
            }
            for internal_id, values in export_outputs.items():
                user_id = export_internal_to_user[internal_id]
                all_results[user_id] = archive_result_items(
                    _flatten_leaf_outputs({internal_id: values})[internal_id]
                )
            round_index += 1
            completed_rounds = round_index - 1
            async with job_routes.jobs_lock:
                if record.status in TERMINAL_JOB_STATUSES:
                    return
                record.progress.execution = ProgressStep(
                    completed=False,
                    details=ProgressDetails(
                        succeeded=completed_rounds,
                        failed=0,
                        pending=max_rounds - completed_rounds,
                        dispatched=0,
                    ),
                )
            await asyncio.to_thread(job_routes._job_storage.save, record)
        async with job_routes.jobs_lock:
            if record.status in TERMINAL_JOB_STATUSES:
                return
            record.status = "completed"
            record.finished_at = _now()
            record.progress.execution = ProgressStep(
                completed=True,
                details=ProgressDetails(
                    succeeded=len(plans),
                    failed=0,
                    pending=0,
                    dispatched=0,
                ),
            )
            record.result = LumilakeResponse(
                outputs={"round": {"plan": plans, "results": round_results}}
            )
        await asyncio.to_thread(job_routes._job_storage.save, record)
    except Exception as exc:  # noqa: BLE001 - any escape must fail the parent
        async with job_routes.jobs_lock:
            if record.status in TERMINAL_JOB_STATUSES:
                return
            record.status = "failed"
            record.error = f"dynamic run failed: {exc}"
            record.finished_at = _now()
        try:
            await asyncio.to_thread(job_routes._job_storage.save, record)
        except Exception as save_exc:  # noqa: BLE001 - persistence is best-effort
            logger.error(
                "failed to persist failed dynamic parent %s: %s",
                parent_job_id,
                save_exc,
            )
    finally:
        await _fire_parent_terminal_hooks(
            record, principal, parent_job_id, runtime_token
        )
