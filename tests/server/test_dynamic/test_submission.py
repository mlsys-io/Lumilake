"""End-to-end tests for the dynamic-workflow submission path.

A dynamic workflow is a YAML spec with a root ``type: dynamic`` plus a
plaintext ``goal`` and a ``driver`` section. Submitting it renders round 0 into
a native graph and runs the planning loop server-side; invalid specs are
rejected at submission time. A YAML without ``type: dynamic`` is a normal
static workflow.

These are the core E2E scenarios: the loop running to STOP, cross-round
registry handoff, cancel propagation, the round budget, failure paths reaching
a terminal state, leaf-output integrity, and the static-workflow regression.
"""

import asyncio
import json
import logging
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from lumid_hooks import PrincipalContext, ResourceRef

import lumilake_server.utils.job_storage as job_storage_module
from lumilake_server import hooks
from lumilake_server.middleware import TraceIdMiddleware
from lumilake_server.routes import jobs as job_routes_module
from lumilake_server.runtime.protocol import LumilakeResponse
from lumilake_server.utils.job_storage import InMemoryJobStorage

_DEMO_PRINCIPAL = PrincipalContext(
    principal_id="alice",
    org_id="demo",
    external_id="alice@example.com",
    principal_type="user",
    scopes=["admin"],
)

_VALID_DYNAMIC_YAML = """
name: dynamic
type: dynamic
goal: analyze market data
driver:
  model: Qwen/Qwen3-8B
  max_rounds: 4
  max_nodes_per_round: 4
  output_location:
    type: s3
    prefix: dynamic/data-free/
"""

_SUBGRAPH_PLAN = {
    "next": "subgraph",
    "ops": [
        {
            "id": "q1",
            "op": "DataRetrievalOp",
            "inputs": ["Symbols"],
            "data_spec": {
                "type": "lumid",
                "mode": "sql",
                "output_format": "jsonl",
                "verify": False,
                "template": "SELECT * FROM x",
                "params": [{"label": "s", "node": "Symbols"}],
            },
        }
    ],
}


class _AllowAllIdentity:
    name = "test.identity"

    async def resolve(
        self, token: str, logger: logging.Logger
    ) -> PrincipalContext | None:
        return _DEMO_PRINCIPAL.model_copy(deep=True) if token == "token" else None


class _AllowAllPermissions:
    name = "test.permissions"

    async def accessible_ids(
        self,
        principal: PrincipalContext,
        kind: str,
        action: str,
        logger: logging.Logger,
    ) -> frozenset[str] | None:
        return None

    async def require(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        action: str,
        logger: logging.Logger,
    ) -> None:
        return None


class _AllowAllRegistrar:
    name = "test.registrar"

    async def register(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        return None

    async def deregister(
        self,
        principal: PrincipalContext,
        resource: ResourceRef,
        logger: logging.Logger,
    ) -> None:
        return None


class _AllowAllGuards:
    name = "test.guards"

    async def check(self, principal: PrincipalContext, logger: logging.Logger) -> None:
        return None


class _FakeRuntimeManager:
    def set_dispatch_token(self, job_id: str, token: str | None) -> None:
        return None


class _FakeRuntimeServer:
    is_started = True

    def __init__(self) -> None:
        self.cancel_calls: list[str] = []
        self.parse_query_calls: list[dict[str, dict[str, Any]]] = []
        self.execute_calls: list[str] = []
        self.executed_graphs: list[dict[str, Any]] = []
        self.configs: list[Any] = []
        self.plans: list[dict[str, Any]] | None = None
        self.fail_rounds: set[int] = set()
        self.hang_rounds: set[int] = set()
        self.no_result_rounds: set[int] = set()
        self.hang_event = asyncio.Event()
        self.fail_cancel = False
        self.cancel_raises_cancelled = False
        self.omit_leaf_outputs = False
        self.hanging_requests: set[str] = set()
        self._traces: dict[str, list[str]] = {}
        self.runtime_manager = _FakeRuntimeManager()

    def parse_query(self, graph_specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.parse_query_calls.append(graph_specs)
        from lumilake_server.graphs import Graph

        return {
            name: Graph.from_json(spec["graph"]).compile(**spec["inputs"])
            for name, spec in graph_specs.items()
        }

    async def execute(
        self,
        graphs: dict[str, Any],
        request_id: str | None = None,
        config: Any | None = None,
        workflow_slices: dict[str, Any] | None = None,
    ) -> LumilakeResponse:
        self.execute_calls.append(request_id or "")
        self.executed_graphs.append(graphs)
        self.configs.append(config)
        if request_id:
            self._traces[request_id] = [f"trace-{request_id}"]
        round_index = len(self.execute_calls) - 1
        if round_index in self.hang_rounds:
            if request_id:
                self.hanging_requests.add(request_id)
            try:
                await self.hang_event.wait()
            finally:
                if request_id:
                    self.hanging_requests.discard(request_id)
            if self.cancel_raises_cancelled:
                from lumilake_server.runtime.protocol import RequestCancelledError

                raise RequestCancelledError(request_id or "")
        if round_index in self.fail_rounds:
            return LumilakeResponse(outputs={}, error_info=[{"message": "boom"}])
        if round_index in self.no_result_rounds:
            return LumilakeResponse(outputs={})
        plan = {"next": "STOP"}
        if self.plans:
            plan = self.plans[min(round_index, len(self.plans) - 1)]
        outputs: dict[str, Any] = {"round": {"plan": [json.dumps(plan)]}}
        # Every round archives its leaves (the round's actual subgraph, not the
        # returned plan); emit one value per leaf OutputOp so the loop can
        # validate and accumulate an observation (unless the test asks to omit
        # them).
        if not self.omit_leaf_outputs:
            leaf_names = self._leaf_output_names(graphs)
            for name in leaf_names:
                outputs["round"][name] = ['{"market_cap": 10}']
        return LumilakeResponse(outputs=outputs)

    def _leaf_output_names(self, graphs: dict[str, Any]) -> list[str]:
        from lumilake_server.ops import OutputOp

        names: list[str] = []
        for graph in graphs.values():
            for op in graph.iter_ops(OutputOp):
                if str(op.name).startswith("leaf_"):
                    names.append(op.name)
        return names

    def trace_ids_for_request(self, job_id: str) -> list[str]:
        return list(self._traces.get(job_id, []))

    def optimization_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def selection_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def clustering_seconds_for_request(self, job_id: str) -> float:
        return 0.0

    def release_request_workflows(self, job_id: str) -> None:
        self._traces.pop(job_id, None)

    async def cancel_request(self, job_id: str) -> None:
        self.cancel_calls.append(job_id)
        self.hang_event.set()
        if self.fail_cancel:
            raise RuntimeError("cancel RPC failed")

    async def preview_schedule(self, **kwargs: Any) -> Any:
        class _Schedule:
            worker_assignment: dict[str, Any] = {}

        class _Preview:
            request_id: str = "preview"
            selected_workers: list[str] = []
            schedule: Any = _Schedule()
            runtime_graph_node_counts: dict[str, int] = {}
            merged_runtime_node_count: int = 0
            selection_seconds: float = 0.0
            clustering_seconds: float = 0.0
            optimization_seconds: float = 0.0

        return _Preview()


@pytest.fixture(autouse=True)
def _reset_hook_state() -> Iterator[None]:
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.USAGE_SINKS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.RESOURCE_REGISTRARS.clear()
    hooks.IDENTITY_PROVIDERS.append(_AllowAllIdentity())
    hooks.PERMISSION_CHECKERS.append(_AllowAllPermissions())
    hooks.SUBMISSION_GUARDS.append(_AllowAllGuards())
    hooks.RESOURCE_REGISTRARS.append(_AllowAllRegistrar())
    yield
    hooks.IDENTITY_PROVIDERS.clear()
    hooks.PERMISSION_CHECKERS.clear()
    hooks.SUBMISSION_GUARDS.clear()
    hooks.RESOURCE_REGISTRARS.clear()
    hooks.USAGE_SINKS.clear()


@pytest.fixture
def job_routes(monkeypatch: pytest.MonkeyPatch) -> Any:
    storage = InMemoryJobStorage()
    job_storage_module._job_storage = storage
    job_routes_module.jobs.clear()
    job_routes_module._job_storage = storage
    fake_server = _FakeRuntimeServer()
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_started_instance",
        classmethod(lambda cls: fake_server),
    )
    monkeypatch.setattr(
        job_routes_module.LumilakeServer,
        "get_instance",
        classmethod(lambda cls: fake_server),
    )
    monkeypatch.setattr(
        job_routes_module,
        "build_request_data_profile_tasks",
        lambda **kwargs: [],
    )

    async def _dump_output_locations(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        job_routes_module,
        "_dump_output_locations",
        _dump_output_locations,
    )
    setattr(job_routes_module, "_fake_runtime_server", fake_server)
    return job_routes_module


@pytest.fixture
def app(job_routes: Any) -> FastAPI:
    application = FastAPI()
    application.state.logger = logging.getLogger("test.dynamic_submission")
    application.state.background_tasks = set()
    application.add_middleware(TraceIdMiddleware)
    application.include_router(job_routes.router)
    return application


def _submit_body(yaml_text: str) -> dict[str, Any]:
    return {
        "data": [
            {
                "name": "demo",
                "workflow": yaml_text,
                "inputs": {"Symbols": ["NVDA"]},
                "output_location": {"type": "s3", "prefix": "dynamic/data-free/"},
            }
        ]
    }


async def _run_background(app: FastAPI) -> None:
    for task in list(app.state.background_tasks):
        await task


@pytest.mark.anyio
async def test_dynamic_submit_runs_loop_to_stop(app: FastAPI, job_routes: Any) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [
        _SUBGRAPH_PLAN,
        {"next": "STOP"},
    ]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "completed"
    assert len(record.child_job_ids) == 2
    # The parent's execution progress reflects the completed rounds: two
    # subgraph rounds ran before STOP, so succeeded=2 and the step is done.
    execution = record.progress.execution
    assert execution.completed is True
    assert execution.details is not None
    assert execution.details.succeeded == 2
    assert execution.details.pending == 0
    # The parent's result carries both the per-round plan and the per-round
    # leaf results (the actual data each round produced), as nested JSON.
    # Round 0 is the initial proposal with an empty subgraph, so it archives
    # no leaves; round 1 executes the proposed subgraph and archives its leaf
    # data; the terminal STOP round has no leaves.
    assert record.result is not None
    round_outputs = record.result.outputs["round"]
    assert len(round_outputs["plan"]) == 2
    assert len(round_outputs["results"]) == 2
    assert round_outputs["results"][0] == {}
    subgraph_leaf = round_outputs["results"][1]
    assert len(subgraph_leaf) == 1
    assert list(subgraph_leaf.values())[0] == [{"market_cap": 10}]


@pytest.mark.anyio
async def test_dynamic_child_carries_chain_lineage(
    app: FastAPI, job_routes: Any
) -> None:
    """Each dynamic round's child job carries the parent's id as ``chain_id``
    and a strictly increasing zero-based ``chain_round``."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    # Each round must emit a distinct node id (nodes are immutable).
    second_plan = json.loads(json.dumps(_SUBGRAPH_PLAN))
    second_plan["ops"][0]["id"] = "q2"
    fake_server.plans = [_SUBGRAPH_PLAN, second_plan, {"next": "STOP"}]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "completed"
    # Three child rounds ran; each config must name the parent as its chain and
    # carry its own zero-based round index.
    assert len(fake_server.configs) == 3
    for round_index, config in enumerate(fake_server.configs):
        assert config.chain_id == job_id
        assert config.chain_round == round_index
    assert [c.chain_round for c in fake_server.configs] == [0, 1, 2]


@pytest.mark.anyio
async def test_dynamic_cancel_propagates_to_inflight_child(
    app: FastAPI, job_routes: Any, wait_for_inflight_child: Any
) -> None:
    """Cancelling a dynamic parent through the real endpoint must cancel its
    in-flight child."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    fake_server.hang_rounds = {1}
    loop_task = asyncio.create_task(_run_background(app))
    # Wait for the round-1 child (the one that hangs) to be genuinely in flight.
    child_id = await wait_for_inflight_child(job_routes, job_id)
    # Cancel through the real endpoint.
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        cancel_resp = await client.post(
            f"/jobs/{job_id}/cancel",
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert cancel_resp.status_code == 200, cancel_resp.text
    await loop_task
    assert job_routes.jobs[job_id].status == "cancelled"
    assert (
        job_routes.jobs[child_id].status == "cancelled"
    ), f"in-flight child not cancelled; got {job_routes.jobs[child_id].status!r}"
    # The backend cancel RPC must have been issued for the in-flight child.
    assert (
        child_id in fake_server.cancel_calls
    ), f"child backend cancel RPC not issued; cancel_calls={fake_server.cancel_calls}"


_LIBRARY_DYNAMIC_YAML = """
name: dynamic
type: dynamic
goal: analyze market data
driver:
  model: Qwen/Qwen3-8B
  max_rounds: 4
  max_nodes_per_round: 4
  output_location:
    type: s3
    prefix: dynamic/data-free/
library:
  sector_market_cap:
    op: DataRetrievalOp
    data_spec:
      type: lumid
      mode: sql
      output_format: jsonl
      verify: false
      template: SELECT sector, AVG(market_cap) FROM reference.profile GROUP BY sector
"""


@pytest.mark.anyio
async def test_dynamic_multi_round_registry_handoff(
    app: FastAPI, job_routes: Any
) -> None:
    """Round 0's emitted library ref is resolved and stored in the registry,
    then round 1 references that node by id and the round-1 graph builds against
    the resolved op."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_LIBRARY_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    # Round 0 emits a library ref; round 1 emits an op referencing that node.
    fake_server.plans = [
        {
            "next": "subgraph",
            "ops": [{"ref": "sector_market_cap", "id": "s0", "inputs": []}],
        },
        {
            "next": "subgraph",
            "ops": [
                {
                    "id": "q1",
                    "op": "DataRetrievalOp",
                    "inputs": ["s0"],
                    "data_spec": {
                        "type": "lumid",
                        "mode": "sql",
                        "output_format": "jsonl",
                        "verify": False,
                        "template": "SELECT * FROM t WHERE id = {p}",
                        "params": [{"label": "p", "node": "s0"}],
                    },
                }
            ],
        },
        {"next": "STOP"},
    ]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "completed"
    assert len(record.child_job_ids) == 3
    # The registry must hold the RESOLVED op (has an "op" field), not a bare
    # ref. The round-1 child built successfully only if round-0's node was
    # resolved and stored with its real type; a bare ref would fail validation.
    # Assert the round-1 child completed (its graph built and ran).
    round1_child = job_routes.jobs[record.child_job_ids[1]]
    assert round1_child.status == "completed"


@pytest.mark.anyio
async def test_dynamic_max_rounds_cutoff(app: FastAPI, job_routes: Any) -> None:
    spec_yaml = _VALID_DYNAMIC_YAML.replace("  max_rounds: 4\n", "  max_rounds: 2\n")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(spec_yaml),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    # Each round must emit a distinct node id (nodes are immutable).
    second_plan = json.loads(json.dumps(_SUBGRAPH_PLAN))
    second_plan["ops"][0]["id"] = "q2"
    fake_server.plans = [_SUBGRAPH_PLAN, second_plan]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "completed"
    assert len(record.child_job_ids) == 2


@pytest.mark.anyio
async def test_dynamic_round_failure_fails_parent(
    app: FastAPI, job_routes: Any
) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    fake_server.plans = [_SUBGRAPH_PLAN, {"next": "STOP"}]
    fake_server.fail_rounds = {1}
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "failed"
    assert "terminated with status" in (record.error or "")


@pytest.mark.anyio
async def test_dynamic_stop_round_missing_leaf_fails_parent(
    app: FastAPI, job_routes: Any
) -> None:
    # The terminal STOP round must still satisfy the leaf-output contract: a
    # STOP that omits an expected archived leaf fails the parent.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_VALID_DYNAMIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    fake_server = job_routes._fake_runtime_server
    # Round 0 emits a subgraph (q2); round 1 runs it and returns STOP, but
    # omits q2's leaf output.
    plan = json.loads(json.dumps(_SUBGRAPH_PLAN))
    plan["ops"][0]["id"] = "q2"
    fake_server.plans = [plan, {"next": "STOP"}]
    fake_server.omit_leaf_outputs = True
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "failed"
    assert "missing expected leaf output" in (record.error or "")


@pytest.mark.anyio
async def test_dynamic_submit_missing_goal_returns_422(app: FastAPI) -> None:
    spec = "name: dynamic\ntype: dynamic\ndriver:\n  model: Qwen/Qwen3-8B\n"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(spec),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_dynamic_submit_empty_symbol_returns_422(app: FastAPI) -> None:
    body = _submit_body(_VALID_DYNAMIC_YAML)
    body["data"][0]["inputs"] = {"Symbols": [""]}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=body,
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 422


_STATIC_YAML = """
name: static
inputs:
  Symbols: ["NVDA"]
ops:
  - id: "Greeting"
    op: FormatOp
    inputs: [Symbols]
    template: "Hello, {name}!"
    format_kwargs:
      name: Symbols
outputs:
  - name: reply
    ref: "Greeting"
"""


@pytest.mark.anyio
async def test_static_yaml_without_type_runs_as_static(
    app: FastAPI, job_routes: Any
) -> None:
    """A YAML workflow with no ``type`` field defaults to static: it is parsed
    as a normal graph and dispatched through the static ``_run_job`` path, not
    the dynamic planning loop."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/jobs",
            json=_submit_body(_STATIC_YAML),
            headers={"Authorization": "Bearer token", "Workflow-Format": "yaml"},
        )
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["data"]["job_id"]
    await _run_background(app)
    record = job_routes.jobs[job_id]
    assert record.status == "completed"
    # A static job is a single job with no dynamic child rounds.
    assert record.child_job_ids == []
