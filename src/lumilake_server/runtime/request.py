from dataclasses import dataclass, field

from lumilake_server.graphs import CompiledGraph
from lumilake_server.runtime.data_profile_utils import DataProfileSource
from lumilake_server.runtime.protocol import (
    HardwareRequirements,
    LumilakeRequestConfig,
    LumilakeResponse,
)
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.utils.queue import TSQueue


@dataclass(slots=True)
class WorkflowSliceMeta:
    public_graph_name: str
    slice_index: int
    slice_start: int
    slice_length: int
    total_length: int
    template_hash: str
    varying_input_keys: tuple[str, ...] = ()
    whole_set: bool = False
    """True when this slice holds a template round's whole input set, whose row
    count is its own answer."""


@dataclass(slots=True)
class RequestInfo:
    request_id: str

    runtime_graphs: dict[str, RuntimeGraph]
    data_profile_graphs: dict[str, RuntimeGraph]

    data_profile_sources: dict[str, list[DataProfileSource]] = field(
        default_factory=dict
    )
    hardware_requirements: HardwareRequirements | None = None
    """Per-batch hardware override mirrored from the originating
    ``LumilakeRequestConfig``. Batched items share the same hardware tuple
    because the job manager partitions by hardware."""

    member_request_ids: set[str] = field(default_factory=set)
    """The originating job (``req-*``) ids this execution batch belongs to.
    API credentials are stored per job id, so dispatch resolves them through
    these ids rather than the synthetic ``exec-*`` request id."""

    task_owner_members: dict[str, set[str]] = field(default_factory=dict)
    """Attribution of each canonical (post-dedup) node id to the member
    request ids whose pre-dedup graphs contributed it. Used to compute
    per-job usage for jobs sharing a FlowMesh workflow."""

    gpu_worker_ids: set[str] = field(default_factory=set)
    """The selected workers whose profile has a GPU, read from the same
    profiles the optimizer scheduled with. CPU nodes offer every other
    selected worker, so an idle GPU worker never takes a CPU step."""

    store_records: bool = False
    """Whether any member job asked for the record store; a batch may hold a
    storing and a non-storing job, and only the storing one writes and
    indexes."""
    member_store: dict[str, bool] = field(default_factory=dict)
    """Whether each member job asked for the store, keyed by job id."""
    member_rounds: dict[str, int] = field(default_factory=dict)
    """The semantic round each member job's ops belong to, so two dynamic jobs
    at different rounds never store records under one round."""
    round_number: int = 0
    """The semantic round this batch's ops belong to, ``0`` for a standalone
    job."""

    reuse_records: bool = False
    """Whether a member job asked the store to answer what it already holds,
    so the plan runs before dispatch and only its misses are submitted."""

    # These are to be set and modified by the optimizer and processor
    runtime_graph: RuntimeGraph = field(init=False)
    data_profile_graph: RuntimeGraph = field(init=False)
    output_node_map: dict[str, tuple[str, str]] = field(init=False)
    batch_id: str = field(init=False)


class RequestHandler:
    def __init__(
        self,
        query: dict[str, RuntimeGraph],
        data_profile_graphs: dict[str, RuntimeGraph],
        dsl_graphs: dict[str, CompiledGraph],
        workflow_slices: dict[str, WorkflowSliceMeta],
        request_id: str,
        config: LumilakeRequestConfig | None = None,
    ) -> None:
        self._request_id = request_id
        self.query = query
        self.data_profile_graphs = data_profile_graphs
        self.dsl_graphs = dsl_graphs
        self.workflow_slices = workflow_slices
        self.config = (
            LumilakeRequestConfig(user_id=request_id, principal_id=request_id)
            if config is None
            else config
        )
        self._result_queue: TSQueue[LumilakeResponse] = TSQueue()

    @property
    def request_id(self) -> str:
        return self._request_id

    async def put_result(self, result: LumilakeResponse) -> None:
        await self._result_queue.put(result)

    async def get_result(self) -> LumilakeResponse:
        return await self._result_queue.get()
