from dataclasses import dataclass, field

from lumilake_server.graphs import CompiledGraph
from lumilake_server.runtime.data_profile_utils import DataProfileSource
from lumilake_server.runtime.protocol import LumilakeRequestConfig, LumilakeResponse
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


@dataclass(slots=True)
class RequestInfo:
    request_id: str

    runtime_graphs: dict[str, RuntimeGraph]
    data_profile_graphs: dict[str, RuntimeGraph]

    data_profile_sources: dict[str, list[DataProfileSource]] = field(
        default_factory=dict
    )

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
