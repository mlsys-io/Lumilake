from typing import cast

from support.runtime_server import FakeHandler

from lumilake_server.runtime.protocol import LumilakeRequestConfig
from lumilake_server.runtime.request import RequestHandler
from lumilake_server.runtime.server import RequestState


def _state() -> RequestState:
    return RequestState(
        handler=cast(RequestHandler, FakeHandler()),
        config=LumilakeRequestConfig(user_id="u", principal_id="u"),
        pending_workflows=set(),
    )


def test_record_job_manager_time_splits_share_across_members(server_factory) -> None:
    server = server_factory()
    server._requests["req-a"] = _state()
    server._requests["req-b"] = _state()

    server._record_job_manager_time(
        member_request_ids={"req-a", "req-b"},
        selection_seconds=0.40,
        clustering_seconds=0.10,
    )

    assert server._requests["req-a"].selection_seconds == 0.20
    assert server._requests["req-a"].clustering_seconds == 0.05
    assert server._requests["req-b"].selection_seconds == 0.20
    assert server._requests["req-b"].clustering_seconds == 0.05

    assert server.selection_seconds_for_request("req-a") == 0.20
    assert server.clustering_seconds_for_request("req-a") == 0.05
    assert server.selection_seconds_for_request("missing") is None


def test_record_job_manager_time_accumulates_across_batches(server_factory) -> None:
    server = server_factory()
    server._requests["req-a"] = _state()

    server._record_job_manager_time(
        member_request_ids={"req-a"},
        selection_seconds=0.10,
        clustering_seconds=0.02,
    )
    server._record_job_manager_time(
        member_request_ids={"req-a"},
        selection_seconds=0.30,
        clustering_seconds=0.05,
    )

    assert server._requests["req-a"].selection_seconds == 0.40
    assert server._requests["req-a"].clustering_seconds == 0.07


def test_record_job_manager_time_ignores_unknown_requests(server_factory) -> None:
    server = server_factory()
    server._requests["req-a"] = _state()

    server._record_job_manager_time(
        member_request_ids={"req-a", "ghost"},
        selection_seconds=0.20,
        clustering_seconds=0.04,
    )

    assert server._requests["req-a"].selection_seconds == 0.10
    assert server._requests["req-a"].clustering_seconds == 0.02


def test_record_job_manager_time_no_op_on_zero_time(server_factory) -> None:
    server = server_factory()
    server._requests["req-a"] = _state()

    server._record_job_manager_time(
        member_request_ids={"req-a"},
        selection_seconds=0.0,
        clustering_seconds=0.0,
    )

    assert server._requests["req-a"].selection_seconds == 0.0
    assert server._requests["req-a"].clustering_seconds == 0.0
