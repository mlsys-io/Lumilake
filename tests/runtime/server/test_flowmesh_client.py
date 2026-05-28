import asyncio

import httpx
import pytest

from lumilake_server.runtime import flowmesh_client


@pytest.mark.asyncio
async def test_inject_auth_header_single_task() -> None:
    received: list[httpx.Request] = []

    class RecordingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received.append(request)
            return httpx.Response(200)

    client = httpx.AsyncClient(
        transport=RecordingTransport(),
        event_hooks={"request": [flowmesh_client._inject_auth_header]},
    )

    # With a token set the hook should inject the Authorization header.
    flowmesh_client._outgoing_token_var.set("test-token-abc")
    await client.get("https://example.invalid/foo")

    assert len(received) == 1
    assert received[0].headers["Authorization"] == "Bearer test-token-abc"

    # With no token (None) no Authorization header should be added.
    flowmesh_client._outgoing_token_var.set(None)
    received.clear()
    await client.get("https://example.invalid/bar")

    assert len(received) == 1
    assert "authorization" not in received[0].headers

    await client.aclose()


@pytest.mark.asyncio
async def test_inject_auth_header_concurrent_task_isolation() -> None:
    received: dict[str, httpx.Request] = {}

    class RecordingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            received[str(request.url)] = request
            return httpx.Response(200)

    client = httpx.AsyncClient(
        transport=RecordingTransport(),
        event_hooks={"request": [flowmesh_client._inject_auth_header]},
    )

    # Force interleaving: task_a sets its token then yields, task_b sets a
    # different token in its own context, then task_a resumes and sends. If
    # the hook read from a shared global, task_a's request would carry
    # token-B; if it reads from the calling task's ContextVar, it carries
    # token-A.
    a_set = asyncio.Event()
    b_set = asyncio.Event()

    async def task_a() -> None:
        flowmesh_client._outgoing_token_var.set("token-A")
        a_set.set()
        await b_set.wait()
        await client.get("https://example.invalid/task-a")

    async def task_b() -> None:
        await a_set.wait()
        flowmesh_client._outgoing_token_var.set("token-B")
        b_set.set()
        await client.get("https://example.invalid/task-b")

    await asyncio.gather(task_a(), task_b())

    assert (
        received["https://example.invalid/task-a"].headers["Authorization"]
        == "Bearer token-A"
    )
    assert (
        received["https://example.invalid/task-b"].headers["Authorization"]
        == "Bearer token-B"
    )

    await client.aclose()
