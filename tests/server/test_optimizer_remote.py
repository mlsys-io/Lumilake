"""Tests for RemoteOptimizer: bearer forwarding, URL-unset error, response mapping."""

import json

import httpx
import pytest
import respx
from lumilake import envs as envs_mod

from lumilake_server.hooks.security import runtime_token_var
from lumilake_server.runtime.optimizer.base import Schedule
from lumilake_server.runtime.optimizer.remote import RemoteOptimizer
from lumilake_server.runtime.runtime_graph import RuntimeGraph
from lumilake_server.runtime.runtime_ops import RuntimeOp


def _make_graph() -> RuntimeGraph:
    op = RuntimeOp(
        node_id="n1",
        task_type="inference",
        backend="vllm",
        model="gpt2",
        data_spec={},
        model_spec={},
        inference_spec={},
        dependencies=(),
    )
    return RuntimeGraph(
        nodes={"n1": op},
        node_order=["n1"],
        output_node_map={"n1": "out"},
        dsl_to_runtime={},
    )


def test_remote_optimizer_raises_when_url_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs_mod, "LUMILAKE_REMOTE_OPTIMIZER_URL", "")
    with pytest.raises(ValueError, match="LUMILAKE_REMOTE_OPTIMIZER_URL"):
        RemoteOptimizer(optimizer_type="halo-greedy")


def test_remote_optimizer_raises_when_optimizer_type_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        envs_mod, "LUMILAKE_REMOTE_OPTIMIZER_URL", "http://127.0.0.1:9000"
    )
    with pytest.raises(ValueError, match="optimizer_type"):
        RemoteOptimizer(optimizer_type=None)


@respx.mock
def test_remote_optimizer_posts_schedule_and_maps_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        envs_mod, "LUMILAKE_REMOTE_OPTIMIZER_URL", "http://127.0.0.1:9000"
    )

    route = respx.post("http://127.0.0.1:9000/api/v1/optimizer/schedule").mock(
        return_value=httpx.Response(
            200,
            json={"worker_assignment": {"w1": ["n1"]}},
        )
    )

    optimizer = RemoteOptimizer(optimizer_type="halo-greedy")
    graph = _make_graph()
    schedule = optimizer.generate_schedule(
        graph,
        worker_names=["w1"],
        worker_profiles={"w1": {"gpu": 1}},
    )

    assert isinstance(schedule, Schedule)
    assert schedule.worker_assignment == {"w1": ["n1"]}
    assert route.called


@respx.mock
def test_remote_optimizer_sends_constructor_optimizer_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        envs_mod, "LUMILAKE_REMOTE_OPTIMIZER_URL", "http://127.0.0.1:9000"
    )

    captured_body: dict = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content))
        return httpx.Response(200, json={"worker_assignment": {"w1": ["n1"]}})

    respx.post("http://127.0.0.1:9000/api/v1/optimizer/schedule").mock(
        side_effect=_capture
    )

    optimizer = RemoteOptimizer(
        base_url="http://127.0.0.1:9000", optimizer_type="halo-greedy"
    )
    optimizer.generate_schedule(
        _make_graph(),
        worker_names=["w1"],
        worker_profiles={"w1": {}},
    )

    assert captured_body.get("optimizer_type") == "halo-greedy"


@respx.mock
def test_remote_optimizer_forwards_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        envs_mod, "LUMILAKE_REMOTE_OPTIMIZER_URL", "http://127.0.0.1:9000"
    )

    captured_headers: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"worker_assignment": {"w1": ["n1"]}})

    respx.post("http://127.0.0.1:9000/api/v1/optimizer/schedule").mock(
        side_effect=_capture
    )

    token = runtime_token_var.set("test-bearer-xyz")
    try:
        optimizer = RemoteOptimizer(optimizer_type="halo-greedy")
        schedule = optimizer.generate_schedule(
            _make_graph(),
            worker_names=["w1"],
            worker_profiles={"w1": {}},
        )
    finally:
        runtime_token_var.reset(token)

    assert schedule.worker_assignment == {"w1": ["n1"]}
    assert captured_headers.get("authorization") == "Bearer test-bearer-xyz"


@respx.mock
def test_remote_optimizer_no_bearer_when_token_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        envs_mod, "LUMILAKE_REMOTE_OPTIMIZER_URL", "http://127.0.0.1:9000"
    )

    captured_headers: dict[str, str] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        captured_headers.update(dict(request.headers))
        return httpx.Response(200, json={"worker_assignment": {}})

    respx.post("http://127.0.0.1:9000/api/v1/optimizer/schedule").mock(
        side_effect=_capture
    )

    token = runtime_token_var.set(None)
    try:
        optimizer = RemoteOptimizer(optimizer_type="halo-greedy")
        optimizer.generate_schedule(
            _make_graph(),
            worker_names=[],
            worker_profiles={},
        )
    finally:
        runtime_token_var.reset(token)

    assert "authorization" not in captured_headers


@respx.mock
def test_remote_optimizer_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        envs_mod, "LUMILAKE_REMOTE_OPTIMIZER_URL", "http://127.0.0.1:9000"
    )

    respx.post("http://127.0.0.1:9000/api/v1/optimizer/schedule").mock(
        return_value=httpx.Response(400, json={"detail": "bad request"})
    )

    optimizer = RemoteOptimizer(optimizer_type="halo-greedy")
    with pytest.raises(httpx.HTTPStatusError):
        optimizer.generate_schedule(
            _make_graph(),
            worker_names=["w1"],
            worker_profiles={"w1": {}},
        )


# ── URL scheme guard ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("http://remote-optimizer.example.com", id="plain_http_remote"),
        pytest.param("http://0.0.0.0:9000", id="wildcard_bind_address"),
    ],
)
def test_remote_optimizer_rejects_unsafe_url(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Reject http:// on non-loopback addresses (https or loopback http only)."""
    monkeypatch.setattr(envs_mod, "LUMILAKE_REMOTE_OPTIMIZER_URL", url)
    with pytest.raises(ValueError, match="https://"):
        RemoteOptimizer(optimizer_type="halo-greedy")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"env_url": "http://127.0.0.1:9000"},
            id="loopback_http",
        ),
        pytest.param(
            {"env_url": "https://remote-optimizer.example.com"},
            id="https_remote",
        ),
        pytest.param(
            {"env_url": "", "base_url": "https://custom.example.com"},
            id="base_url_kwarg_overrides_empty_env",
        ),
    ],
)
def test_remote_optimizer_accepts_valid_url(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict
) -> None:
    """Valid URLs (loopback http, https, or explicit base_url kwarg) must not raise."""
    monkeypatch.setattr(envs_mod, "LUMILAKE_REMOTE_OPTIMIZER_URL", kwargs["env_url"])
    constructor_kwargs: dict = {"optimizer_type": "halo-greedy"}
    if "base_url" in kwargs:
        constructor_kwargs["base_url"] = kwargs["base_url"]
    # Must not raise ValueError — the URL guard accepts this URL.
    RemoteOptimizer(**constructor_kwargs)
