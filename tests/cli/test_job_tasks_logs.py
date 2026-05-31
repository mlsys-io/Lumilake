"""CLI coverage for ``lumilake job tasks`` and ``lumilake job logs``."""

import inspect
import json
from typing import Any

import pytest
from lumilake_cli.commands import job as job_cmd
from lumilake_cli.commands.job import app
from lumilake_cli.core.http import HttpClient
from typer.testing import CliRunner


class _StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubClient:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self._pages = pages
        self.calls: list[tuple[str, Any]] = []

    def get(self, path: str, **kwargs: Any) -> _StubResponse:
        self.calls.append((path, kwargs.get("params")))
        index = min(len(self.calls) - 1, len(self._pages) - 1)
        return _StubResponse(self._pages[index])


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_tasks_renders_table(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _StubClient(
        [
            {
                "ok": True,
                "data": {
                    "job_id": "j-1",
                    "tasks": [
                        {
                            "task_id": "t-a",
                            "status": "SUCCEEDED",
                            "graph_node_name": "data_prep",
                            "assigned_worker": "w-1",
                            "workflow_id": "wf-1",
                        }
                    ],
                },
            }
        ]
    )
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)

    result = runner.invoke(app, ["tasks", "j-1"])
    assert result.exit_code == 0, result.stdout
    assert client.calls[0][0] == "/jobs/j-1/tasks"
    assert "t-a" in result.stdout
    assert "SUCCEEDED" in result.stdout
    assert "data_prep" in result.stdout


def test_tasks_emits_json_when_requested(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"job_id": "j-1", "tasks": [{"task_id": "t-a"}]}
    client = _StubClient([{"ok": True, "data": payload}])
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)

    result = runner.invoke(app, ["tasks", "j-1", "--json"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout.strip()) == payload


def test_logs_oneshot_prints_entries(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _StubClient(
        [
            {
                "ok": True,
                "data": {
                    "job_id": "j-1",
                    "task_id": "t-a",
                    "entries": [
                        {
                            "cursor": "c1",
                            "event": {
                                "ts": "2026-05-31T00:00:00Z",
                                "level": "INFO",
                                "stream": "stdout",
                                "message": "hello world",
                            },
                        }
                    ],
                    "next_cursor": "c1",
                    "prev_cursor": None,
                },
            }
        ]
    )
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)

    result = runner.invoke(app, ["logs", "j-1", "t-a", "--limit", "50"])
    assert result.exit_code == 0, result.stdout
    assert "hello world" in result.stdout
    assert "INFO" in result.stdout
    params = dict(client.calls[0][1] or [])
    assert params["limit"] == "50"


def test_logs_json_mode_emits_one_object_per_line(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = {
        "cursor": "c1",
        "event": {"message": "hi", "level": "INFO"},
    }
    client = _StubClient(
        [
            {
                "ok": True,
                "data": {
                    "job_id": "j-1",
                    "task_id": "t-a",
                    "entries": [entry],
                    "next_cursor": None,
                    "prev_cursor": None,
                },
            }
        ]
    )
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)
    result = runner.invoke(app, ["logs", "j-1", "t-a", "--json"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout.strip()) == entry


def test_logs_follow_advances_cursor_then_stops_on_interrupt(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = [
        {
            "ok": True,
            "data": {
                "job_id": "j-1",
                "task_id": "t-a",
                "entries": [
                    {"cursor": "c1", "event": {"message": "first"}},
                ],
                "next_cursor": "c1",
                "prev_cursor": None,
            },
        },
        {
            "ok": True,
            "data": {
                "job_id": "j-1",
                "task_id": "t-a",
                "entries": [
                    {"cursor": "c2", "event": {"message": "second"}},
                ],
                "next_cursor": "c2",
                "prev_cursor": None,
            },
        },
    ]
    client = _StubClient(pages)
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)

    # Make sleep raise KeyboardInterrupt after the second poll so the loop
    # exercises cursor advancement exactly once, then exits cleanly.
    call_count = {"n": 0}

    def _sleep(_seconds: float) -> None:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(job_cmd.time, "sleep", _sleep)

    result = runner.invoke(app, ["logs", "j-1", "t-a", "--follow", "--interval", "0"])
    assert result.exit_code == 0, result.stdout
    assert "first" in result.stdout
    assert "second" in result.stdout
    # First call has no ``after``; second call uses cursor advanced from page 1.
    params_first = dict(client.calls[0][1] or [])
    params_second = dict(client.calls[1][1] or [])
    assert "after" not in params_first
    assert params_second.get("after") == "c1"


def test_logs_rejects_out_of_range_limit(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        job_cmd, "client_from_config", lambda: _StubClient([{"data": {}}])
    )
    result = runner.invoke(app, ["logs", "j-1", "t-a", "--limit", "0"])
    assert result.exit_code == 1


def test_http_client_get_signature_accepts_params() -> None:
    sig = inspect.signature(HttpClient.get)
    # ``params`` is forwarded via **kwargs to requests; verify the method
    # declares a VAR_KEYWORD parameter so callers can pass ``params=...``.
    var_keyword_params = [
        p for p in sig.parameters.values() if p.kind is inspect.Parameter.VAR_KEYWORD
    ]
    assert (
        var_keyword_params
    ), "HttpClient.get must accept **kwargs so that params= can be forwarded"


def test_logs_propagates_http_error(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_get(_path: str, **_kwargs: Any) -> Any:
        raise job_cmd.HttpError("boom")

    class _Boom:
        get = staticmethod(_raise_get)

    monkeypatch.setattr(job_cmd, "client_from_config", lambda: _Boom())
    result = runner.invoke(app, ["logs", "j-1", "t-a"])
    assert result.exit_code == 1
