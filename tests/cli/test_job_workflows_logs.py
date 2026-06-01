"""CLI coverage for ``lumilake job workflows`` and ``lumilake job logs``."""

import inspect
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
import requests as _requests_mod
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
        self.base_url = "http://test"
        self.api_key = None
        self.timeout = 10.0

    def get(self, path: str, **kwargs: Any) -> _StubResponse:
        self.calls.append((path, kwargs.get("params")))
        index = min(len(self.calls) - 1, len(self._pages) - 1)
        return _StubResponse(self._pages[index])

    def download(self, path: str, output_path: Path, **kwargs: Any) -> None:
        self.calls.append((path, None))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Write an empty tar by default; tests that need content override this.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w"):
            pass
        output_path.write_bytes(buf.getvalue())


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_workflows_renders_table(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _StubClient(
        [
            {
                "ok": True,
                "data": {
                    "job_id": "j-1",
                    "workflows": [
                        {
                            "workflow_id": "wf-1",
                            "status": "COMPLETED",
                            "submitted_at": "2026-05-31T00:00:00Z",
                            "task_count": 5,
                            "succeeded_count": 4,
                            "failed_count": 1,
                        }
                    ],
                },
            }
        ]
    )
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)

    result = runner.invoke(app, ["workflows", "j-1"])
    assert result.exit_code == 0, result.stdout
    assert client.calls[0][0] == "/jobs/j-1/workflows"
    assert "wf-1" in result.stdout
    assert "COMPLETED" in result.stdout


def test_workflows_emits_json_when_requested(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "job_id": "j-1",
        "workflows": [{"workflow_id": "wf-1", "status": "COMPLETED"}],
    }
    client = _StubClient([{"ok": True, "data": payload}])
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)

    result = runner.invoke(app, ["workflows", "j-1", "--json"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout.strip()) == payload


def test_logs_show_prints_entries(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _StubClient(
        [
            {
                "ok": True,
                "data": {
                    "job_id": "j-1",
                    "workflow_id": "wf-1",
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

    result = runner.invoke(app, ["logs", "show", "j-1", "wf-1", "--limit", "50"])
    assert result.exit_code == 0, result.stdout
    assert "hello world" in result.stdout
    assert "INFO" in result.stdout
    params = dict(client.calls[0][1] or [])
    assert params["limit"] == "50"


def test_logs_show_json_mode_emits_one_object_per_line(
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
                    "workflow_id": "wf-1",
                    "entries": [entry],
                    "next_cursor": None,
                    "prev_cursor": None,
                },
            }
        ]
    )
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)
    result = runner.invoke(app, ["logs", "show", "j-1", "wf-1", "--json"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout.strip()) == entry


def test_logs_show_rejects_out_of_range_limit(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        job_cmd, "client_from_config", lambda: _StubClient([{"data": {}}])
    )
    result = runner.invoke(app, ["logs", "show", "j-1", "wf-1", "--limit", "0"])
    assert result.exit_code == 1


def test_logs_show_propagates_http_error(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_get(_path: str, **_kwargs: Any) -> Any:
        raise job_cmd.HttpError("boom")

    class _Boom:
        base_url = "http://test"
        api_key = None
        timeout = 10.0
        get = staticmethod(_raise_get)

    monkeypatch.setattr(job_cmd, "client_from_config", lambda: _Boom())
    result = runner.invoke(app, ["logs", "show", "j-1", "wf-1"])
    assert result.exit_code == 1


def test_logs_stream_streams_sse_entries(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry1 = {"cursor": "c1", "event": {"message": "streamed-first"}}
    entry2 = {"cursor": "c2", "event": {"message": "streamed-second"}}
    sse_body = f"data: {json.dumps(entry1)}\n\ndata: {json.dumps(entry2)}\n\n"

    class _FakeResp:
        status_code = 200

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def iter_content(
            self, chunk_size: Any = None, decode_unicode: bool = False
        ) -> list[str]:
            return [sse_body]

    monkeypatch.setattr(job_cmd, "client_from_config", lambda: _StubClient([{}]))
    monkeypatch.setattr(
        _requests_mod, "get", lambda *a, **kw: _FakeResp(), raising=True
    )

    result = runner.invoke(app, ["logs", "stream", "j-1", "wf-1"])
    assert result.exit_code == 0, result.output
    assert "streamed-first" in result.output
    assert "streamed-second" in result.output


def test_logs_stream_cursor_forwarded(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class _FakeResp:
        status_code = 200

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def iter_content(
            self, chunk_size: Any = None, decode_unicode: bool = False
        ) -> list[str]:
            return []

    def _fake_get(url: str, **kwargs: Any) -> _FakeResp:
        captured["params"] = kwargs.get("params")
        return _FakeResp()

    monkeypatch.setattr(job_cmd, "client_from_config", lambda: _StubClient([{}]))
    monkeypatch.setattr(_requests_mod, "get", _fake_get, raising=True)

    result = runner.invoke(app, ["logs", "stream", "j-1", "wf-1", "--cursor", "abc"])
    assert result.exit_code == 0, result.output
    assert captured.get("params") == [("cursor", "abc")]


def test_logs_download_extracts_files(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b'{"message": "archived"}\n'

    class _TarClient(_StubClient):
        def download(self, path: str, output_path: Path, **kwargs: Any) -> None:
            self.calls.append((path, None))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tf:
                info = tarfile.TarInfo(name="t-1-logs.jsonl")
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))
            output_path.write_bytes(buf.getvalue())

    client = _TarClient([{}])
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)

    out_dir = tmp_path / "logs"
    result = runner.invoke(
        app, ["logs", "download", "j-1", "wf-1", "--output", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    extracted = out_dir / "t-1-logs.jsonl"
    assert extracted.exists()
    assert extracted.read_bytes() == content
    assert str(extracted) in result.output


def test_logs_download_empty_tar_prints_no_logs_message(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _StubClient([{}])
    monkeypatch.setattr(job_cmd, "client_from_config", lambda: client)

    out_dir = tmp_path / "logs"
    result = runner.invoke(
        app, ["logs", "download", "j-1", "wf-1", "--output", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "No logs downloaded for workflow wf-1" in result.output


def test_logs_stream_sse_error_frame_exits_1(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An SSE error event frame must cause the stream command to exit with code 1."""
    error_frame = json.dumps(
        {
            "kind": "stream_error",
            "code": "NotFoundError",
            "message": "FlowMesh log stream ended (not found or expired).",
        }
    )
    sse_body = f"event: error\ndata: {error_frame}\n\n"

    class _FakeResp:
        status_code = 200

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def iter_content(
            self, chunk_size: Any = None, decode_unicode: bool = False
        ) -> list[str]:
            return [sse_body]

    monkeypatch.setattr(job_cmd, "client_from_config", lambda: _StubClient([{}]))
    monkeypatch.setattr(
        _requests_mod, "get", lambda *a, **kw: _FakeResp(), raising=True
    )

    result = runner.invoke(app, ["logs", "stream", "j-1", "wf-1"])
    assert result.exit_code == 1
    assert "NotFoundError" in result.output or "not found or expired" in result.output


def test_http_client_get_signature_accepts_params() -> None:
    sig = inspect.signature(HttpClient.get)
    var_keyword_params = [
        p for p in sig.parameters.values() if p.kind is inspect.Parameter.VAR_KEYWORD
    ]
    assert (
        var_keyword_params
    ), "HttpClient.get must accept **kwargs so that params= can be forwarded"
