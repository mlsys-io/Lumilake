"""Unit tests for lumid_data_client blob and catalog methods.

Tests mock the underlying HTTP layer; no network calls are made.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import lumilake_server.utils.lumid_data_client as m
from lumilake_server.utils.lumid_data_client import (
    BlobNotFound,
    _validate_sql_identifier,
    acatalog_column_exists,
    alist_blob_keys,
    get_blob,
    list_blobs,
    put_blob,
)


def _make_sync_response(
    status_code: int, content: bytes = b"", headers: dict | None = None
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = RuntimeError(f"HTTP {status_code}")
    return resp


def _base_url_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch envs to supply a stable URL and token."""
    monkeypatch.setattr(m.envs, "LUMID_DATA_URL", "http://lumid-data")
    monkeypatch.setattr(m.envs, "LUMID_DATA_TOKEN", "test-token")


class TestPutBlob:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        resp = _make_sync_response(200)
        with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
            mock_req.put.return_value = resp
            put_blob("archive/job1/record.json", b'{"ok":1}', "application/json")
        mock_req.put.assert_called_once()
        call_kwargs = mock_req.put.call_args
        assert "http://lumid-data/blobs/archive/job1/record.json" in call_kwargs.args

    def test_413_raises_runtime_error_naming_quota(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_url_and_token(monkeypatch)
        resp = _make_sync_response(413)
        resp.raise_for_status = MagicMock()
        with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
            mock_req.put.return_value = resp
            with pytest.raises(RuntimeError, match="blob_max_bytes"):
                put_blob("big/file.bin", b"x" * 1000, "application/octet-stream")

    def test_url_encodes_special_chars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Spaces, ``#``, ``?`` in keys are URL-encoded; ``/`` is preserved."""
        _base_url_and_token(monkeypatch)
        resp = _make_sync_response(200)
        with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
            mock_req.put.return_value = resp
            put_blob("dir a/b#c?d.txt", b"x", "text/plain")
        url = mock_req.put.call_args.args[0]
        assert url == "http://lumid-data/blobs/dir%20a/b%23c%3Fd.txt"

    def test_allow_redirects_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        resp = _make_sync_response(200)
        with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
            mock_req.put.return_value = resp
            put_blob("k.txt", b"x", "text/plain")
        assert mock_req.put.call_args.kwargs.get("allow_redirects") is False


class TestGetBlob:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        resp = _make_sync_response(
            200,
            content=b"hello bytes",
            headers={"Content-Type": "text/plain"},
        )
        with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
            mock_req.get.return_value = resp
            body, ct = get_blob("some/key.txt")
        assert body == b"hello bytes"
        assert ct == "text/plain"

    def test_404_raises_blob_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        resp = _make_sync_response(404)
        resp.raise_for_status = MagicMock()
        with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
            mock_req.get.return_value = resp
            with pytest.raises(BlobNotFound):
                get_blob("missing/key.json")

    def test_allow_redirects_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        resp = _make_sync_response(
            200, content=b"x", headers={"Content-Type": "text/plain"}
        )
        with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
            mock_req.get.return_value = resp
            get_blob("k.txt")
        assert mock_req.get.call_args.kwargs.get("allow_redirects") is False


class TestAlistBlobKeys:
    @pytest.mark.asyncio
    async def test_single_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        page = {
            "objects": [
                {"key": "pfx/a.txt"},
                {"key": "pfx/b.txt"},
            ],
            "truncated": False,
        }
        with patch(
            "lumilake_server.utils.lumid_data_client.aget_json",
            new=AsyncMock(return_value=page),
        ):
            keys = await alist_blob_keys(prefix="pfx/", recursive=True)
        assert keys == ["pfx/a.txt", "pfx/b.txt"]

    @pytest.mark.asyncio
    async def test_sends_limit_10000_and_returns_all_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_url_and_token(monkeypatch)
        page = {
            "objects": [
                {"key": "pfx/a.txt"},
                {"key": "pfx/b.txt"},
                {"key": "pfx/c.txt"},
            ],
            "truncated": False,
        }
        captured_params: dict[str, Any] = {}

        async def _fake_aget_json(url: str, **kwargs: Any) -> dict:
            captured_params.update(kwargs.get("params", {}))
            return page

        with patch(
            "lumilake_server.utils.lumid_data_client.aget_json", new=_fake_aget_json
        ):
            keys = await alist_blob_keys(prefix="pfx/", recursive=True)
        assert captured_params.get("limit") == "10000"
        assert keys == ["pfx/a.txt", "pfx/b.txt", "pfx/c.txt"]

    @pytest.mark.asyncio
    async def test_truncated_response_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_url_and_token(monkeypatch)
        page = {
            "objects": [{"key": f"pfx/{i}.txt"} for i in range(10000)],
            "truncated": True,
        }
        with patch(
            "lumilake_server.utils.lumid_data_client.aget_json",
            new=AsyncMock(return_value=page),
        ):
            with pytest.raises(RuntimeError, match="10000-key server cap"):
                await alist_blob_keys(prefix="pfx/", recursive=True)


class TestListBlobs:
    @pytest.mark.asyncio
    async def test_sends_limit_10000(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        captured_params: dict[str, Any] = {}

        async def _fake_aget_json(url: str, **kwargs: Any) -> dict:
            captured_params.update(kwargs.get("params", {}))
            return {"objects": [], "truncated": False}

        with patch(
            "lumilake_server.utils.lumid_data_client.aget_json", new=_fake_aget_json
        ):
            await list_blobs("pfx/")
        assert captured_params.get("limit") == "10000"

    @pytest.mark.asyncio
    async def test_truncated_response_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_url_and_token(monkeypatch)
        page = {
            "objects": [{"key": f"pfx/{i}.txt", "size": 1} for i in range(10000)],
            "truncated": True,
        }
        with patch(
            "lumilake_server.utils.lumid_data_client.aget_json",
            new=AsyncMock(return_value=page),
        ):
            with pytest.raises(RuntimeError, match="10000-key server cap"):
                await list_blobs("pfx/")


class TestValidateSqlIdentifier:
    @pytest.mark.parametrize("name", ["1abc", "pg-cat", "", "drop;--", "a b"])
    def test_rejects_invalid_identifiers(self, name: str) -> None:
        with pytest.raises(ValueError, match="invalid"):
            _validate_sql_identifier(name, "schema")

    @pytest.mark.parametrize("name", ["a", "A", "_x", "abc_123", "PublicSchema"])
    def test_accepts_valid_identifiers(self, name: str) -> None:
        _validate_sql_identifier(name, "schema")


class TestAcatalogColumnExists:
    @pytest.mark.asyncio
    async def test_column_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        fake_resp = MagicMock()
        fake_resp.status = 200
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json = AsyncMock(
            return_value={"columns": [{"name": "user_id"}, {"name": "email"}]}
        )
        with patch(
            "lumilake_server.utils.lumid_data_client.arequest",
            new=AsyncMock(return_value=fake_resp),
        ):
            result = await acatalog_column_exists("public", "users", "email")
        assert result is True

    @pytest.mark.asyncio
    async def test_column_not_in_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        fake_resp = MagicMock()
        fake_resp.status = 200
        fake_resp.raise_for_status = MagicMock()
        fake_resp.json = AsyncMock(return_value={"columns": [{"name": "user_id"}]})
        with patch(
            "lumilake_server.utils.lumid_data_client.arequest",
            new=AsyncMock(return_value=fake_resp),
        ):
            result = await acatalog_column_exists("public", "users", "nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_404_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _base_url_and_token(monkeypatch)
        fake_resp = MagicMock()
        fake_resp.status = 404
        fake_resp.raise_for_status = MagicMock()
        with patch(
            "lumilake_server.utils.lumid_data_client.arequest",
            new=AsyncMock(return_value=fake_resp),
        ):
            result = await acatalog_column_exists("public", "missing_table", "col")
        assert result is False


class TestRequestIdForwarding:
    def test_put_blob_forwards_x_request_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_url_and_token(monkeypatch)
        token = m.trace_id_var.set("trace-abc-123")
        try:
            resp = _make_sync_response(200)
            with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
                mock_req.put.return_value = resp
                put_blob("k.txt", b"x", "text/plain")
            headers = mock_req.put.call_args.kwargs.get("headers", {})
            assert headers.get("X-Request-ID") == "trace-abc-123"
            assert headers.get("Authorization") == "Bearer test-token"
        finally:
            m.trace_id_var.reset(token)

    def test_put_blob_omits_x_request_id_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_url_and_token(monkeypatch)
        resp = _make_sync_response(200)
        with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
            mock_req.put.return_value = resp
            put_blob("k.txt", b"x", "text/plain")
        headers = mock_req.put.call_args.kwargs.get("headers", {})
        assert "X-Request-ID" not in headers

    def test_get_blob_forwards_x_request_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_url_and_token(monkeypatch)
        token = m.trace_id_var.set("trace-get-1")
        try:
            resp = _make_sync_response(
                200, content=b"x", headers={"Content-Type": "text/plain"}
            )
            with patch("lumilake_server.utils.lumid_data_client.requests") as mock_req:
                mock_req.get.return_value = resp
                get_blob("k.txt")
            headers = mock_req.get.call_args.kwargs.get("headers", {})
            assert headers.get("X-Request-ID") == "trace-get-1"
        finally:
            m.trace_id_var.reset(token)

    @pytest.mark.asyncio
    async def test_acatalog_forwards_x_request_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _base_url_and_token(monkeypatch)
        token = m.trace_id_var.set("trace-cat-2")
        captured: dict[str, Any] = {}

        async def _fake_arequest(method: str, url: str, **kwargs: Any) -> MagicMock:
            captured.update(kwargs.get("headers", {}))
            fake_resp = MagicMock()
            fake_resp.status = 404
            fake_resp.raise_for_status = MagicMock()
            return fake_resp

        try:
            with patch(
                "lumilake_server.utils.lumid_data_client.arequest", new=_fake_arequest
            ):
                await acatalog_column_exists("public", "x", "y")
        finally:
            m.trace_id_var.reset(token)
        assert captured.get("X-Request-ID") == "trace-cat-2"
