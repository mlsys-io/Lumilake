"""Workflow trace resource — fetch raw rows or run the analyzer."""

import json
from collections.abc import AsyncIterator, Iterator
from enum import StrEnum

import httpx

from .._base_client import (
    _make_url,
    _raise_for_stream_status,
    _raise_for_stream_status_async,
)
from ..exceptions import FlowMeshConnectionError
from ..models.traces import ProfileSummary
from ._base import AsyncResource, SyncResource


class TraceType(StrEnum):
    """Trace row type. Members serialize as their values."""

    SPANS = "spans"
    ASSETS = "assets"
    LINEAGE = "lineage"


class Traces(SyncResource):
    """Synchronous workflow trace operations."""

    def fetch(self, workflow_id: str, trace_type: TraceType) -> Iterator[dict]:
        """Yield JSONL rows for `spans`, `assets`, or `lineage`."""
        url = _make_url(
            self._client.base_url, f"/traces/workflows/{workflow_id}/{trace_type}"
        )
        try:
            with self._client._http.stream("GET", url) as response:
                _raise_for_stream_status(response, "GET")
                for line in response.iter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    def analyze(self, workflow_id: str) -> ProfileSummary:
        """Run the trace analyzer and return a parsed `ProfileSummary`."""
        return ProfileSummary.model_validate(
            self._client._request("GET", f"/traces/workflows/analyze/{workflow_id}")
        )


class AsyncTraces(AsyncResource):
    """Asynchronous workflow trace operations."""

    async def fetch(
        self, workflow_id: str, trace_type: TraceType
    ) -> AsyncIterator[dict]:
        url = _make_url(
            self._client.base_url, f"/traces/workflows/{workflow_id}/{trace_type}"
        )
        try:
            async with self._client._http.stream("GET", url) as response:
                await _raise_for_stream_status_async(response, "GET")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError as exc:
            raise FlowMeshConnectionError(f"Failed to connect to {url}: {exc}")

    async def analyze(self, workflow_id: str) -> ProfileSummary:
        return ProfileSummary.model_validate(
            await self._client._request(
                "GET", f"/traces/workflows/analyze/{workflow_id}"
            )
        )
