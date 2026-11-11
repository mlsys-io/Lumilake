from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from lumilake_server.ops.data_ops import DataOp
from lumilake_server.ops.ops import FunctionalOp, Op

MAX_RETRIES = 10


@dataclass
class HttpFetchConfig:
    """Bounded HTTP fetch settings for ``HttpFetchOp``."""

    headers: dict[str, str] = field(default_factory=dict)
    timeout_sec: float = 30.0
    retries: int = 2
    max_concurrency: int = 8
    fail_open: bool = False
    request_delay_sec: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.headers.items()
        ):
            raise ValueError("HttpFetchOp headers must be a string-to-string mapping")
        if not isinstance(self.timeout_sec, int | float) or self.timeout_sec <= 0:
            raise ValueError("HttpFetchOp timeout_sec must be greater than zero")
        if not isinstance(self.retries, int) or not 0 <= self.retries <= MAX_RETRIES:
            raise ValueError(f"HttpFetchOp retries must be between 0 and {MAX_RETRIES}")
        if (
            not isinstance(self.request_delay_sec, int | float)
            or self.request_delay_sec < 0
        ):
            raise ValueError("HttpFetchOp request_delay_sec must be non-negative")
        if (
            not isinstance(self.max_concurrency, int)
            or not 1 <= self.max_concurrency <= 8
        ):
            raise ValueError("HttpFetchOp max_concurrency must be between 1 and 8")
        if not isinstance(self.fail_open, bool):
            raise ValueError("HttpFetchOp fail_open must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "headers": dict(self.headers),
            "timeout_sec": self.timeout_sec,
            "retries": self.retries,
            "max_concurrency": self.max_concurrency,
            "fail_open": self.fail_open,
            "request_delay_sec": self.request_delay_sec,
        }


@Op.registry.register("HttpFetchOp")
class HttpFetchOp(FunctionalOp):
    """Fetch JSON over HTTP once per input row.

    ``url``, optional ``params``, and optional ``body`` are Python format
    templates rendered with each row's value as ``{query}``. The executor
    issues one FlowMesh API request per input row and returns one result row
    for each request.
    """

    def __init__(
        self,
        url: str,
        query: str | list[str] | Op,
        config: HttpFetchConfig | None = None,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> None:
        self.url = url.strip()
        self.method = method.upper()
        self.params = params
        self.body = body
        self.config = config or HttpFetchConfig()
        validate_fetch_request(self.url, self.method, self.params, self.body)
        query_op = (
            query if isinstance(query, Op) else DataOp(data=_as_query_list(query))
        )
        super().__init__([query_op])

    @property
    def query(self) -> Op:
        return self.inputs[0]

    def consumer_result_path(self) -> str:
        return "items.json"

    def consumer_row_count(self, input_counts: Sequence[int | None]) -> int | None:
        return input_counts[0] if input_counts else None

    def _serialize(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": self.url,
            "method": self.method,
            "query": self.query.id,
            "config": self.config.to_dict(),
        }
        if self.params is not None:
            payload["params"] = self.params
        if self.body is not None:
            payload["body"] = self.body
        return payload

    @classmethod
    def _from_json(
        cls, data: dict[str, Any], other_ops: dict[str, Op]
    ) -> "HttpFetchOp":
        return cls(
            url=data["url"],
            query=other_ops[data["query"]],
            config=HttpFetchConfig(**data["config"]),
            method=data["method"],
            params=data.get("params"),
            body=data.get("body"),
        )


def _as_query_list(query: str | list[str]) -> list[str]:
    values = [query] if isinstance(query, str) else list(query)
    if not values or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError("HttpFetchOp query input must contain non-empty strings")
    return values


def validate_fetch_request(
    url: str,
    method: str,
    params: Any,
    body: Any,
    *,
    where: str = "HttpFetchOp",
) -> None:
    """Check an HTTP fetch's request template, naming the op in ``where``.

    The parser and the constructor both call this, so a YAML job and a
    constructed op reject the same requests with the same message.
    """
    if not url:
        raise ValueError(f"{where} requires a non-empty url template")
    if method not in {"GET", "POST"}:
        raise ValueError(f"{where} method must be GET or POST")
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"{where} params must be a mapping")
    if not any(contains_query_placeholder(value) for value in (url, params, body)):
        raise ValueError(f"{where} requires '{{query}}' in url, params, or body")
    if method == "POST" and body is None:
        raise ValueError(f"{where} POST requires a body")


def contains_query_placeholder(value: Any) -> bool:
    """Whether a URL, params, or body template contains ``{query}``."""
    if isinstance(value, str):
        return "{query}" in value
    if isinstance(value, dict):
        return any(contains_query_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_query_placeholder(item) for item in value)
    return False


def http_fetch(
    url: str,
    query: str | list[str] | Op,
    config: HttpFetchConfig | None = None,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> HttpFetchOp:
    return HttpFetchOp(url, query, config, method, params, body)
