from dataclasses import dataclass, field
from typing import Any

from lumilake_server.ops.data_ops import DataOp
from lumilake_server.ops.ops import FunctionalOp, Op


@dataclass
class HttpFetchConfig:
    """Bounded HTTP fetch settings for ``HttpFetchOp``."""

    headers: dict[str, str] = field(default_factory=dict)
    timeout_sec: float = 30.0
    retries: int = 2
    max_concurrency: int = 8
    fail_open: bool = False
    request_delay_sec: float = 0.0
    cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.timeout_sec <= 0:
            raise ValueError("HttpFetchOp timeout_sec must be greater than zero")
        if self.retries < 0:
            raise ValueError("HttpFetchOp retries must be non-negative")
        if self.request_delay_sec < 0:
            raise ValueError("HttpFetchOp request_delay_sec must be non-negative")
        if not 1 <= self.max_concurrency <= 8:
            raise ValueError("HttpFetchOp max_concurrency must be between 1 and 8")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.headers.items()
        ):
            raise ValueError("HttpFetchOp headers must be a string-to-string mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "headers": dict(self.headers),
            "timeout_sec": self.timeout_sec,
            "retries": self.retries,
            "max_concurrency": self.max_concurrency,
            "fail_open": self.fail_open,
            "request_delay_sec": self.request_delay_sec,
            "cache_dir": self.cache_dir,
        }


@Op.registry.register("HttpFetchOp")
class HttpFetchOp(FunctionalOp):
    """Fetch JSON over HTTP once per input row.

    ``url`` and the optional ``body`` are Python format templates rendered with
    each row's value as ``{query}``. The executor issues one FlowMesh API
    request per input row and returns one result row for each request.
    """

    def __init__(
        self,
        url: str,
        query: str | list[str] | Op,
        config: HttpFetchConfig | None = None,
        method: str = "GET",
        body: dict[str, Any] | None = None,
    ) -> None:
        self.url = url.strip()
        self.method = method.upper()
        self.body = body
        self.config = config or HttpFetchConfig()
        if not self.url:
            raise ValueError("HttpFetchOp requires a non-empty url template")
        if self.method not in {"GET", "POST"}:
            raise ValueError("HttpFetchOp method must be GET or POST")
        if "{query}" not in self.url:
            raise ValueError("HttpFetchOp url must contain a {query} placeholder")
        if self.method == "POST" and self.body is None:
            raise ValueError("HttpFetchOp POST requires a body template")
        query_op = (
            query if isinstance(query, Op) else DataOp(data=_as_query_list(query))
        )
        super().__init__([query_op])

    @property
    def query(self) -> Op:
        return self.inputs[0]

    def _serialize(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": self.url,
            "method": self.method,
            "query": self.query.id,
            "config": self.config.to_dict(),
        }
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
            body=data.get("body"),
        )


def _as_query_list(query: str | list[str]) -> list[str]:
    values = [query] if isinstance(query, str) else list(query)
    if not values or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError("HttpFetchOp query input must contain non-empty strings")
    return values


def http_fetch(
    url: str,
    query: str | list[str] | Op,
    config: HttpFetchConfig | None = None,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> HttpFetchOp:
    return HttpFetchOp(url, query, config, method, body)
