from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from lumilake_server.common import Message


class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LumilakeRequestConfig(BaseModel):
    """
    Configuration object associated with a Lumilake request.
    """

    priority: Priority = Priority.MEDIUM
    """Scheduling priority for the request"""
    user_id: str = Field(min_length=1)
    """User identifier used for fair scheduling across tenants"""
    org_id: str = Field(default="default", min_length=1)
    """Organization identifier for org-scoped server-side operations"""
    principal_id: str = Field(min_length=1)
    """Principal identifier; runtime dispatch partitions by this so a single
    FlowMesh submission never spans multiple principals."""
    optimizer_type: str | None = None
    """Optimizer to use for this request. Must be a name in
    ``OPTIMIZER_TYPES`` or advertised by a loaded ``OptimizerProvider``;
    ``None`` falls back to ``LUMILAKE_DEFAULT_OPTIMIZER``."""


class LumilakeRequest(BaseModel):
    """
    Request object to be processed by the Lumilake server.
    """

    request_id: str
    query: dict[str, Any]
    """Query for the request that contains one or more compute graphs"""
    config: LumilakeRequestConfig | None = None
    """Configuration for the request"""


PrefixKey = tuple[str, str, str, str]  # (llm_service, model, base_url, api_key)
PrefixMap = dict[PrefixKey, list[str | list[Message]]]


class LumilakeResponse(BaseModel):
    """
    Response object returned by the Lumilake server.
    """

    outputs: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict,
        description="Outputs grouped by graph name and output name.",
    )
    """Outputs of the request"""
    static_prefix_map: PrefixMap | None = Field(
        default=None, description="Static prefix map for cached prefixes."
    )
    """Mapping from prefix key to static prefixes used in the request"""
    dynamic_prefix_map: dict[int, PrefixMap] | None = Field(
        default=None, description="Dynamic prefix map by query index."
    )
    """Mapping from query index to prefix key to dynamic prefixes used in the request"""
    error_info: list[dict[str, Any]] | None = Field(
        default=None, description="Error details encountered during processing."
    )
    """Information of errors encountered during processing the request"""
    chat_histories: dict[str, Any] | None = Field(
        default=None, description="Chat histories metadata for output nodes."
    )
    """Chat histories metadata for output nodes (history tracking)"""


class RequestCancelledError(RuntimeError):
    """Raised when a request is cancelled before completion."""
