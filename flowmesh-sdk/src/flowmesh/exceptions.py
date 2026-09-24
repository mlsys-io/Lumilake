"""FlowMesh SDK exception hierarchy."""

from typing import Any


class FlowMeshError(Exception):
    """Base exception for all FlowMesh SDK errors."""


class APIError(FlowMeshError):
    """Raised when the FlowMesh API returns an error response (status >= 400)."""

    def __init__(
        self,
        message: str,
        status_code: int,
        method: str,
        url: str,
        body: Any = None,
    ) -> None:
        self.status_code = status_code
        self.method = method
        self.url = url
        self.body = body
        super().__init__(message)

    def __str__(self) -> str:
        return (
            f"{self.method} {self.url} returned {self.status_code}: "
            f"{super().__str__()}"
        )


class AuthenticationError(APIError):
    """Raised on 401/403 responses."""


class NotFoundError(APIError):
    """Raised on 404 responses."""


class ValidationError(APIError):
    """Raised on 400/422 responses."""


class FlowMeshConnectionError(FlowMeshError):
    """Raised when the SDK cannot connect to the FlowMesh server."""


class ConfigNotFoundError(FlowMeshError):
    """Raised when the SDK cannot find a configuration file."""


class ConfigInvalidError(FlowMeshError):
    """Raised when the SDK encounters invalid configurations."""
