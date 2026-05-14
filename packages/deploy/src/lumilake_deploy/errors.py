"""Shared exception types for the deploy orchestration."""


class DeployError(RuntimeError):
    """Raised when the deploy orchestration can't proceed."""
