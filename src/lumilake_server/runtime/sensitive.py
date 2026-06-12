"""Shared constants and helpers for redacting credentials from runtime
payloads that may cross process or service boundaries (optimizer dispatch,
archive writes, template hashing).
"""

from collections.abc import Mapping
from typing import Any

REDACTED_TOKEN_PLACEHOLDER = "***REDACTED***"

SENSITIVE_DATA_SPEC_KEYS: frozenset[str] = frozenset({"lumid_data_token"})


def redact_sensitive(value: Any) -> Any:
    """Recursively replace sensitive ``data_spec`` keys with a placeholder."""
    if isinstance(value, Mapping):
        return {
            key: (
                REDACTED_TOKEN_PLACEHOLDER
                if key in SENSITIVE_DATA_SPEC_KEYS and isinstance(sub, str) and sub
                else redact_sensitive(sub)
            )
            for key, sub in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value
