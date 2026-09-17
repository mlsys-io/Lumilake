"""Shared constants and helpers for redacting credentials from runtime
payloads that may cross process or service boundaries (optimizer dispatch,
archive writes, template hashing, and error bodies echoed back by remote
services such as FlowMesh).
"""

import json
import re
from collections.abc import Mapping
from typing import Any

REDACTED_TOKEN_PLACEHOLDER = "***REDACTED***"

SENSITIVE_DATA_SPEC_KEYS: frozenset[str] = frozenset(
    {"lumid_data_token", "Authorization"}
)

_BEARER_TOKEN_RE = re.compile(r"Bearer\s+\S+")
_AUTH_HEADER_JSON_RE = re.compile(r'("Authorization"\s*:\s*)"[^"]*"')


def _redact_text(text: str) -> str:
    """Regex-scrub a string that may embed a credential (a non-JSON error
    body or already-serialized text)."""
    scrubbed = _BEARER_TOKEN_RE.sub(f"Bearer {REDACTED_TOKEN_PLACEHOLDER}", text)
    scrubbed = _AUTH_HEADER_JSON_RE.sub(rf'\1"{REDACTED_TOKEN_PLACEHOLDER}"', scrubbed)
    return scrubbed


def redact_sensitive(value: Any) -> Any:
    """Recursively replace sensitive keys (tokens, Authorization headers)
    with a placeholder, and regex-scrub bearer tokens embedded in a string
    value under an unrecognized key (e.g. an untrusted endpoint's body)."""
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
    if isinstance(value, str):
        return _redact_text(value)
    return value


def redact_secrets_in_text(value: Any) -> str:
    """Redact credentials from a value that will be logged or persisted as
    text: structured values are redacted key-by-key and re-serialized; plain
    strings fall back to a regex scrub for embedded bearer tokens."""
    if isinstance(value, (Mapping, list)):
        return json.dumps(redact_sensitive(value), default=str)
    text = value if isinstance(value, str) else str(value)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return _redact_text(text)
    if isinstance(parsed, (Mapping, list)):
        return json.dumps(redact_sensitive(parsed), default=str)
    return _redact_text(text)
