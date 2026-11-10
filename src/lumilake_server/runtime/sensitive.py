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

_BEARER_TOKEN_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_AUTH_HEADER_RE = re.compile(
    r"(Authorization\s*:\s*[A-Za-z][A-Za-z0-9_-]*\s+)[^\n]*", re.IGNORECASE
)
_AUTH_HEADER_JSON_RE = re.compile(r'("Authorization"\s*:\s*)"[^"]*"', re.IGNORECASE)

_REQUEST_HEADER_PARENT_KEYS: frozenset[str] = frozenset({"api", "api_spec", "config"})


def _is_sensitive_key(key: Any) -> bool:
    return isinstance(key, str) and key.lower() in {
        k.lower() for k in SENSITIVE_DATA_SPEC_KEYS
    }


def _redact_header_mapping(value: Mapping) -> dict:
    """Replace every non-empty string header value with the placeholder: a
    request header mapping routinely carries the credential itself
    (``x-api-key``, ``Ocp-Apim-Subscription-Key``), under a name no key list
    can enumerate."""
    return {
        key: (
            REDACTED_TOKEN_PLACEHOLDER
            if isinstance(sub, str) and sub
            else redact_sensitive(sub)
        )
        for key, sub in value.items()
    }


def _redact_text(text: str) -> str:
    """Regex-scrub a string that may embed a credential (a non-JSON error
    body or already-serialized text)."""
    scrubbed = _BEARER_TOKEN_RE.sub(f"Bearer {REDACTED_TOKEN_PLACEHOLDER}", text)
    scrubbed = _AUTH_HEADER_RE.sub(rf"\1{REDACTED_TOKEN_PLACEHOLDER}", scrubbed)
    scrubbed = _AUTH_HEADER_JSON_RE.sub(rf'\1"{REDACTED_TOKEN_PLACEHOLDER}"', scrubbed)
    return scrubbed


def redact_sensitive(value: Any) -> Any:
    """Recursively replace sensitive keys (tokens, Authorization headers)
    with a placeholder, redact every value of a request header mapping
    nested under an ``api`` or ``config`` key, and regex-scrub bearer tokens
    embedded in a string value under an unrecognized key (e.g. an untrusted
    endpoint's body)."""
    return _redact(value, parent_key=None)


def _redact(value: Any, *, parent_key: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, sub in value.items():
            if isinstance(sub, str) and sub and _is_sensitive_key(key):
                redacted[key] = REDACTED_TOKEN_PLACEHOLDER
            elif (
                key == "headers"
                and isinstance(sub, Mapping)
                and parent_key in _REQUEST_HEADER_PARENT_KEYS
            ):
                redacted[key] = _redact_header_mapping(sub)
            else:
                redacted[key] = _redact(sub, parent_key=key)
        return redacted
    if isinstance(value, list):
        return [_redact(item, parent_key=parent_key) for item in value]
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
