import json

from lumilake_server.runtime.sensitive import (
    REDACTED_TOKEN_PLACEHOLDER,
    redact_secrets_in_text,
    redact_sensitive,
)


def test_redact_secrets_in_text_redacts_dict_input() -> None:
    value = {"Authorization": "Bearer sk-live-secret", "other": "keep-me"}

    result = redact_secrets_in_text(value)

    assert "sk-live-secret" not in result
    assert "keep-me" in result
    assert REDACTED_TOKEN_PLACEHOLDER in result


def test_redact_secrets_in_text_redacts_json_encoded_string() -> None:
    value = json.dumps({"Authorization": "Bearer sk-live-secret", "other": "keep-me"})

    result = redact_secrets_in_text(value)

    assert "sk-live-secret" not in result
    assert "keep-me" in result
    assert REDACTED_TOKEN_PLACEHOLDER in result


def test_redact_secrets_in_text_falls_back_to_regex_for_plain_text() -> None:
    """A non-JSON string (e.g. a plain-text HTTP error body) still needs its
    embedded bearer token scrubbed; this is the _redact_text regex path,
    not the structural redact_sensitive path."""
    value = "request rejected: Authorization: Bearer sk-live-secret is invalid"

    result = redact_secrets_in_text(value)

    assert "sk-live-secret" not in result
    assert REDACTED_TOKEN_PLACEHOLDER in result
    assert "request rejected" in result


def test_redact_secrets_in_text_scrubs_authorization_json_text_pattern() -> None:
    """A JSON-*looking* fragment that is not itself valid top-level JSON
    (e.g. embedded in a larger non-JSON error message) must still have its
    "Authorization": "..." pattern scrubbed by the regex fallback."""
    value = 'error body: {"Authorization": "Bearer sk-live-secret"} <- rejected'

    result = redact_secrets_in_text(value)

    assert "sk-live-secret" not in result
    assert REDACTED_TOKEN_PLACEHOLDER in result


def test_redact_sensitive_leaves_non_sensitive_values_untouched() -> None:
    value = {"model": "meta-llama/Llama-3.1-8B-Instruct", "nested": {"count": 3}}

    assert redact_sensitive(value) == value


def test_redact_sensitive_scrubs_bearer_token_under_unrecognized_key() -> None:
    """redact_sensitive's key-based replacement only fires for keys in
    SENSITIVE_DATA_SPEC_KEYS. Untrusted content (e.g. a remote endpoint's
    response body) can reflect a credential back under any key, so every
    string leaf must also get the regex-based bearer-token scrub."""
    value = {"debug": {"request_headers": "Authorization: Bearer sk-live-secret"}}

    result = redact_sensitive(value)

    assert "sk-live-secret" not in json.dumps(result)
    assert REDACTED_TOKEN_PLACEHOLDER in json.dumps(result)
