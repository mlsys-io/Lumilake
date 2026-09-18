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


def test_redact_secrets_in_text_scrubs_non_bearer_auth_scheme() -> None:
    """config.api.authorization accepts arbitrary schemes (e.g. Basic), so a
    plain-text reflected ``Authorization: Basic ...`` must be scrubbed too,
    not just Bearer tokens."""
    value = "rejected: Authorization: Basic dXNlcjpwYXNz is invalid"

    result = redact_secrets_in_text(value)

    assert "dXNlcjpwYXNz" not in result
    assert REDACTED_TOKEN_PLACEHOLDER in result
    assert "rejected" in result


def test_redact_secrets_in_text_scrubs_multi_part_auth_value() -> None:
    """A multi-part Authorization value (e.g. AWS SigV4 or HTTP Digest) must be
    redacted in full, not just its first fragment: later parameters such as
    ``Signature`` would otherwise leak into FlowMesh error logs or archived
    task responses."""
    value = (
        "rejected: Authorization: AWS4-HMAC-SHA256 "
        "Credential=AKIA123/x, Signature=SUPERSECRET"
    )

    result = redact_secrets_in_text(value)

    assert "AKIA123" not in result
    assert "SUPERSECRET" not in result
    assert REDACTED_TOKEN_PLACEHOLDER in result
    assert "rejected" in result


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


def test_redact_sensitive_matches_authorization_key_case_insensitively() -> None:
    """Header names are case-insensitive, so a lower- or mixed-case
    ``authorization`` key must be redacted exactly like ``Authorization``."""
    for key in ("authorization", "AUTHORIZATION", "Authorization"):
        result = redact_sensitive({key: "Basic dXNlcjpwYXNz"})
        assert result[key] == REDACTED_TOKEN_PLACEHOLDER


def test_redact_sensitive_scrubs_lowercase_auth_schemes_in_dict() -> None:
    """A lower-case ``authorization`` value carrying Basic, Digest, or SigV4
    credentials must be redacted in full, not left to leak."""
    for secret in (
        "Basic dXNlcjpwYXNz",
        'Digest username="u", realm="r", nonce="n", uri="/x", response="deadbeef"',
        "AWS4-HMAC-SHA256 Credential=AKIA123/x, Signature=SUPERSECRET",
    ):
        result = redact_sensitive({"authorization": secret})
        assert result["authorization"] == REDACTED_TOKEN_PLACEHOLDER


def test_redact_secrets_in_text_scrubs_lowercase_auth_header() -> None:
    """A plain-text lower-case ``authorization:`` header carrying Basic,
    Digest, or SigV4 credentials must be scrubbed, not just Bearer."""
    for secret in (
        "Basic dXNlcjpwYXNz",
        'Digest username="u", realm="r", nonce="n", uri="/x", response="deadbeef"',
        "AWS4-HMAC-SHA256 Credential=AKIA123/x, Signature=SUPERSECRET",
    ):
        value = f"rejected: authorization: {secret} is invalid"
        result = redact_secrets_in_text(value)
        assert secret not in result
        assert REDACTED_TOKEN_PLACEHOLDER in result


def test_redact_secrets_in_text_scrubs_lowercase_bearer_in_text() -> None:
    """A lower-case ``bearer <token>`` embedded in a plain-text blob must be
    scrubbed, not just the capitalized ``Bearer`` form."""
    value = "token is bearer abc123SECRET here"

    result = redact_secrets_in_text(value)

    assert "abc123SECRET" not in result
    assert REDACTED_TOKEN_PLACEHOLDER in result


def test_redact_secrets_in_text_scrubs_lowercase_authorization_json_text() -> None:
    """A lower-case ``"authorization": "..."`` JSON fragment embedded in a
    larger text blob must be scrubbed, not just the capitalized form."""
    value = 'error body: {"authorization": "Basic bWU6c2VjcmV0"} <- rejected'

    result = redact_secrets_in_text(value)

    assert "bWU6c2VjcmV0" not in result
    assert REDACTED_TOKEN_PLACEHOLDER in result
