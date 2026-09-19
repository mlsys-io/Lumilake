from typing import Any

from lumilake_server.routes.jobs import _workflow_template_hash


def _payload(secret: str, model: str = "dummy-model") -> dict[str, Any]:
    return {
        "name": "demo",
        "ops": [
            {
                "id": "ask",
                "op": "LLMChatOp",
                "messages": [{"role": "user", "content": "hello"}],
                "config": {
                    "model": model,
                    "api": {
                        "url": "https://api.example.com/v1/chat/completions",
                        "authorization": secret,
                    },
                },
            }
        ],
        "outputs": [{"name": "out", "ref": "ask"}],
    }


def test_template_hash_insensitive_to_api_credential() -> None:
    """Two identical workflows whose only difference is the API credential
    must produce the same template hash, so the secret never leaks into the
    grouping key."""
    hash_a = _workflow_template_hash(_payload("Bearer secret-one"), "yaml")
    hash_b = _workflow_template_hash(_payload("Bearer secret-two"), "yaml")

    assert hash_a == hash_b


def test_template_hash_sensitive_to_non_credential_fields() -> None:
    """A change to a non-credential field still changes the template hash."""
    hash_a = _workflow_template_hash(_payload("Bearer secret", model="model-a"), "yaml")
    hash_b = _workflow_template_hash(_payload("Bearer secret", model="model-b"), "yaml")

    assert hash_a != hash_b
