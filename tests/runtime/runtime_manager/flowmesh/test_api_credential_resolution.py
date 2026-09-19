from typing import Any

import pytest
from lumilake import envs

from lumilake_server.runtime.runtime_manager.flowmesh import (
    _API_CREDENTIAL_PLACEHOLDER,
    FlowmeshRuntimeManager,
)

_TRUSTED_URL = "https://lum.id/llm/v1/chat/completions"
_UNTRUSTED_URL = "https://api.example.com/v1/chat/completions"


def _task_spec(url: str) -> dict[str, Any]:
    return {
        "spec": {
            "graph": {
                "nodes": [
                    {
                        "spec": {
                            "api": {
                                "url": url,
                                "headers": {
                                    "Authorization": _API_CREDENTIAL_PLACEHOLDER
                                },
                            }
                        }
                    }
                ]
            }
        }
    }


def test_resolve_trusted_origin_uses_server_pat(
    flowmesh_manager: FlowmeshRuntimeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envs, "RUNTIME_TOKEN", "server-pat")
    task_spec = _task_spec(_TRUSTED_URL)

    flowmesh_manager._resolve_api_credentials("req-1", task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "server-pat"


def test_resolve_untrusted_origin_uses_caller_credential(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    flowmesh_manager.set_api_credential("req-1", "Bearer caller-key")
    task_spec = _task_spec(_UNTRUSTED_URL)

    flowmesh_manager._resolve_api_credentials("req-1", task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Bearer caller-key"


def test_resolve_untrusted_origin_without_credential_leaves_placeholder(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    task_spec = _task_spec(_UNTRUSTED_URL)

    flowmesh_manager._resolve_api_credentials("req-1", task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert (
        node["spec"]["api"]["headers"]["Authorization"] == _API_CREDENTIAL_PLACEHOLDER
    )


def test_resolve_skips_non_placeholder_headers(
    flowmesh_manager: FlowmeshRuntimeManager,
) -> None:
    task_spec: dict[str, Any] = {
        "spec": {
            "graph": {
                "nodes": [
                    {
                        "spec": {
                            "api": {
                                "url": _UNTRUSTED_URL,
                                "headers": {"Authorization": "Bearer already-set"},
                            }
                        }
                    }
                ]
            }
        }
    }

    flowmesh_manager._resolve_api_credentials("req-1", task_spec)

    node = task_spec["spec"]["graph"]["nodes"][0]
    assert node["spec"]["api"]["headers"]["Authorization"] == "Bearer already-set"
