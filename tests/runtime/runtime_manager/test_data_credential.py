"""lum.id/data job credentials at dispatch.

A submission bound to a lum.id/data job (``lumid_job``) carries a job-scoped
data token. At dispatch every data spec in the FlowMesh task (DataRetrievalOp
``data`` and s3 list-input ``lumid_cfg``) is pointed at that credential, so the
worker reads with a token limited to the job's declared sources instead of the
server's LUMID_DATA_TOKEN. Other bearer-carrying fields are left alone.
"""

import copy
from types import SimpleNamespace
from typing import Any

from lumilake_server.runtime.runtime_manager import (
    DataCredential,
    FlowmeshRuntimeManager,
    _rewrite_data_credentials,
)

_JOB = DataCredential(url="https://data.example", token="ljt_job")


def _task_spec() -> dict[str, Any]:
    return {
        "metadata": {"annotations": {}},
        "spec": {
            "graph": {
                "nodes": [
                    {
                        "name": "retrieve",
                        "spec": {
                            "taskType": "data_retrieval",
                            "data": {
                                "type": "lumid",
                                "mode": "sql",
                                "lumid_data_url": "https://lum.id/data",
                                "lumid_data_token": "server-token",
                                "template": "SELECT 1",
                            },
                        },
                    },
                    {
                        "name": "list-input",
                        "spec": {
                            "data": {
                                "type": "list",
                                "lumid_cfg": {
                                    "lumid_data_url": "https://lum.id/data",
                                    "lumid_data_token": "server-token",
                                },
                            }
                        },
                    },
                    {
                        "name": "lambda",
                        "spec": {
                            "blobs": [
                                {
                                    "url": "https://lum.id/data/blobs/runtime",
                                    "headers": {"Authorization": "Bearer server-token"},
                                }
                            ]
                        },
                    },
                ]
            }
        },
    }


def test_rewrite_points_every_data_spec_at_the_job_credential() -> None:
    spec = _task_spec()
    _rewrite_data_credentials(spec, _JOB)
    retrieve, list_input, lambda_node = spec["spec"]["graph"]["nodes"]
    assert retrieve["spec"]["data"]["lumid_data_token"] == "ljt_job"
    assert retrieve["spec"]["data"]["lumid_data_url"] == "https://data.example"
    assert retrieve["spec"]["data"]["template"] == "SELECT 1"
    assert list_input["spec"]["data"]["lumid_cfg"] == {
        "lumid_data_url": "https://data.example",
        "lumid_data_token": "ljt_job",
    }
    # Lumilake's own runtime blobs are not job data: untouched.
    assert lambda_node["spec"]["blobs"][0]["headers"]["Authorization"] == (
        "Bearer server-token"
    )


def test_rewrite_leaves_specs_without_data_tokens_unchanged() -> None:
    spec = {"spec": {"graph": {"nodes": [{"spec": {"api": {"url": "x"}}}]}}}
    before = copy.deepcopy(spec)
    _rewrite_data_credentials(spec, _JOB)
    assert spec == before


def _manager(credentials: dict[str, DataCredential]) -> Any:
    return SimpleNamespace(get_data_credential=credentials.get)


def test_resolve_uses_the_member_jobs_credential() -> None:
    spec = _task_spec()
    FlowmeshRuntimeManager._resolve_data_credentials(
        _manager({"req-1": _JOB}), {"exec-x", "req-1"}, spec
    )
    data = spec["spec"]["graph"]["nodes"][0]["spec"]["data"]
    assert data["lumid_data_token"] == "ljt_job"


def test_resolve_without_a_job_credential_keeps_the_server_token() -> None:
    spec = _task_spec()
    FlowmeshRuntimeManager._resolve_data_credentials(_manager({}), {"req-1"}, spec)
    data = spec["spec"]["graph"]["nodes"][0]["spec"]["data"]
    assert data["lumid_data_token"] == "server-token"
    assert data["lumid_data_url"] == "https://lum.id/data"
