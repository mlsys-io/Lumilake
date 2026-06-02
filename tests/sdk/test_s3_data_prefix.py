"""Tests for the S3_URL / S3_DATA_PREFIX split introduced in envs.py.

S3_URL is now connection-only (endpoint + credentials, no path).
S3_DATA_PREFIX carries the bucket/prefix for SQL/S3 DataRetrievalOp reads.
"""

import importlib
import sys
import types

import pytest


def _reload_envs(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> types.ModuleType:
    """Reload lumilake.envs with a controlled os.environ snapshot.

    Reloading is necessary because envs reads module-level variables at import
    time. We remove the cached module, patch os.environ, then re-import.
    """
    monkeypatch.delitem(sys.modules, "lumilake.envs", raising=False)
    for key in ("S3_URL", "S3_DATA_PREFIX", "S3_ARCHIVE_PREFIX", "S3_CERT_FILE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.import_module("lumilake.envs")


def test_clean_split_parses_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """S3_URL connection-only + S3_DATA_PREFIX → both parsed cleanly."""
    envs = _reload_envs(
        monkeypatch,
        {
            "S3_URL": "s3://access:secret@endpoint:9000",
            "S3_DATA_PREFIX": "foo/bar",
        },
    )

    assert envs.S3_ENDPOINT == "endpoint:9000"
    assert envs.S3_ACCESS_KEY == "access"
    assert envs.S3_CONNECTION_VALUE == "secret"
    assert envs.S3_DATA_PREFIX == "foo/bar"


def test_s3_data_prefix_matches_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """envs.S3_DATA_PREFIX is the exact value of the env var."""
    envs = _reload_envs(
        monkeypatch,
        {
            "S3_URL": "s3://access:secret@endpoint:9000",
            "S3_DATA_PREFIX": "mybucket/myprefix",
        },
    )

    assert envs.S3_DATA_PREFIX == "mybucket/myprefix"


def test_s3_url_without_data_prefix_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate() raises when S3_URL is set but S3_DATA_PREFIX is missing."""
    envs = _reload_envs(
        monkeypatch,
        {"S3_URL": "s3://access:secret@endpoint:9000"},
    )

    assert envs.S3_DATA_PREFIX is None
    monkeypatch.setattr(envs, "LUMILAKE_OPTIMIZER_TYPE", "halo")
    monkeypatch.setattr(envs, "LUMILAKE_SERVER_HOST", "0.0.0.0")
    monkeypatch.setattr(envs, "RUNTIME_ORCHESTRATOR_URL", "http://localhost:18000")
    monkeypatch.setattr(envs, "LUMILAKE_SERVER_PORT", 9000)
    monkeypatch.setattr(envs, "LUMILAKE_JOB_MANAGER_TYPE", "priority")
    monkeypatch.setattr(envs, "LUMILAKE_RUNTIME_MANAGER_TYPE", "default")
    monkeypatch.setattr(envs, "LUMILAKE_STARVATION_LIMIT", 3)
    monkeypatch.setattr(envs, "LUMILAKE_BATCH_ACCUMULATION_SECONDS", 0.0)
    monkeypatch.setattr(envs, "LUMILAKE_CPU_WORKER_GROUP_SIZE", 1)
    monkeypatch.setattr(envs, "LUMILAKE_GPU_WORKER_GROUP_SIZE", 0)
    monkeypatch.setattr(envs, "LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS", 60.0)

    with pytest.raises(ValueError, match="S3_DATA_PREFIX must be a non-empty"):
        envs.validate()


def test_s3_slash_only_data_prefix_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate() raises when S3_DATA_PREFIX is all slashes (no real bucket)."""
    envs = _reload_envs(
        monkeypatch,
        {
            "S3_URL": "s3://access:secret@endpoint:9000",
            "S3_DATA_PREFIX": "/",
        },
    )

    monkeypatch.setattr(envs, "LUMILAKE_OPTIMIZER_TYPE", "halo")
    monkeypatch.setattr(envs, "LUMILAKE_SERVER_HOST", "0.0.0.0")
    monkeypatch.setattr(envs, "RUNTIME_ORCHESTRATOR_URL", "http://localhost:18000")
    monkeypatch.setattr(envs, "LUMILAKE_SERVER_PORT", 9000)
    monkeypatch.setattr(envs, "LUMILAKE_JOB_MANAGER_TYPE", "priority")
    monkeypatch.setattr(envs, "LUMILAKE_RUNTIME_MANAGER_TYPE", "default")
    monkeypatch.setattr(envs, "LUMILAKE_STARVATION_LIMIT", 3)
    monkeypatch.setattr(envs, "LUMILAKE_BATCH_ACCUMULATION_SECONDS", 0.0)
    monkeypatch.setattr(envs, "LUMILAKE_CPU_WORKER_GROUP_SIZE", 1)
    monkeypatch.setattr(envs, "LUMILAKE_GPU_WORKER_GROUP_SIZE", 0)
    monkeypatch.setattr(envs, "LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS", 60.0)

    with pytest.raises(ValueError, match="S3_DATA_PREFIX must be a non-empty"):
        envs.validate()


def test_no_s3_url_no_prefix_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """validate() does not require S3_DATA_PREFIX when S3_URL is unset."""
    envs = _reload_envs(monkeypatch, {})

    assert envs.S3_URL is None
    assert envs.S3_DATA_PREFIX is None
    monkeypatch.setattr(envs, "LUMILAKE_OPTIMIZER_TYPE", "halo")
    monkeypatch.setattr(envs, "LUMILAKE_SERVER_HOST", "0.0.0.0")
    monkeypatch.setattr(envs, "RUNTIME_ORCHESTRATOR_URL", "http://localhost:18000")
    monkeypatch.setattr(envs, "LUMILAKE_SERVER_PORT", 9000)
    monkeypatch.setattr(envs, "LUMILAKE_JOB_MANAGER_TYPE", "priority")
    monkeypatch.setattr(envs, "LUMILAKE_RUNTIME_MANAGER_TYPE", "default")
    monkeypatch.setattr(envs, "LUMILAKE_STARVATION_LIMIT", 3)
    monkeypatch.setattr(envs, "LUMILAKE_BATCH_ACCUMULATION_SECONDS", 0.0)
    monkeypatch.setattr(envs, "LUMILAKE_CPU_WORKER_GROUP_SIZE", 1)
    monkeypatch.setattr(envs, "LUMILAKE_GPU_WORKER_GROUP_SIZE", 0)
    monkeypatch.setattr(envs, "LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS", 60.0)

    envs.validate()


def test_s3_url_with_path_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """validate() raises ValueError when S3_URL carries a path component."""
    envs = _reload_envs(
        monkeypatch,
        {"S3_URL": "s3://access:secret@endpoint:9000/with/path"},
    )

    monkeypatch.setattr(envs, "LUMILAKE_OPTIMIZER_TYPE", "halo")
    monkeypatch.setattr(envs, "LUMILAKE_SERVER_HOST", "0.0.0.0")
    monkeypatch.setattr(envs, "RUNTIME_ORCHESTRATOR_URL", "http://localhost:18000")
    monkeypatch.setattr(envs, "LUMILAKE_SERVER_PORT", 9000)
    monkeypatch.setattr(envs, "LUMILAKE_JOB_MANAGER_TYPE", "priority")
    monkeypatch.setattr(envs, "LUMILAKE_RUNTIME_MANAGER_TYPE", "default")
    monkeypatch.setattr(envs, "LUMILAKE_STARVATION_LIMIT", 3)
    monkeypatch.setattr(envs, "LUMILAKE_BATCH_ACCUMULATION_SECONDS", 0.0)
    monkeypatch.setattr(envs, "LUMILAKE_CPU_WORKER_GROUP_SIZE", 1)
    monkeypatch.setattr(envs, "LUMILAKE_GPU_WORKER_GROUP_SIZE", 0)
    monkeypatch.setattr(envs, "LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS", 60.0)

    with pytest.raises(ValueError, match="S3_DATA_PREFIX"):
        envs.validate()


def test_s3_worker_url_composed_from_url_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3_WORKER_URL = S3_URL + / + S3_DATA_PREFIX (no double slash)."""
    envs = _reload_envs(
        monkeypatch,
        {
            "S3_URL": "s3://endpoint",
            "S3_DATA_PREFIX": "bucket/data",
        },
    )
    assert envs.S3_WORKER_URL == "s3://endpoint/bucket/data"


def test_s3_worker_url_handles_trailing_slash_on_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing slash on S3_URL + leading slash on S3_DATA_PREFIX collapse to one /."""
    envs = _reload_envs(
        monkeypatch,
        {
            "S3_URL": "s3://endpoint/",
            "S3_DATA_PREFIX": "/bucket/data",
        },
    )
    assert envs.S3_WORKER_URL == "s3://endpoint/bucket/data"


def test_s3_worker_url_none_when_prefix_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3_WORKER_URL is None when S3_DATA_PREFIX is not set."""
    envs = _reload_envs(
        monkeypatch,
        {"S3_URL": "s3://endpoint"},
    )
    assert envs.S3_WORKER_URL is None


def test_s3_worker_url_none_when_s3_url_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3_WORKER_URL is None when S3_URL is not set."""
    envs = _reload_envs(monkeypatch, {})
    assert envs.S3_WORKER_URL is None
