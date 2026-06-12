"""Tests for S3_DATA_PREFIX and S3_ARCHIVE_PREFIX in envs.py.

S3_DATA_PREFIX and S3_ARCHIVE_PREFIX are logical blob-key prefixes used
in lumid-data-app's store. They are plain strings; there is no S3 connection
parsed from envs.
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
    Patching find_dotenv prevents any local .env file from polluting the
    controlled snapshot.
    """
    import dotenv

    monkeypatch.setattr(dotenv, "find_dotenv", lambda: "")
    monkeypatch.delitem(sys.modules, "lumilake.envs", raising=False)
    for key in ("S3_DATA_PREFIX", "S3_ARCHIVE_PREFIX"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.import_module("lumilake.envs")


def test_s3_data_prefix_matches_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """envs.S3_DATA_PREFIX is the exact value of the env var."""
    envs = _reload_envs(monkeypatch, {"S3_DATA_PREFIX": "mybucket/myprefix"})
    assert envs.S3_DATA_PREFIX == "mybucket/myprefix"


def test_s3_data_prefix_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """envs.S3_DATA_PREFIX is None when the env var is unset."""
    envs = _reload_envs(monkeypatch, {})
    assert envs.S3_DATA_PREFIX is None


def test_s3_archive_prefix_matches_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """envs.S3_ARCHIVE_PREFIX is the exact value of the env var."""
    envs = _reload_envs(monkeypatch, {"S3_ARCHIVE_PREFIX": "archive/artifacts"})
    assert envs.S3_ARCHIVE_PREFIX == "archive/artifacts"


def test_s3_archive_prefix_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """envs.S3_ARCHIVE_PREFIX is None when the env var is unset."""
    envs = _reload_envs(monkeypatch, {})
    assert envs.S3_ARCHIVE_PREFIX is None


def test_validate_passes_without_s3_data_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate() does not require S3_DATA_PREFIX."""
    envs = _reload_envs(monkeypatch, {})
    assert envs.S3_DATA_PREFIX is None
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
