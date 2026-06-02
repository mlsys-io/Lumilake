"""Environment-variable registry.

Module-level reads are defensive: every variable has a default, no
``os.environ[X]`` lookups, no value validation, no ``.env``-existence
check. ``from lumilake import envs`` succeeds in any environment, even
one without any ``LUMILAKE_*`` set.

Server startup (``lumilake_server.main.lifespan``) calls
:func:`load_env_file_or_raise` (recovers the loud-fail-on-missing-.env
behavior) and :func:`validate` (enforces required vars and value
ranges). SDK / CLI / deploy consumers don't import or call those.
"""

import logging
import math
import os
from collections.abc import Mapping
from typing import overload
from urllib.parse import unquote, urlparse

from dotenv import find_dotenv, load_dotenv

# Side effect: pick up a .env if one exists. No raise — server callers
# that need the file invoke load_env_file_or_raise() explicitly.
_env_path = find_dotenv()
if _env_path:
    load_dotenv(_env_path)


def load_env_file_or_raise() -> None:
    """Reproduce the loud-fail-on-missing-.env behavior at the call site.

    Server startup invokes this so operators get the clear error message
    when ``.env`` is absent. SDK / CLI / deploy consumers do not call it.
    """
    if find_dotenv():
        return
    if os.environ.get("LUMILAKE_SKIP_DOTENV_CHECK"):
        return
    raise FileNotFoundError(
        "No .env file found. Copy .env.example to .env, or set "
        "LUMILAKE_SKIP_DOTENV_CHECK=1 if you are deploying via Docker "
        "where env vars are injected directly."
    )


# Logging — always-defaulted, safe to read on any consumer.
LUMILAKE_LOG_LEVEL: str = os.environ.get("LUMILAKE_LOG_LEVEL", "INFO")

# Server-required (empty defaults; validate() enforces non-empty).
LUMILAKE_OPTIMIZER_TYPE: str = os.environ.get("LUMILAKE_OPTIMIZER_TYPE", "halo")
LUMILAKE_SERVER_HOST: str = os.environ.get("LUMILAKE_SERVER_HOST", "")
LUMILAKE_SERVER_PORT: int = int(os.environ.get("LUMILAKE_SERVER_PORT", "0") or "0")
RUNTIME_ORCHESTRATOR_URL: str = os.environ.get("LUMILAKE_RUNTIME_ORCHESTRATOR_URL", "")
RUNTIME_TOKEN: str | None = os.environ.get("LUMILAKE_RUNTIME_TOKEN") or None
"""Scheduler-internal FlowMesh credential.

Carries Lumilake's own identity for control-plane reads (worker enumeration,
profile fetches the scheduler needs to plan dispatch). Never used by HTTP
route handlers — those forward the per-request bearer.
"""
LUMILAKE_REQUIRE_IDENTITY_PROVIDER: bool = os.environ.get(
    "LUMILAKE_REQUIRE_IDENTITY_PROVIDER", ""
).strip().lower() in {"1", "true", "yes", "on"}
LUMILAKE_RECOVER_IN_FLIGHT_JOBS: bool = os.environ.get(
    "LUMILAKE_RECOVER_IN_FLIGHT_JOBS", "1"
).strip().lower() in {"1", "true", "yes", "on"}
"""Whether server startup marks pending/running jobs as failed.

Default ``True`` for single-instance deployments (a crash leaves jobs
stuck in ``running``; failing them forward unblocks operators). Set to
``0`` in any HA / multi-instance deployment where a starting standby
must not clobber jobs the active instance is still running.

Default-on semantics: ``KEY=`` (explicit empty) and ``KEY=0`` both
disable, unlike ``LUMILAKE_REQUIRE_IDENTITY_PROVIDER`` whose default is
off and whose explicit-empty matches the unset case.
"""

# Server-tunable, defaulted.
LUMILAKE_JOB_MANAGER_TYPE: str = os.environ.get(
    "LUMILAKE_JOB_MANAGER_TYPE", "priority"
).lower()
LUMILAKE_RUNTIME_MANAGER_TYPE: str = os.environ.get(
    "LUMILAKE_RUNTIME_MANAGER_TYPE", "default"
).lower()
# ``or "<default>"`` so an empty env value (KEY="" in .env) falls through
# to the default. ``os.environ.get(k, default)`` only uses ``default``
# when the key is *unset*; an empty-string value bypasses it and the
# subsequent ``int()`` / ``float()`` would crash.
LUMILAKE_OPTIMIZER_BATCH_SIZE: int = int(
    os.environ.get("LUMILAKE_OPTIMIZER_BATCH_SIZE") or "10"
)
LUMILAKE_QUEUE_QUANTUM_HIGH: int = int(
    os.environ.get("LUMILAKE_QUEUE_QUANTUM_HIGH") or "20"
)
LUMILAKE_QUEUE_QUANTUM_MEDIUM: int = int(
    os.environ.get("LUMILAKE_QUEUE_QUANTUM_MEDIUM") or "10"
)
LUMILAKE_QUEUE_QUANTUM_LOW: int = int(
    os.environ.get("LUMILAKE_QUEUE_QUANTUM_LOW") or "5"
)
LUMILAKE_STARVATION_LIMIT: int = int(os.environ.get("LUMILAKE_STARVATION_LIMIT") or "3")
LUMILAKE_BATCH_ACCUMULATION_SECONDS: float = float(
    os.environ.get("LUMILAKE_BATCH_ACCUMULATION_SECONDS") or "0"
)
LUMILAKE_CPU_WORKER_GROUP_SIZE: int = int(
    os.environ.get("LUMILAKE_CPU_WORKER_GROUP_SIZE") or "0"
)
LUMILAKE_GPU_WORKER_GROUP_SIZE: int = int(
    os.environ.get("LUMILAKE_GPU_WORKER_GROUP_SIZE") or "0"
)

LUMILAKE_POLL_TIMEOUT_SECONDS: float = float(
    os.environ.get("LUMILAKE_POLL_TIMEOUT_SECONDS") or "inf"
)
LUMILAKE_POLL_INTERVAL_SECONDS: float = float(
    os.environ.get("LUMILAKE_POLL_INTERVAL_SECONDS") or "5"
)
LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS: float = float(
    os.environ.get("LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS") or "60"
)

# Optional data-plane forwarding via lumid.data. When ``LUMID_DATA_URL``
# is set, every DBLocation SQL + every S3 op is routed through that
# service instead of speaking psycopg / minio directly. Direct mode
# stays the default when the URL is empty.
LUMID_DATA_URL: str = os.environ.get("LUMID_DATA_URL", "").strip()
LUMID_DATA_TOKEN: str | None = os.environ.get("LUMID_DATA_TOKEN") or None
# ``or "<default>"`` covers the LUMID_DATA_TIMEOUT_SECONDS="" case —
# ``os.environ.get(k, default)`` returns "" if the key is set to empty,
# bypassing the default and crashing ``float("")``.
LUMID_DATA_TIMEOUT_SECONDS: float = float(
    os.environ.get("LUMID_DATA_TIMEOUT_SECONDS") or "30"
)

LUMILAKE_HTTP_TIMEOUT_SECONDS: float = float(
    os.environ.get("LUMILAKE_HTTP_TIMEOUT_SECONDS") or "300"
)
LUMILAKE_LOG_DOWNLOAD_SPOOL_MAX_MB: int = int(
    os.environ.get("LUMILAKE_LOG_DOWNLOAD_SPOOL_MAX_MB") or "16"
)
LUMILAKE_PLUGINS: tuple[str, ...] = tuple(
    plugin
    for raw in os.environ.get("LUMILAKE_PLUGINS", "").split(",")
    if (plugin := raw.strip())
)

LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES: int = int(
    os.environ.get("LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES", "1")
)
LUMILAKE_S3_PROFILE_COST_PER_FILE: float = float(
    os.environ.get("LUMILAKE_S3_PROFILE_COST_PER_FILE", "0.05")
)
LUMILAKE_S3_PROFILE_COST_PER_MIB: float = float(
    os.environ.get("LUMILAKE_S3_PROFILE_COST_PER_MIB", "0.01")
)
LUMILAKE_LOCAL_DATA_PROFILE_PLAN_VARIANTS: str = os.environ.get(
    "LUMILAKE_LOCAL_DATA_PROFILE_PLAN_VARIANTS",
    "default,prefer_index,prefer_seq,prefer_nestloop",
)
LUMILAKE_DISABLE_DATA_PROFILE: bool = os.environ.get(
    "LUMILAKE_DISABLE_DATA_PROFILE", ""
).strip().lower() in {"1", "true", "yes", "on"}
LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING: bool = os.environ.get(
    "LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING", ""
).strip().lower() in {"1", "true", "yes", "on"}


def _positive_float(name: str, raw: str | None, default: float) -> float:
    _log = logging.getLogger(__name__)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        _log.warning(
            "%s: could not parse %r as a float; using default %.1f",
            name,
            raw,
            default,
        )
        return default
    if not math.isfinite(value) or value <= 0:
        _log.warning(
            "%s: value %r is not a positive finite number; using default %.1f",
            name,
            raw,
            default,
        )
        return default
    return value


LUMILAKE_DATA_PROFILE_CONNECT_TIMEOUT_S: float = _positive_float(
    "LUMILAKE_DATA_PROFILE_CONNECT_TIMEOUT_S",
    os.environ.get("LUMILAKE_DATA_PROFILE_CONNECT_TIMEOUT_S"),
    5.0,
)
LUMILAKE_DATA_PROFILE_STATEMENT_TIMEOUT_S: float = _positive_float(
    "LUMILAKE_DATA_PROFILE_STATEMENT_TIMEOUT_S",
    os.environ.get("LUMILAKE_DATA_PROFILE_STATEMENT_TIMEOUT_S"),
    10.0,
)

# vLLM runtime defaults
LUMILAKE_VLLM_MAX_NUM_BATCHED_TOKENS: int = int(
    os.environ.get("LUMILAKE_VLLM_MAX_NUM_BATCHED_TOKENS", "2048")
)
LUMILAKE_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE: int = int(
    os.environ.get("LUMILAKE_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE", "64")
)
LUMILAKE_VLLM_GPU_MEMORY_UTILIZATION: float = float(
    os.environ.get("LUMILAKE_VLLM_GPU_MEMORY_UTILIZATION", "0.9")
)


def get_database_url(environ: Mapping[str, str] | None = None) -> str | None:
    source = os.environ if environ is None else environ
    return source.get("DATABASE_URL") or None


DATABASE_URL: str | None = get_database_url()


def _parse_s3_url(
    raw_url: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not raw_url:
        return None, None, None
    parsed = urlparse(raw_url)
    if parsed.scheme != "s3" or not parsed.hostname:
        return None, None, None
    endpoint = parsed.hostname
    if parsed.port is not None:
        endpoint = f"{endpoint}:{parsed.port}"
    return (
        endpoint,
        unquote(parsed.username) if parsed.username else None,
        unquote(parsed.password) if parsed.password else None,
    )


S3_URL: str | None = os.getenv("S3_URL") or None

S3_ENDPOINT, S3_ACCESS_KEY, S3_CONNECTION_VALUE = _parse_s3_url(S3_URL)
S3_CERT_FILE: str | None = os.getenv("S3_CERT_FILE")

S3_DATA_PREFIX: str | None = os.getenv("S3_DATA_PREFIX")
S3_WORKER_URL: str | None = (
    f"{S3_URL.rstrip('/')}/{S3_DATA_PREFIX.lstrip('/')}"
    if S3_URL and S3_DATA_PREFIX
    else None
)
S3_ARCHIVE_PREFIX: str | None = os.getenv("S3_ARCHIVE_PREFIX")

FLOWMESH_OUTPUT_DESTINATION: str = os.environ.get(
    "LUMILAKE_FLOWMESH_OUTPUT_DESTINATION", "local"
)

LUMILAKE_REGISTRY: str = os.environ.get("LUMILAKE_REGISTRY", "ghcr.io/mlsys-io")
LUMILAKE_IMAGE_TAG: str = os.environ.get("LUMILAKE_IMAGE_TAG", "")

REDIS_TLS_DIR: str = os.environ.get("REDIS_TLS_DIR", "")
SERVER_TLS_DIR: str = os.environ.get("SERVER_TLS_DIR", "")
SERVER_WORKER_CONFIG: str = os.environ.get("SERVER_WORKER_CONFIG", "")


@overload
def get_lumilake_timeout(
    default: float,
    environ: Mapping[str, str] | None = None,
) -> float: ...


@overload
def get_lumilake_timeout(
    default: None = None,
    environ: Mapping[str, str] | None = None,
) -> float | None: ...


def get_lumilake_timeout(
    default: float | None = None,
    environ: Mapping[str, str] | None = None,
) -> float | None:
    """Return the current SDK/CLI timeout override, if valid."""
    source = os.environ if environ is None else environ
    raw = source.get("LUMILAKE_TIMEOUT")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def get_lumilake_base_url(environ: Mapping[str, str] | None = None) -> str | None:
    source = os.environ if environ is None else environ
    return source.get("LUMILAKE_BASE_URL") or None


def get_lumilake_api_key(environ: Mapping[str, str] | None = None) -> str | None:
    """Bearer presented to the Lumilake server's IdentityProvider plugin."""
    source = os.environ if environ is None else environ
    raw = source.get("LUMILAKE_API_KEY")
    if not raw:
        return None
    stripped = raw.strip()
    return stripped or None


HARDWARE_CPU_REQUIREMENT: int = int(os.environ.get("HARDWARE_CPU_REQUIREMENT", "8"))
HARDWARE_MEMORY_REQUIREMENT: str = os.environ.get("HARDWARE_MEMORY_REQUIREMENT", "16Gi")
HARDWARE_GPU_REQUIREMENT: int = int(os.environ.get("HARDWARE_GPU_REQUIREMENT", "1"))
HARDWARE_GPU_MEMORY_REQUIREMENT: str = os.environ.get(
    "HARDWARE_GPU_MEMORY_REQUIREMENT", "8Gi"
)


def validate() -> None:
    """Enforce required env vars and valid value ranges. Raises ``ValueError``
    on the first problem.

    Server startup invokes this; SDK / CLI / deploy consumers don't.
    """
    required = {
        "LUMILAKE_OPTIMIZER_TYPE": LUMILAKE_OPTIMIZER_TYPE,
        "LUMILAKE_SERVER_HOST": LUMILAKE_SERVER_HOST,
        "LUMILAKE_RUNTIME_ORCHESTRATOR_URL": RUNTIME_ORCHESTRATOR_URL,
    }
    for name, value in required.items():
        if not value:
            raise ValueError(f"{name} is required and must be non-empty")
    if LUMILAKE_REQUIRE_IDENTITY_PROVIDER and not RUNTIME_TOKEN:
        raise ValueError(
            "LUMILAKE_RUNTIME_TOKEN is required when "
            "LUMILAKE_REQUIRE_IDENTITY_PROVIDER is set; the scheduler needs "
            "a server-internal FlowMesh credential for control-plane reads."
        )
    if LUMILAKE_SERVER_PORT <= 0:
        raise ValueError("LUMILAKE_SERVER_PORT must be a positive integer")

    if LUMILAKE_JOB_MANAGER_TYPE not in ("priority",):
        raise ValueError("LUMILAKE_JOB_MANAGER_TYPE must be 'priority'")
    if LUMILAKE_RUNTIME_MANAGER_TYPE not in ("default", "flowmesh"):
        raise ValueError(
            "LUMILAKE_RUNTIME_MANAGER_TYPE must be 'default' or 'flowmesh'"
        )
    if LUMILAKE_STARVATION_LIMIT < 0:
        raise ValueError("LUMILAKE_STARVATION_LIMIT must be >= 0")
    if LUMILAKE_BATCH_ACCUMULATION_SECONDS < 0:
        raise ValueError("LUMILAKE_BATCH_ACCUMULATION_SECONDS must be >= 0")
    if LUMILAKE_CPU_WORKER_GROUP_SIZE < 0:
        raise ValueError("LUMILAKE_CPU_WORKER_GROUP_SIZE must be >= 0")
    if LUMILAKE_GPU_WORKER_GROUP_SIZE < 0:
        raise ValueError("LUMILAKE_GPU_WORKER_GROUP_SIZE must be >= 0")
    if LUMILAKE_CPU_WORKER_GROUP_SIZE == 0 and LUMILAKE_GPU_WORKER_GROUP_SIZE == 0:
        raise ValueError(
            "At least one of LUMILAKE_CPU_WORKER_GROUP_SIZE or "
            "LUMILAKE_GPU_WORKER_GROUP_SIZE must be > 0"
        )
    if LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS <= 0:
        raise ValueError("LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS must be > 0")
    if S3_URL and urlparse(S3_URL).path.lstrip("/"):
        raise ValueError(
            "S3_URL must not carry a path component; move the prefix to "
            "S3_DATA_PREFIX (e.g. S3_URL=s3://endpoint, "
            "S3_DATA_PREFIX=my/data/path)"
        )
    if S3_URL and not S3_DATA_PREFIX:
        raise ValueError(
            "S3_DATA_PREFIX is required when S3_URL is set "
            "(e.g. S3_DATA_PREFIX=bucket/data)"
        )
