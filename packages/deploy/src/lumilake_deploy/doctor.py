"""Validate ``.env`` (and optionally ``.env.flowmesh``) before ``deploy up``."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

type FindingLevel = Literal["note", "warning", "error"]


@dataclass(frozen=True)
class DoctorFinding:
    level: FindingLevel
    message: str


@dataclass
class DoctorReport:
    findings: list[DoctorFinding] = field(default_factory=list)
    callback: Callable[[DoctorFinding], Any] | None = None

    @property
    def errors(self) -> list[str]:
        return [f.message for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[str]:
        return [f.message for f in self.findings if f.level == "warning"]

    @property
    def notes(self) -> list[str]:
        return [f.message for f in self.findings if f.level == "note"]

    def error(self, message: str) -> None:
        self._add("error", message)

    def warning(self, message: str) -> None:
        self._add("warning", message)

    def note(self, message: str) -> None:
        self._add("note", message)

    def _add(self, level: FindingLevel, message: str) -> None:
        finding = DoctorFinding(level, message)
        self.findings.append(finding)
        if self.callback:
            self.callback(finding)


# Keys the server always needs.
_ALWAYS_REQUIRED: tuple[str, ...] = (
    "LUMILAKE_SERVER_HOST",
    "LUMILAKE_SERVER_PORT",
    "LUMILAKE_RUNTIME_ORCHESTRATOR_URL",
    "S3_ARCHIVE_PREFIX",
    "LUMILAKE_IMAGE_TAG",
)

# Required whenever any workflow contains a DataRetrievalOp (all modes
# route through lumid-data-app). LUMID_DATA_TOKEN is optional — it falls
# back to LUMILAKE_RUNTIME_TOKEN at SDK load time.
_RETRIEVAL_REQUIRED: tuple[str, ...] = ("LUMID_DATA_URL",)

_OPTIONAL_KEYS: tuple[str, ...] = (
    "LUMILAKE_LOG_LEVEL",
    "LUMILAKE_REGISTRY",
    "LUMILAKE_DEFAULT_OPTIMIZER",
    "LUMILAKE_OPTIMIZER_BATCH_SIZE",
    "LUMILAKE_STARVATION_LIMIT",
    "LUMILAKE_BATCH_ACCUMULATION_SECONDS",
    "LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS",
    "LUMILAKE_CPU_WORKER_GROUP_SIZE",
    "LUMILAKE_GPU_WORKER_GROUP_SIZE",
    "LUMILAKE_FLOWMESH_OUTPUT_DESTINATION",
    "LUMILAKE_RUNTIME_TOKEN",
    "LUMILAKE_REQUIRE_IDENTITY_PROVIDER",
    "LUMILAKE_RECOVER_IN_FLIGHT_JOBS",
    "LUMILAKE_JOB_MANAGER_TYPE",
    "LUMILAKE_RUNTIME_MANAGER_TYPE",
    "LUMILAKE_POLL_TIMEOUT_SECONDS",
    "LUMILAKE_POLL_INTERVAL_SECONDS",
    "LUMILAKE_HTTP_TIMEOUT_SECONDS",
    "LUMILAKE_QUEUE_QUANTUM_HIGH",
    "LUMILAKE_QUEUE_QUANTUM_LOW",
    "LUMILAKE_QUEUE_QUANTUM_MEDIUM",
    "LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES",
    "LUMILAKE_S3_PROFILE_COST_PER_FILE",
    "LUMILAKE_S3_PROFILE_COST_PER_MIB",
    "LUMILAKE_DISABLE_DATA_PROFILE",
    "LUMILAKE_LOG_DOWNLOAD_SPOOL_MAX_MB",
    "LUMILAKE_DATA_PROFILE_ENABLE_LIVE_SAMPLING",
    "LUMILAKE_PLUGINS",
    "LUMILAKE_PLUGIN_DIR",
    "LUMILAKE_REMOTE_OPTIMIZER_URL",
    "LUMILAKE_SKIP_DOTENV_CHECK",
    "S3_DATA_PREFIX",
    "LUMID_DATA_TOKEN",
    "LUMID_DATA_TIMEOUT_SECONDS",
    "HARDWARE_CPU_REQUIREMENT",
    "HARDWARE_MEMORY_REQUIREMENT",
    "HARDWARE_GPU_REQUIREMENT",
    "HARDWARE_GPU_MEMORY_REQUIREMENT",
    "LUMILAKE_GPU_DEVICES",
    "LUMILAKE_LOCAL_DATA_PROFILE_PLAN_VARIANTS",
    "LUMILAKE_VLLM_GPU_MEMORY_UTILIZATION",
    "LUMILAKE_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE",
    "LUMILAKE_VLLM_MAX_NUM_BATCHED_TOKENS",
    "REDIS_TLS_DIR",
    "SERVER_TLS_DIR",
    "SERVER_WORKER_CONFIG",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    """Tolerant ``.env`` parser — accepts ``KEY=value`` or ``KEY="value"``."""
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        values[key] = value
    return values


def _check_required(values: dict[str, str], report: DoctorReport) -> None:
    for key in _ALWAYS_REQUIRED:
        if not values.get(key):
            report.error(
                f"{key} is required but missing or empty (fix: set {key}=... in .env)"
            )
    for key in _RETRIEVAL_REQUIRED:
        if not values.get(key):
            report.error(
                f"{key} is required for DataRetrievalOp — all retrieval modes "
                f"route through lumid-data-app (fix: set {key}=... in .env)"
            )


def _check_data_plane(values: dict[str, str], report: DoctorReport) -> None:
    """Validate the lumid.data block configuration."""
    if values.get("LUMID_DATA_URL") and not values.get("LUMID_DATA_TIMEOUT_SECONDS"):
        report.note("LUMID_DATA_TIMEOUT_SECONDS unset; defaulting to 30s.")


def _check_positive_floats(values: dict[str, str], report: DoctorReport) -> None:
    """Validate that optional timeout knobs are positive floats when set."""
    for key in ("LUMID_DATA_TIMEOUT_SECONDS", "LUMILAKE_HTTP_TIMEOUT_SECONDS"):
        raw = values.get(key)
        if not raw:
            continue
        try:
            parsed = float(raw)
        except ValueError:
            report.error(
                f"{key}={raw!r} is not a valid number "
                f"(fix: set {key} to a positive number of seconds)"
            )
            continue
        if parsed <= 0:
            report.error(
                f"{key}={raw!r} must be > 0 "
                f"(fix: set {key} to a positive number of seconds)"
            )


def run_env_checks(
    env_path: Path,
    callback: Callable[[DoctorFinding], Any] | None = None,
) -> DoctorReport:
    """Validate a single lumilake ``.env`` file."""
    report = DoctorReport(callback=callback)
    if not env_path.is_file():
        report.error(
            f"{env_path} not found "
            "(fix: run ``lumilake deploy init`` to create one from "
            "``.env.example``)"
        )
        return report
    values = _parse_env_file(env_path)
    _check_required(values, report)
    _check_data_plane(values, report)
    _check_positive_floats(values, report)
    if not report.errors:
        report.note(f"{env_path}: schema looks correct")
    return report


def run_doctor(
    env_path: Path,
    *,
    flowmesh_env_path: Path | None = None,
    callback: Callable[[DoctorFinding], Any] | None = None,
) -> DoctorReport:
    """Validate ``.env`` and note the presence of ``.env.flowmesh`` when supplied."""
    report = run_env_checks(env_path, callback=callback)
    if flowmesh_env_path is not None:
        if flowmesh_env_path.is_file():
            report.note(
                f"{flowmesh_env_path}: present "
                "(run ``flowmesh stack doctor`` for FlowMesh-side validation)"
            )
        else:
            report.error(
                f"{flowmesh_env_path} not found "
                "(fix: run ``lumilake deploy init --flowmesh`` to create one)"
            )
    return report
