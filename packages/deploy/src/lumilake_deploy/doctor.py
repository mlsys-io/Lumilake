"""Validate ``.env`` (and optionally ``.env.flowmesh``) before ``deploy up``.

Mirrors :func:`flowmesh_stack.doctor.run_doctor_checks` in spirit: emits a
structured report of errors / warnings / notes the operator can read at a
glance. The primary purpose is to catch a malformed ``.env`` (missing
required keys, obviously-bad values) before ``deploy up`` brings the
server up against broken environment.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

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

# Keys required for direct compute mode (when ``LUMID_DATA_URL`` is empty).
# When LUMID_DATA_URL is set, lumid.data owns these credentials and the
# server skips direct connections — the keys become optional.
_DIRECT_MODE_REQUIRED: tuple[str, ...] = (
    "DATABASE_URL",
    "S3_URL",
    "S3_USER_DATA_PREFIX",
)

_OPTIONAL_KEYS: tuple[str, ...] = (
    "LUMILAKE_LOG_LEVEL",
    "LUMILAKE_REGISTRY",
    "LUMILAKE_OPTIMIZER_TYPE",
    "LUMILAKE_OPTIMIZER_BATCH_SIZE",
    "LUMILAKE_STARVATION_LIMIT",
    "LUMILAKE_BATCH_ACCUMULATION_SECONDS",
    "LUMILAKE_OPTIMIZER_SUBPROCESS_TIMEOUT_SECONDS",
    "LUMILAKE_CPU_WORKER_GROUP_SIZE",
    "LUMILAKE_GPU_WORKER_GROUP_SIZE",
    "LUMILAKE_FLOWMESH_OUTPUT_DESTINATION",
    "LUMILAKE_RUNTIME_TOKEN",
    "LUMILAKE_JOB_MANAGER_TYPE",
    "LUMILAKE_RUNTIME_MANAGER_TYPE",
    "LUMILAKE_POLL_TIMEOUT_SECONDS",
    "LUMILAKE_POLL_INTERVAL_SECONDS",
    "LUMILAKE_HTTP_TIMEOUT_SECONDS",
    "LUMILAKE_QUEUE_QUANTUM_MEDIUM",
    "LUMILAKE_DATA_PROFILE_NUM_TEST_QUERIES",
    "LUMILAKE_S3_PROFILE_COST_PER_FILE",
    "LUMILAKE_S3_PROFILE_COST_PER_MIB",
    "LUMILAKE_SKIP_DOTENV_CHECK",
    "S3_CERT_FILE",
    "LUMID_DATA_URL",
    "LUMID_DATA_TOKEN",
    "LUMID_DATA_TIMEOUT_SECONDS",
    "HARDWARE_CPU_REQUIREMENT",
    "HARDWARE_MEMORY_REQUIREMENT",
    "HARDWARE_GPU_REQUIREMENT",
    "HARDWARE_GPU_MEMORY_REQUIREMENT",
)

_KNOWN_KEYS: frozenset[str] = frozenset(
    _ALWAYS_REQUIRED + _DIRECT_MODE_REQUIRED + _OPTIONAL_KEYS
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
            report.error(f"{key} is required but missing or empty")
    if not values.get("LUMID_DATA_URL"):
        for key in _DIRECT_MODE_REQUIRED:
            if not values.get(key):
                report.error(
                    f"{key} is required in direct mode "
                    "(set ``LUMID_DATA_URL`` to route through lumid.data instead)"
                )


def _check_unknown(values: dict[str, str], report: DoctorReport) -> None:
    for key in values:
        if key not in _KNOWN_KEYS:
            report.warning(f"{key} is unknown — typo, or stale schema?")


def _check_data_plane(values: dict[str, str], report: DoctorReport) -> None:
    """When LUMID_DATA_URL is set, flag the rest of the lumid block to
    encourage consistent configuration.
    """
    if values.get("LUMID_DATA_URL") and not values.get("LUMID_DATA_TIMEOUT_SECONDS"):
        report.note("LUMID_DATA_TIMEOUT_SECONDS unset; defaulting to 30s.")


def _check_s3_url(values: dict[str, str], report: DoctorReport) -> None:
    if values.get("LUMID_DATA_URL"):
        return
    raw_url = values.get("S3_URL", "")
    if not raw_url:
        return
    parsed = urlparse(raw_url)
    if parsed.scheme != "s3":
        report.error("S3_URL must use the s3:// scheme")
    if not parsed.hostname:
        report.error("S3_URL must include an endpoint host")
    if not parsed.username or not parsed.password:
        report.error("S3_URL must include access key and secret")
    if not parsed.path.lstrip("/"):
        report.error("S3_URL must include a bucket or bucket/prefix path")


def run_env_checks(
    env_path: Path,
    callback: Callable[[DoctorFinding], Any] | None = None,
) -> DoctorReport:
    """Validate a single lumilake ``.env`` file."""
    report = DoctorReport(callback=callback)
    if not env_path.is_file():
        report.error(
            f"{env_path} not found. Run ``lumilake deploy init`` to "
            "create one from ``.env.example``."
        )
        return report
    values = _parse_env_file(env_path)
    _check_required(values, report)
    _check_unknown(values, report)
    _check_data_plane(values, report)
    _check_s3_url(values, report)
    if not report.errors:
        report.note(f"{env_path}: schema looks correct")
    return report


def run_doctor(
    env_path: Path,
    *,
    flowmesh_env_path: Path | None = None,
    callback: Callable[[DoctorFinding], Any] | None = None,
) -> DoctorReport:
    """Top-level entry point. Validates ``.env`` and, when supplied,
    notes the presence of ``.env.flowmesh`` (full validation belongs to
    ``flowmesh stack doctor``).
    """
    report = run_env_checks(env_path, callback=callback)
    if flowmesh_env_path is not None:
        if flowmesh_env_path.is_file():
            report.note(
                f"{flowmesh_env_path}: present "
                "(run ``flowmesh stack doctor`` for FlowMesh-side validation)"
            )
        else:
            report.error(
                f"{flowmesh_env_path} not found. Run "
                "``lumilake deploy init --flowmesh`` to create one."
            )
    return report
