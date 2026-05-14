"""Re-lock and install the latest FlowMesh packages."""

from pathlib import Path

from .errors import DeployError
from .shell import info, require_commands, run

FM_PACKAGES = (
    "flowmesh-sdk",
    "flowmesh-sdk-stack",
    "flowmesh-cli",
    "flowmesh-cli-stack",
)


def run_update(project_root: Path) -> None:
    """Lock then sync so FlowMesh dependencies pick up released updates."""
    if not (project_root / "pyproject.toml").is_file():
        raise DeployError(
            f"{project_root}/pyproject.toml not found. "
            "`update-flowmesh` only runs from a Lumilake workspace checkout."
        )
    require_commands(["uv"])
    info("Fetching latest FlowMesh packages...")
    lock_cmd = ["uv", "lock"]
    for pkg in FM_PACKAGES:
        lock_cmd.extend(["--upgrade-package", pkg])
    run(lock_cmd, cwd=project_root)

    info("Installing...")
    run(["uv", "sync", "--extra", "cli"], cwd=project_root)
    info("Done.")
