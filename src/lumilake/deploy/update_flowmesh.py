"""Re-lock and install the latest FlowMesh packages."""

from pathlib import Path

from .shell import info, run

FM_PACKAGES = (
    "flowmesh-sdk",
    "flowmesh-sdk-stack",
    "flowmesh-cli",
    "flowmesh-cli-stack",
)


def run_update(project_root: Path) -> None:
    """Lock then sync so FlowMesh dependencies pick up released updates."""
    info("Fetching latest FlowMesh packages...")
    lock_cmd = ["uv", "lock"]
    for pkg in FM_PACKAGES:
        lock_cmd.extend(["--upgrade-package", pkg])
    run(lock_cmd, cwd=project_root)

    info("Installing...")
    run(["uv", "sync", "--extra", "cli"], cwd=project_root)
    info("Done.")
