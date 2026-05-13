"""Subprocess and Docker helpers for the deploy orchestration."""

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from lumilake.log import init_child_logger

from . import docker_client
from .errors import DeployError

logger = init_child_logger("deploy")


def info(msg: str) -> None:
    logger.info(msg)


def err(msg: str) -> None:
    logger.error(msg)


def run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = False,
    stdin: bytes | str | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, raising DeployError on failure unless check=False."""
    full_env = None if env is None else {**os.environ, **env}
    input_bytes: bytes | None
    if stdin is None:
        input_bytes = None
    elif isinstance(stdin, bytes):
        input_bytes = stdin
    else:
        input_bytes = stdin.encode()
    try:
        result = subprocess.run(
            list(cmd),
            cwd=cwd,
            env=full_env,
            capture_output=capture_output,
            input=input_bytes,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DeployError(f"Command not found: {cmd[0]}") from exc
    if check and result.returncode != 0:
        if capture_output:
            sys.stderr.write(result.stderr.decode(errors="replace"))
        raise DeployError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def run_output(cmd: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run a command and return its stripped stdout. Raises on non-zero exit."""
    result = run(cmd, cwd=cwd, capture_output=True)
    return result.stdout.decode().strip()


def run_silent(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> bool:
    """Run a command silently. Returns True on success, False otherwise."""
    result = run(cmd, cwd=cwd, env=env, capture_output=True, check=False)
    return result.returncode == 0


def require_commands(commands: Sequence[str]) -> None:
    """Fail fast if any of the given executables are not on PATH."""
    missing = [c for c in commands if shutil.which(c) is None]
    if missing:
        raise DeployError(
            f"Required command(s) not found on PATH: {', '.join(missing)}"
        )


def check_docker() -> None:
    """Ensure the Docker daemon is reachable."""
    require_commands(["docker"])
    if not docker_client.engine_is_up():
        raise DeployError(
            "Cannot connect to Docker. Ensure the daemon is running and your "
            "user is in the docker group (sudo usermod -aG docker $USER)."
        )
    # Compose v2 is still invoked via subprocess — check its presence separately.
    if not run_silent(["docker", "compose", "version"]):
        raise DeployError("Docker Compose v2 is required.")


def wait_healthy(container: str, timeout: int = 60) -> None:
    """Block until ``container`` reports healthy, or raise after ``timeout``s."""
    info(f"Waiting for {container} to be healthy (up to {timeout}s)...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if docker_client.container_health_status(container) == "healthy":
            info(f"{container} is healthy.")
            return
        time.sleep(2)
    if docker_client.container_exists(container):
        tail = docker_client.container_logs_tail(container, tail=30)
        if tail:
            sys.stderr.write(tail)
    raise DeployError(f"{container} did not become healthy in {timeout}s")
