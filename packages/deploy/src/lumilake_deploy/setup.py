"""Deploy-the-stack orchestration."""

import importlib
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from lumilake import envs

from . import flowmesh as fm_mod
from .docker_client import image_exists, image_pull
from .env import ENV_FILE_NAME, FLOWMESH_ENV_FILE_NAME
from .errors import DeployError
from .shell import check_docker, info, require_commands, run, wait_healthy

SERVER_IMAGE_NAME = "lumilake_server"


def server_image_ref(image_tag: str) -> str:
    """Fully-qualified server image reference from ``envs``."""
    registry = (envs.LUMILAKE_REGISTRY or "ghcr.io/mlsys-io").rstrip("/")
    return f"{registry}/{SERVER_IMAGE_NAME}:{image_tag}"


@dataclass
class SetupOptions:
    reset: bool = False
    no_server: bool = False
    background: bool = False


@dataclass
class _InfraLayout:
    """Snapshot of infra decisions derived from ``.env``."""

    deploy_fm: bool


def load_project_env(project_root: Path) -> None:
    """Refresh :mod:`lumilake.envs` from this deploy root's ``.env``."""
    env_file = project_root / ENV_FILE_NAME
    if env_file.is_file():
        load_dotenv(env_file, override=True)
        importlib.reload(envs)


def _resolve_infra_layout(project_root: Path) -> _InfraLayout:
    """Decide which sibling infra to bring up alongside the server."""
    return _InfraLayout(deploy_fm=(project_root / FLOWMESH_ENV_FILE_NAME).is_file())


def _reset_stack(project_root: Path) -> None:
    """Wipe Docker-managed server and FlowMesh state."""
    env_fm = project_root / FLOWMESH_ENV_FILE_NAME
    if env_fm.is_file():
        info("Cleaning FlowMesh stack...")
        try:
            fm_mod.stack_clean(env_fm)
        except Exception as exc:  # noqa: BLE001 - best effort cleanup
            info(f"FlowMesh cleanup failed (continuing): {exc}")

    info("Resetting: stopping the lumilake server container...")
    run(
        ["docker", "compose", "--profile", "server", "down", "-v"],
        cwd=project_root,
        check=False,
    )


def build_server_image(project_root: Path, image_tag: str) -> None:
    image = server_image_ref(image_tag)
    info(f"Building lumilake server image ({image})...")
    cmd = ["docker", "build", "-t", image, "."]
    run(cmd, cwd=project_root, env={"DOCKER_BUILDKIT": "1"})


def pull_server_image(image_tag: str) -> None:
    image = server_image_ref(image_tag)
    info(f"Pulling {image}...")
    image_pull(image)


def _start_server(
    project_root: Path,
    *,
    background: bool,
    image_tag: str,
) -> None:
    host = envs.LUMILAKE_SERVER_HOST or "0.0.0.0"
    port = str(envs.LUMILAKE_SERVER_PORT or 9000)
    if not background:
        os.execvp(
            "uv",
            [
                "uv",
                "run",
                "python",
                "scripts/serve.py",
                "--host",
                host,
                "--port",
                port,
            ],
        )
    image = server_image_ref(image_tag)
    if not image_exists(image):
        raise DeployError(
            f"Server image {image} not found locally. "
            "Run `lumilake deploy pull` (to fetch the published image) or "
            "`lumilake deploy build` (to build from source) before `up`."
        )
    run(
        ["docker", "compose", "--profile", "server", "up", "-d", "--wait", "server"],
        cwd=project_root,
    )
    wait_healthy("lumilake-server")


def _print_ready_summary() -> None:
    port = envs.LUMILAKE_SERVER_PORT or 9000
    info("")
    info("=" * 41)
    info("  Lumilake is running!")
    info("=" * 41)
    info("")
    info(f"  Server:    http://127.0.0.1:{port}")
    info(f"  API Docs:  http://127.0.0.1:{port}/docs")
    info("")
    info("  Stop all:    lumilake deploy down")
    info("  Full reset:  lumilake deploy reset")


def run_setup(project_root: Path, options: SetupOptions) -> None:
    """Bring up Lumilake from ``.env``.

    Deployment starts the Lumilake server and, when ``.env.flowmesh`` exists,
    the bundled FlowMesh stack. It does not provision Postgres, S3-compatible
    storage, credentials, or sample data.
    """
    require_commands(["docker", "uv"])
    check_docker()

    env_file = project_root / ENV_FILE_NAME
    if not env_file.is_file():
        raise DeployError(
            f"{env_file} not found. Run ``lumilake deploy init`` "
            "(optionally with ``--flowmesh``) first."
        )
    load_project_env(project_root)

    run(["uv", "sync", "--quiet", "--extra", "cli"], cwd=project_root)

    info(f"Using env file: {env_file}")
    image_tag = envs.LUMILAKE_IMAGE_TAG or "latest"
    info(f"Image tag: {image_tag}")

    layout = _resolve_infra_layout(project_root)

    if options.reset:
        _reset_stack(project_root)

    if layout.deploy_fm:
        env_fm = project_root / FLOWMESH_ENV_FILE_NAME
        fm_mod.stack_pull(env_fm)
        fm_mod.stack_up(env_fm)
        if not fm_mod.wait_healthy(env_fm, timeout=120):
            raise DeployError("FlowMesh stack did not become healthy.")
        cpu_count = int(envs.LUMILAKE_CPU_WORKER_GROUP_SIZE or 0)
        gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        fm_mod.create_workers(env_fm, cpu_count=cpu_count, gpu_devices=gpu_devices)

    if options.no_server:
        info("Infrastructure ready. Skipping server start (--no-server).")
        info("")
        info("To start manually:")
        info(
            "  uv run python scripts/serve.py "
            f"--host {envs.LUMILAKE_SERVER_HOST or '0.0.0.0'} "
            f"--port {envs.LUMILAKE_SERVER_PORT or 9000}"
        )
        return

    if not options.background:
        info("")
        info("Starting server in foreground...")
        info(f"  API Docs:  http://127.0.0.1:{envs.LUMILAKE_SERVER_PORT or 9000}/docs")
        info("")
        _start_server(
            project_root,
            background=False,
            image_tag=image_tag,
        )
        return

    info("Starting server via Docker...")
    _start_server(
        project_root,
        background=True,
        image_tag=image_tag,
    )

    _print_ready_summary()
