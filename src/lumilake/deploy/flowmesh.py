"""FlowMesh stack lifecycle management for lumilake deploy."""

import subprocess
import time
from pathlib import Path

from docker import DockerClient
from docker.errors import ImageNotFound
from flowmesh import FlowMesh
from flowmesh.exceptions import FlowMeshConnectionError
from flowmesh_cli_stack.utils import (
    STACK_PATH_KEYS,
    stack_compose_file,
    stack_env_example,
    stack_resource_env_overrides,
)
from flowmesh_stack.docker import DockerComposeStack
from flowmesh_stack.env import ensure_env_file, load_env, parse_env_file
from flowmesh_stack.node_client import NodeClient
from flowmesh_stack.paths import ensure_dir, ensure_file, resolve_path
from flowmesh_stack.workers import create_workers as sdk_create_workers

from lumilake import envs
from lumilake.log import init_child_logger

logger = init_child_logger("deploy.flowmesh")


def _info(msg: str) -> None:
    logger.info(msg)


def _error(msg: str) -> None:
    logger.error(msg)


# ---------------------------------------------------------------------------
# Port collision check
# ---------------------------------------------------------------------------


def check_ports(ports: dict[str, int]) -> list[str]:
    """Return error messages for ports already in use."""
    errors: list[str] = []
    for label, port in ports.items():
        result = subprocess.run(
            ["ss", "-tln", f"sport = :{port}"],
            capture_output=True,
            text=True,
        )
        if result.stdout.strip().count("\n") > 0:
            errors.append(f"Port {port} ({label}) is already in use.")
    return errors


# ---------------------------------------------------------------------------
# Stack management via SDK
# ---------------------------------------------------------------------------

_WORKDIR = Path(".flowmesh")


def _make_workdir() -> Path:
    """Create a working directory for FlowMesh compose mounts."""
    _WORKDIR.mkdir(exist_ok=True)
    (_WORKDIR / "secrets" / "tls" / "redis").mkdir(parents=True, exist_ok=True)
    (_WORKDIR / "secrets" / "tls" / "server").mkdir(parents=True, exist_ok=True)
    wc = _WORKDIR / "configs" / "worker_config.yaml"
    wc.parent.mkdir(parents=True, exist_ok=True)
    if not wc.exists():
        wc.write_text("default_worker_config:\n  hb_interval: 30\n\nworkers: []\n")
    return _WORKDIR


def _ensure_deploy_paths_in_workdir(_base_dir: Path) -> None:
    """ensure_deploy_paths pinned to _WORKDIR (ignores the cwd-based base_dir)."""
    base = _WORKDIR.resolve()
    ensure_dir(
        resolve_path(envs.REDIS_TLS_DIR, default="./secrets/tls/redis", base_dir=base)
    )
    ensure_dir(
        resolve_path(
            envs.SERVER_TLS_DIR,
            default="./secrets/tls/server",
            base_dir=base,
        )
    )
    ensure_file(
        resolve_path(
            envs.SERVER_WORKER_CONFIG,
            default="./configs/worker_config.yaml",
            base_dir=base,
        )
    )


def _load_env(ef: Path) -> None:
    ensure_env_file(ef, stack_env_example())
    load_env(ef, base_dir=_WORKDIR.resolve(), path_keys=STACK_PATH_KEYS)


_stack = DockerComposeStack(
    compose_file=stack_compose_file(),
    env_file_var="STACK_ENV_FILE",
    load_env=_load_env,
    ensure_deploy_paths=_ensure_deploy_paths_in_workdir,
)


def _env_ports(env_file: str | Path) -> dict[str, int]:
    """Extract configured ports from env file for collision checking."""
    env = parse_env_file(Path(env_file))
    return {
        "FlowMesh HTTP": int(env["SERVER_HTTP_PORT"]),
        "FlowMesh gRPC": int(env["SERVER_GRPC_PORT"]),
        "FlowMesh Redis control": int(env["REDIS_CONTROL_PORT"]),
        "FlowMesh Redis telemetry": int(env["REDIS_TELEMETRY_PORT"]),
    }


def _server_image_exists(env_file: str | Path) -> bool:
    """Check if the FlowMesh server image is already available locally."""
    env = parse_env_file(Path(env_file))
    registry = env["FLOWMESH_REGISTRY"]
    version = env["FLOWMESH_VERSION"]
    tag = f"{registry}/flowmesh_server:{version}"
    try:
        client = DockerClient.from_env()
        client.images.get(tag)
        return True
    except ImageNotFound:
        return False


def _slug_env(env_file: Path) -> dict[str, str]:
    """Resolve the FlowMesh stack slug into compose-bound env vars."""
    slug = stack_slug(env_file)
    return {
        "FLOWMESH_STACK_SLUG": slug,
        "COMPOSE_PROJECT_NAME": slug,
        "WORKER_RESULTS_DIR": f"{slug}_results",
    }


def stack_slug(env_file: Path) -> str:
    """Return the FlowMesh stack slug derived from ``env_file``."""
    env = parse_env_file(env_file)
    return stack_resource_env_overrides(env)["FLOWMESH_STACK_SLUG"]


def stack_pull(env_file: str | Path) -> None:
    """Pull FlowMesh Docker images (skipped if already present locally)."""
    _make_workdir()
    env_path = Path(env_file).resolve()
    if _server_image_exists(env_file):
        _info("FlowMesh images already present locally, skipping pull.")
        return
    _info("Pulling FlowMesh images...")
    _stack.run(["pull"], env_file=env_path, env=_slug_env(env_path))


def stack_up(env_file: str | Path) -> None:
    """Start the FlowMesh Docker Compose stack."""
    _make_workdir()
    env_path = Path(env_file).resolve()

    errors = check_ports(_env_ports(env_file))
    if errors:
        for e in errors:
            _error(e)
        raise RuntimeError("Fix port conflicts in .env.flowmesh and retry.")

    args = ["up", "-d", "--wait"]

    _info("Starting FlowMesh stack...")
    result = _stack.run(
        args, env_file=env_path, env=_slug_env(env_path), to_deploy=True
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to start FlowMesh stack.")
    _info("FlowMesh stack is up.")


def destroy_all_workers(env_file: str | Path) -> None:
    """Destroy every FlowMesh-managed worker via the SDK."""
    env = parse_env_file(Path(env_file))
    base_url = f"http://localhost:{env['SERVER_HTTP_PORT']}"
    token = env["SERVER_TOKEN"]
    _info("Destroying FlowMesh workers...")
    with NodeClient(base_url=base_url, token=token) as client:
        client.destroy_all_workers(ignore_unreachable=True)


def stack_down(env_file: str | Path) -> None:
    """Stop the FlowMesh stack."""
    if not _WORKDIR.exists():
        return
    env_path = Path(env_file).resolve()
    destroy_all_workers(env_path)
    _info("Stopping FlowMesh stack...")
    _stack.run(["down"], env_file=env_path, env=_slug_env(env_path))


def stack_clean(env_file: str | Path) -> None:
    """Stop the FlowMesh stack and remove volumes."""
    if not _WORKDIR.exists():
        return
    env_path = Path(env_file).resolve()
    _info("Cleaning FlowMesh stack (removing volumes)...")
    _stack.run(["down", "-v"], env_file=env_path, env=_slug_env(env_path))


# Aliases used by ``stop.py`` — keep the older verb-first names readable.
def stop_stack(*, env_file: str | Path) -> None:
    stack_down(env_file)


def clean_stack(*, env_file: str | Path) -> None:
    stack_clean(env_file)


# ---------------------------------------------------------------------------
# Health check via SDK
# ---------------------------------------------------------------------------


def wait_healthy(env_file: str | Path, timeout: int = 120) -> bool:
    """Wait for FlowMesh server health using the FlowMesh SDK."""
    env = parse_env_file(Path(env_file))
    base_url = f"http://localhost:{env['SERVER_HTTP_PORT']}"
    _info(f"Waiting for FlowMesh server at {base_url}...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with FlowMesh(base_url=base_url) as client:
                resp = client.system.health()
                if resp.ok:
                    _info("FlowMesh server is healthy.")
                    return True
        except (FlowMeshConnectionError, Exception):
            pass
        time.sleep(2)
    _error(f"FlowMesh server not healthy at {base_url} after {timeout}s")
    return False


# ---------------------------------------------------------------------------
# Worker creation via SDK
# ---------------------------------------------------------------------------


def _pull_worker_image(env_file: str | Path, kind: str) -> None:
    """Pre-pull the worker image for a given kind ("cpu" or "gpu") on the host.

    FlowMesh's server spawns worker containers via the Docker Engine API
    (mounted socket). Pulling via the CLI first caches the image locally
    and lets FlowMesh's subsequent image reference resolve without a pull.
    """
    env = parse_env_file(Path(env_file))
    registry = env["FLOWMESH_REGISTRY"]
    version = env["FLOWMESH_VERSION"]
    image = f"{registry}/flowmesh_worker:{version}-{kind}"
    try:
        DockerClient.from_env().images.get(image)
        _info(f"Worker image {image} already present locally; skipping pre-pull.")
        return
    except ImageNotFound:
        pass
    _info(f"Pre-pulling worker image {image}...")
    result = subprocess.run(
        ["docker", "pull", image], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to pull {image}. "
            f"Ensure `docker login {registry.split('/')[0]}` has been run "
            f"with a token that can read this image.\n{result.stderr}"
        )


def create_workers(env_file: str | Path, cpu_count: int, gpu_devices: str) -> None:
    """Create FlowMesh workers via SDK."""
    env = parse_env_file(Path(env_file))
    base_url = f"http://localhost:{env['SERVER_HTTP_PORT']}"
    token = env["SERVER_TOKEN"]

    with NodeClient(base_url=base_url, token=token) as client:
        if cpu_count > 0:
            _pull_worker_image(env_file, "cpu")
            _info(f"Creating {cpu_count} CPU worker(s)...")
            created = sdk_create_workers(client, kind="cpu", count=cpu_count)
            for label, resp in created:
                _info(f"Created {label}: {resp['name']}")

        if gpu_devices:
            _pull_worker_image(env_file, "gpu")
            _info(f"Creating GPU worker(s) for devices: {gpu_devices}...")
            created = sdk_create_workers(client, kind="gpu", targets=gpu_devices)
            for label, resp in created:
                _info(f"Created {label}: {resp['name']}")
