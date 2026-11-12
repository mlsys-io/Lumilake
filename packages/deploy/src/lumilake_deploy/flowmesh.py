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
    STACK_SLUG_ENV,
    STACK_SUFFIX_ENV,
    ensure_deploy_paths,
    stack_compose_file,
    stack_env_example,
    stack_resource_env_overrides,
)
from flowmesh_stack.docker import DockerComposeStack
from flowmesh_stack.env import ensure_env_file, load_env, parse_env_file
from flowmesh_stack.node_client import NodeClient
from flowmesh_stack.workers import create_workers as sdk_create_workers

from lumilake.log import init_child_logger

from .env import ENV_FILE_NAME, read_env_value

logger = init_child_logger("deploy.flowmesh")


def _info(msg: str) -> None:
    logger.info(msg)


def _warn(msg: str) -> None:
    logger.warning(msg)


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

WORKDIR_NAME = ".flowmesh"


def _workdir_for(env_file: str | Path) -> Path:
    """Return the FlowMesh working directory of the deployment owning ``env_file``."""
    return Path(env_file).resolve().parent / WORKDIR_NAME


def _make_workdir(workdir: Path) -> Path:
    """Create the deployment's FlowMesh working directory and default worker config."""
    workdir.mkdir(parents=True, exist_ok=True)
    wc = workdir / "configs" / "worker_config.yaml"
    wc.parent.mkdir(parents=True, exist_ok=True)
    if not wc.exists():
        wc.write_text("default_worker_config:\n  hb_interval: 30\n\nworkers: []\n")
    return workdir


def _load_env(ef: Path) -> None:
    """Load the stack env file, resolving its relative path keys under the workdir."""
    ensure_env_file(ef, stack_env_example())
    load_env(ef, base_dir=_workdir_for(ef), path_keys=STACK_PATH_KEYS)


_stack = DockerComposeStack(
    compose_file=stack_compose_file(),
    env_file_var="STACK_ENV_FILE",
    load_env=_load_env,
)


def env_ports(env_file: str | Path) -> dict[str, int]:
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


def stack_name_for(env_file: Path) -> str:
    """Read ``LUMILAKE_STACK_NAME`` from the ``.env`` beside ``env_file``."""
    return read_env_value(env_file.parent / ENV_FILE_NAME, "LUMILAKE_STACK_NAME")


def stack_slug(env_file: Path, *, stack_name: str | None = None) -> str:
    """Return the FlowMesh stack slug for ``env_file``.

    A non-empty ``stack_name`` (the deployment's ``LUMILAKE_STACK_NAME``)
    names the bundled stack; without one the suffix comes from
    ``.env.flowmesh``.
    """
    env = parse_env_file(env_file)
    name = stack_name_for(env_file) if stack_name is None else stack_name
    if name:
        env[STACK_SUFFIX_ENV] = name
    return stack_resource_env_overrides(env)[STACK_SLUG_ENV]


def stack_pull(env_file: str | Path) -> None:
    """Pull FlowMesh Docker images (skipped if already present locally)."""
    env_path = Path(env_file).resolve()
    _make_workdir(_workdir_for(env_path))
    if _server_image_exists(env_file):
        _info("FlowMesh images already present locally, skipping pull.")
        return
    _info("Pulling FlowMesh images...")
    _stack.run(["pull"], env_file=env_path, env=_slug_env(env_path))


def stack_up(env_file: str | Path) -> None:
    """Start the FlowMesh Docker Compose stack."""
    env_path = Path(env_file).resolve()
    workdir = _make_workdir(_workdir_for(env_path))
    _load_env(env_path)
    ensure_deploy_paths(workdir)

    errors = check_ports(env_ports(env_file))
    if errors:
        for e in errors:
            _error(e)
        raise RuntimeError("Fix port conflicts in .env.flowmesh and retry.")

    # ``--profile root`` is required so compose activates redis_control +
    # redis_telemetry; the bundled FlowMesh compose puts them under
    # ``profiles: [root]`` and without this flag they get silently skipped.
    args = ["--profile", "root", "up", "-d", "--wait"]

    _info("Starting FlowMesh stack...")
    result = _stack.run(args, env_file=env_path, env=_slug_env(env_path))
    if result.returncode != 0:
        raise RuntimeError("Failed to start FlowMesh stack.")
    _info("FlowMesh stack is up.")


def destroy_all_workers(env_file: str | Path) -> None:
    """Destroy every FlowMesh-managed worker via the SDK."""
    env = parse_env_file(Path(env_file))
    base_url = f"http://localhost:{env['SERVER_HTTP_PORT']}"
    # Bundled .env.flowmesh template ships no SERVER_TOKEN; default to
    # empty (the unauthed local stack accepts an empty token).
    token = env.get("SERVER_TOKEN", "")
    _info("Destroying FlowMesh workers...")
    with NodeClient(base_url=base_url, token=token) as client:
        client.destroy_all_workers(ignore_unreachable=True)


def stack_down(env_file: str | Path) -> None:
    """Stop the FlowMesh stack."""
    env_path = Path(env_file).resolve()
    destroy_all_workers(env_path)
    _info("Stopping FlowMesh stack...")
    # ``--profile root`` so compose sees redis_control + redis_telemetry
    # and tears them down too; without it they orphan and block re-up.
    _stack.run(
        ["--profile", "root", "down"], env_file=env_path, env=_slug_env(env_path)
    )


def stack_clean(env_file: str | Path) -> None:
    """Stop the FlowMesh stack and remove volumes."""
    env_path = Path(env_file).resolve()
    _info("Cleaning FlowMesh stack (removing volumes)...")
    _stack.run(
        ["--profile", "root", "down", "-v"],
        env_file=env_path,
        env=_slug_env(env_path),
    )


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
    """Create FlowMesh workers via SDK.

    Worker names carry the stack name as an alias prefix when the
    deployment sets one, so two stacks on one host don't collide.
    """
    env = parse_env_file(Path(env_file))
    base_url = f"http://localhost:{env['SERVER_HTTP_PORT']}"
    token = env.get("SERVER_TOKEN", "")
    slug = stack_name_for(Path(env_file))

    with NodeClient(base_url=base_url, token=token) as client:
        if cpu_count > 0:
            _pull_worker_image(env_file, "cpu")
            _info(f"Creating {cpu_count} CPU worker(s)...")
            created = sdk_create_workers(
                client, kind="cpu", count=cpu_count, slug=slug or None
            )
            for label, resp in created:
                _info(f"Created {label}: {resp['alias']}")

        device_ids, skip_message = _resolve_gpu_device_ids(gpu_devices)
        if not device_ids:
            _warn(skip_message)
        else:
            _pull_worker_image(env_file, "gpu")
            targets = ",".join(device_ids)
            _info(f"Creating {len(device_ids)} GPU worker(s): {targets}...")
            created = sdk_create_workers(
                client,
                kind="gpu",
                count=len(device_ids),
                targets=targets,
                slug=slug or None,
            )
            for label, resp in created:
                _info(f"Created {label}: {resp['alias']}")


def _resolve_gpu_device_ids(gpu_devices: str) -> tuple[list[str], str]:
    """Resolve a LUMILAKE_GPU_DEVICES-style string into explicit device ids.

    A blank value selects no device, a list selects exactly those indices, and
    ``all`` selects every GPU ``nvidia-smi`` reports. The second item explains an
    empty selection for the deploy log and is blank when devices were selected.
    """
    normalized = gpu_devices.strip()
    if not normalized:
        return [], (
            "LUMILAKE_GPU_DEVICES is blank; creating no GPU workers. "
            "Set it to a comma-separated list of indices, or to `all`, "
            "to create GPU workers."
        )
    if normalized.lower() != "all":
        selected = [tok.strip() for tok in normalized.split(",") if tok.strip()]
        if not selected:
            return [], (
                f"LUMILAKE_GPU_DEVICES={normalized} names no device; "
                "creating no GPU workers."
            )
        return selected, ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return [], (
            "LUMILAKE_GPU_DEVICES=all but nvidia-smi was not found; "
            "creating no GPU workers."
        )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        detail = f"nvidia-smi exited with status {result.returncode}"
        if stderr:
            detail = f"{detail}: {stderr}"
        return [], (
            f"LUMILAKE_GPU_DEVICES=all but the GPU probe failed ({detail}); "
            "creating no GPU workers."
        )
    detected = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not detected:
        return [], (
            "LUMILAKE_GPU_DEVICES=all but nvidia-smi reported no GPU; "
            "creating no GPU workers."
        )
    return detected, ""
