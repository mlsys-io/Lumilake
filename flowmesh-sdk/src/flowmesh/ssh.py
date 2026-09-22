"""SSH session helpers shared by SDK resources and CLI wrappers."""

import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from ._constants import API_VERSION_PREFIX
from .exceptions import FlowMeshError
from .models.common import TaskStatus
from .models.tasks import TaskInfo


def task_ssh_info(task: TaskInfo) -> dict[str, Any] | None:
    """Extract SSH publish information from a task update."""
    latest = task.latest_update
    if not isinstance(latest, dict):
        return None
    ssh_info = latest.get("ssh")
    return ssh_info if isinstance(ssh_info, dict) else None


def wait_for_ssh_info(
    retrieve_task: Callable[[str], TaskInfo],
    task_id: str,
    interval: float,
    sleep: Callable[[float], Any],
) -> dict[str, Any]:
    """Poll a task until SSH session information is available."""
    while True:
        task = retrieve_task(task_id)
        status = task.status
        if status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            error = task.error or status.lower()
            raise FlowMeshError(f"Task {task_id} {status.lower()}: {error}")
        ssh_info = task_ssh_info(task)
        if ssh_info is not None:
            return ssh_info
        if status == TaskStatus.DONE:
            raise FlowMeshError("Task completed without providing SSH info.")
        sleep(interval)


async def wait_for_ssh_info_async(
    retrieve_task: Callable[[str], Awaitable[TaskInfo]],
    task_id: str,
    interval: float,
    sleep: Any,
) -> dict[str, Any]:
    """Async poll a task until SSH session information is available."""
    while True:
        task = await retrieve_task(task_id)
        status = task.status
        if status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
            error = task.error or status.lower()
            raise FlowMeshError(f"Task {task_id} {status.lower()}: {error}")
        ssh_info = task_ssh_info(task)
        if ssh_info is not None:
            return ssh_info
        if status == TaskStatus.DONE:
            raise FlowMeshError("Task completed without providing SSH info.")
        await sleep(interval)


def build_ssh_task_yaml(
    name: str,
    public_key: str | None,
    user: str,
    mode: str,
    ttl: int,
    idle_timeout: int,
    gpu: int | None,
    gpu_memory: str | None,
    cpu: int | None,
    memory: str | None,
    image: str | None,
    worker: str | None,
    env_pairs: list[str] | None,
    interactive: bool | None = None,
    command: list[str] | None = None,
    entrypoint: list[str] | None = None,
) -> str:
    """Build an SSH task definition from structured options."""
    if interactive is None:
        interactive = command is None and entrypoint is None
    elif interactive and (command is not None or entrypoint is not None):
        raise FlowMeshError("Interactive SSH tasks cannot set command or entrypoint.")
    spec: dict[str, Any] = {
        "taskType": "ssh",
    }

    if interactive:
        if not public_key:
            raise FlowMeshError(
                "Interactive SSH tasks require a public key. "
                "Provide --key or configure ~/.ssh."
            )
        spec["interactive"] = True
        spec["user"] = user
        spec["authorizedKeys"] = [public_key]
        spec["ttlSeconds"] = ttl
        spec["idleTimeoutSeconds"] = idle_timeout
        spec["accessMode"] = mode
    else:
        spec["interactive"] = False
        if command is not None:
            spec["command"] = command
        if entrypoint is not None:
            spec["entrypoint"] = entrypoint
        spec["ttlSeconds"] = ttl

    if image:
        spec["image"] = image

    hw: dict[str, Any] = {}
    if gpu and gpu > 0:
        gpu_spec: dict[str, Any] = {"count": gpu}
        if gpu_memory:
            gpu_spec["memory"] = gpu_memory
        hw["gpu"] = gpu_spec
    if cpu:
        hw["cpu"] = cpu
    if memory:
        hw["memory"] = memory
    if hw:
        spec["resources"] = {"hardware": hw}

    if env_pairs:
        env_dict: dict[str, str] = {}
        for pair in env_pairs:
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            env_dict[key.strip()] = value.strip()
        if env_dict:
            spec["env"] = env_dict

    spec["output"] = {"destination": {"type": "http"}}

    metadata: dict[str, Any] = {"name": name}
    if worker:
        metadata["annotations"] = {"schedule_hint": {"selected_worker": worker}}

    return yaml.dump(
        {
            "apiVersion": "flowmesh/v1",
            "kind": "SSHTask",
            "metadata": metadata,
            "spec": spec,
        },
        default_flow_style=False,
        sort_keys=False,
    )


def ssh_proxy_url(base_url: str, task_id: str) -> str:
    """Build the websocket proxy URL for an SSH task."""
    base = urlsplit(base_url)
    ws_scheme = "wss" if base.scheme == "https" else "ws"
    path = f"{base.path.rstrip('/')}{API_VERSION_PREFIX}/ssh/tasks/{task_id}/proxy"
    return urlunsplit((ws_scheme, base.netloc, path, "", ""))


def direct_route_scope(ssh_info: dict[str, Any]) -> str:
    """Phrase naming where a session's direct route reaches it from.

    Empty when the worker published no scope.
    """
    scope = ssh_info.get("directScope")
    if not scope:
        return ""
    worker = ssh_info.get("workerId")
    if scope == "loopback" and worker:
        return f"loopback on worker {worker}"
    return str(scope)


def describe_direct_route(ssh_info: dict[str, Any], host: Any, port: Any) -> str | None:
    """Render ``host``/``port`` with the scope the worker published for it.

    Returns None when the worker published no scope.
    """
    note = direct_route_scope(ssh_info)
    if not host or port is None or not note:
        return None
    return f"Direct route: {host}:{port} ({note})"


def ssh_connection_commands(
    task_id: str,
    ssh_info: dict[str, Any],
    base_url: str,
) -> list[tuple[str, str]]:
    """Return suggested commands for connecting to an SSH task."""
    mode = str(ssh_info.get("mode", "direct"))
    user = str(ssh_info.get("username", "flowmesh"))
    commands: list[tuple[str, str]] = []
    base_ssh_args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
    ]
    host = str(ssh_info.get("host", "localhost"))
    port = ssh_info.get("port")
    direct_host = str(ssh_info.get("directHost", host))
    direct_port = ssh_info.get("directPort", port)
    scope_note = direct_route_scope(ssh_info)
    direct_label = f"ssh (direct, {scope_note})" if scope_note else "ssh (direct)"
    plain_label = f"ssh ({scope_note})" if scope_note else "ssh"

    def _append_direct(label: str, ssh_host: str, ssh_port: Any) -> None:
        ssh_args = list(base_ssh_args)
        if ssh_port:
            ssh_args += ["-p", str(ssh_port)]
        ssh_args.append(f"{user}@{ssh_host}")
        commands.append((label, " ".join(shlex.quote(arg) for arg in ssh_args)))

    if mode == "proxy":
        commands.append(
            ("flowmesh (proxy)", f"flowmesh ssh connect {shlex.quote(task_id)}")
        )
        commands.append(
            (
                "flowmesh (direct)",
                f"flowmesh ssh connect --direct {shlex.quote(task_id)}",
            )
        )
        _append_direct(direct_label, direct_host, direct_port)
        proxy_cmd = (
            'websocat -H "Authorization: Bearer $FLOWMESH_API_KEY" '
            + shlex.quote(ssh_proxy_url(base_url, task_id))
        )
        ssh_args = list(base_ssh_args)
        ssh_args += ["-o", f"ProxyCommand={proxy_cmd}"]
        ssh_args.append(f"{user}@{task_id}")
        commands.append(("ssh (proxy)", " ".join(shlex.quote(arg) for arg in ssh_args)))
    elif mode == "forward":
        commands.append(
            ("flowmesh (forward)", f"flowmesh ssh connect {shlex.quote(task_id)}")
        )
        commands.append(
            (
                "flowmesh (direct)",
                f"flowmesh ssh connect --direct {shlex.quote(task_id)}",
            )
        )
        _append_direct("ssh (forward)", host, port)
        _append_direct(direct_label, direct_host, direct_port)
    elif mode == "direct":
        commands.append(("flowmesh", f"flowmesh ssh connect {shlex.quote(task_id)}"))
        _append_direct(plain_label, host, port)
    return commands


def detect_public_key(home_dir: Path | None = None) -> str:
    """Find the first SSH public key under ``~/.ssh``."""
    ssh_dir = (home_dir or Path.home()) / ".ssh"
    if not ssh_dir.is_dir():
        raise FlowMeshError(
            "No ~/.ssh directory found. Use --key to specify a public key."
        )
    for pattern in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub", "id_*.pub"):
        matches = sorted(ssh_dir.glob(pattern))
        if matches:
            return matches[0].read_text().strip()
    raise FlowMeshError("No SSH public key found in ~/.ssh/. Use --key to specify one.")
