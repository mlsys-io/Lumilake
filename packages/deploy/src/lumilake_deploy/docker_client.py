"""Thin wrappers over ``docker-py`` for container lifecycle ops."""

import datetime as dt
from collections.abc import Iterator
from functools import cache

import docker
from docker.errors import APIError, DockerException, ImageNotFound
from docker.models.containers import Container
from docker.models.volumes import Volume

from .errors import DeployError


@cache
def get_docker_client() -> docker.DockerClient:
    """Return a cached ``DockerClient`` built from the environment."""
    try:
        return docker.from_env()
    except DockerException as exc:
        raise DeployError(
            "Cannot connect to Docker. Ensure the daemon is running and your "
            "user is in the docker group (sudo usermod -aG docker $USER)."
        ) from exc


def engine_is_up() -> bool:
    """Return True when the Docker daemon is reachable.

    ``ping()`` is an explicit probe — we don't wrap it in a check-first
    pattern because "is the daemon alive" has no underlying resource to
    list. DockerException here is the documented "unreachable" signal,
    not a swallow; anything else would be a library contract violation.
    """
    try:
        return bool(get_docker_client().ping())
    except (DockerException, DeployError):
        return False


def image_exists(tag: str) -> bool:
    """Return True when ``tag`` is present in the local image store."""
    try:
        get_docker_client().images.get(tag)
        return True
    except ImageNotFound:
        return False


def image_pull(tag: str) -> None:
    """Pull ``tag`` from its registry into the local image store."""
    client = get_docker_client()
    try:
        client.images.pull(tag)
    except (APIError, ImageNotFound) as exc:
        raise DeployError(f"Failed to pull image {tag}: {exc}") from exc


def image_remove(reference: str, *, force: bool = False) -> bool:
    """Remove a local Docker image tag.

    Returns ``False`` when the tag is already absent. Raises ``DeployError``
    for Docker API failures such as an image still being used by a container.
    """
    try:
        get_docker_client().images.remove(image=reference, force=force)
        return True
    except ImageNotFound:
        return False
    except APIError as exc:
        raise DeployError(f"Failed to remove image {reference}: {exc}") from exc


def _find_container(name: str) -> Container | None:
    matches = get_docker_client().containers.list(all=True, filters={"name": name})
    for container in matches:
        if container.name == name:
            return container
    return None


def container_exists(name: str) -> bool:
    return _find_container(name) is not None


def container_stop(name: str, *, timeout: int = 30) -> bool:
    """Stop ``name`` if present. Returns True on success, False if missing."""
    container = _find_container(name)
    if container is None:
        return False
    try:
        container.stop(timeout=timeout)
    except APIError as exc:
        raise DeployError(f"Failed to stop container {name}: {exc}") from exc
    return True


def container_restart(name: str, *, timeout: int = 30) -> None:
    container = _find_container(name)
    if container is None:
        raise DeployError(f"Container {name} not found")
    try:
        container.restart(timeout=timeout)
    except APIError as exc:
        raise DeployError(f"Failed to restart container {name}: {exc}") from exc


def container_logs_tail(
    name: str,
    *,
    tail: int = 30,
    since: dt.datetime | None = None,
    timestamps: bool = False,
) -> str:
    """Return the last ``tail`` log lines (or empty string if missing).

    ``since`` restricts output to entries after that timestamp.
    ``timestamps`` prepends an RFC3339 timestamp to each line.
    """
    container = _find_container(name)
    if container is None:
        return ""
    try:
        data = container.logs(tail=tail, since=since, timestamps=timestamps)
    except APIError as exc:
        raise DeployError(f"Failed to read logs for {name}: {exc}") from exc
    return data.decode(errors="replace") if isinstance(data, bytes) else str(data)


def container_logs_stream(
    name: str,
    *,
    since: dt.datetime | None = None,
    timestamps: bool = False,
) -> Iterator[bytes]:
    """Stream follow-mode logs from the container as byte chunks."""
    container = _find_container(name)
    if container is None:
        raise DeployError(f"Container {name} not found")
    try:
        yield from container.logs(
            stream=True, follow=True, since=since, timestamps=timestamps
        )
    except APIError as exc:
        raise DeployError(f"Failed to stream logs for {name}: {exc}") from exc


def container_health_status(name: str) -> str:
    """Return ``healthy`` / ``unhealthy`` / ``starting``, or "" when the
    container is missing or has no healthcheck configured."""
    container = _find_container(name)
    if container is None:
        return ""
    health = container.attrs["State"].get("Health")
    return "" if health is None else str(health["Status"])


def container_status(name: str) -> str:
    """Return the container's lifecycle state, or ``missing`` when absent."""
    container = _find_container(name)
    if container is None:
        return "missing"
    return container.status


def _find_volume(name: str) -> Volume | None:
    matches = get_docker_client().volumes.list(filters={"name": name})
    for volume in matches:
        if volume.name == name:
            return volume
    return None


def volume_exists(name: str) -> bool:
    return _find_volume(name) is not None


def volume_remove(name: str) -> bool:
    """Remove ``name`` if present. Returns True on success, False if missing.

    Raises ``DeployError`` on API errors (e.g. volume still in use).
    """
    volume = _find_volume(name)
    if volume is None:
        return False
    try:
        volume.remove()
    except APIError as exc:
        raise DeployError(f"Failed to remove volume {name}: {exc}") from exc
    return True
