"""Local Docker image cleanup for ``lumilake deploy purge``."""

from dataclasses import dataclass
from pathlib import Path

from . import docker_client
from .errors import DeployError
from .setup import load_project_env, server_image_ref


@dataclass(frozen=True)
class PurgePlan:
    image_ref: str
    exists: bool


@dataclass(frozen=True)
class PurgeResult:
    image_ref: str
    removed: bool


def build_server_image_purge_plan(project_root: Path, image_tag: str) -> PurgePlan:
    """Plan removal of one local Lumilake server image tag."""
    load_project_env(project_root)
    image_ref = server_image_ref(image_tag)
    try:
        exists = docker_client.image_exists(image_ref)
    except Exception as exc:  # noqa: BLE001 - docker-py exposes several failures
        raise DeployError(f"Failed to inspect image {image_ref}: {exc}") from exc
    return PurgePlan(image_ref=image_ref, exists=exists)


def run_server_image_purge(plan: PurgePlan, *, force: bool = False) -> PurgeResult:
    """Remove the image tag in ``plan`` if it exists."""
    if not plan.exists:
        return PurgeResult(image_ref=plan.image_ref, removed=False)
    removed = docker_client.image_remove(plan.image_ref, force=force)
    return PurgeResult(image_ref=plan.image_ref, removed=removed)
