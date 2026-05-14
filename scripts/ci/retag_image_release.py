"""Retag the Lumilake server image as ``:latest`` for a release tag.

Writes a ``:latest`` tag pointing at the same manifest digest as
``--tag`` via ``docker buildx imagetools create``. Pure registry rewrite,
no rebuild and no daemon image-store interaction.

Downgrade protection: if a ``:latest`` reference already exists and its
``org.opencontainers.image.version`` label parses to a version greater
than or equal to ``--tag``, the script aborts unless ``--force`` is
given. A ``:latest`` that exists but lacks a readable version label is
also rejected unless ``--force``.
"""

import argparse
import json
import shutil
import subprocess  # nosec B404
import sys
from typing import Any

from packaging.version import InvalidVersion, Version

IMAGE_NAME = "lumilake_server"

_MISSING_REF_STDERR_PATTERNS = (
    "not found",
    "manifest unknown",
)


class MissingVersionLabel(Exception):
    """Existing ``:latest`` exists but its version label is unreadable."""


class TransientInspectError(Exception):
    """``imagetools inspect`` failed without a clear 'not found' signal."""


def _docker_bin() -> str:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("docker binary not found on PATH")
    return docker


def _imagetools_inspect(
    docker: str, ref: str, output_format: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        [docker, "buildx", "imagetools", "inspect", ref, "--format", output_format],
        capture_output=True,
        text=True,
        check=False,
    )


def _config_version_labels(image_meta: dict[str, Any]) -> list[str]:
    versions: list[str] = []
    for value in image_meta.values():
        if not isinstance(value, dict):
            continue
        cfg = value.get("config", {}) or {}
        labels = cfg.get("Labels") or {}
        version = labels.get("org.opencontainers.image.version")
        if isinstance(version, str) and version:
            versions.append(version)
    return versions


def _existing_latest_version(docker: str, latest_ref: str) -> Version | None:
    result = _imagetools_inspect(docker, latest_ref, "{{json .Image}}")
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if any(pat in stderr for pat in _MISSING_REF_STDERR_PATTERNS):
            return None
        raise TransientInspectError(
            f"imagetools inspect {latest_ref} failed: {result.stderr.strip()}"
        )

    try:
        image_meta = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TransientInspectError(
            f"imagetools inspect {latest_ref} returned non-JSON: {exc}"
        )

    labels = _config_version_labels(image_meta)
    if not labels:
        raise MissingVersionLabel(latest_ref)

    parsed: list[Version] = []
    for raw in labels:
        try:
            parsed.append(Version(raw.removeprefix("v")))
        except InvalidVersion:
            continue
    if not parsed:
        raise MissingVersionLabel(latest_ref)
    return max(parsed)


def _imagetools_create_latest(docker: str, source_ref: str, latest_ref: str) -> None:
    result = subprocess.run(  # nosec B603
        [
            docker,
            "buildx",
            "imagetools",
            "create",
            "--tag",
            latest_ref,
            source_ref,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"imagetools create {latest_ref} from {source_ref} failed: "
            f"{result.stderr.strip()}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--registry", default="ghcr.io/mlsys-io")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass downgrade and missing-label protection.",
    )
    args = parser.parse_args()

    try:
        new_version = Version(args.tag.removeprefix("v"))
    except InvalidVersion as exc:
        print(f"::error::release tag is not PEP 440: {args.tag} ({exc})")
        return 1

    docker = _docker_bin()
    registry = args.registry.rstrip("/")
    source_ref = f"{registry}/{IMAGE_NAME}:{args.tag}"
    latest_ref = f"{registry}/{IMAGE_NAME}:latest"

    if not args.force:
        try:
            current = _existing_latest_version(docker, latest_ref)
        except TransientInspectError as exc:
            print(f"::error::{exc}", file=sys.stderr)
            return 1
        except MissingVersionLabel:
            print(
                f"::error::{latest_ref} exists but its image.version label is "
                "unreadable. Pass --force to overwrite anyway.",
                file=sys.stderr,
            )
            return 1
        if current is not None and current >= new_version:
            print(
                f"::error::{latest_ref} already points at {current} which is "
                f">= {new_version}. Pass --force to override.",
                file=sys.stderr,
            )
            return 1

    try:
        _imagetools_create_latest(docker, source_ref, latest_ref)
    except RuntimeError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    print(f"Retagged {source_ref} as {latest_ref}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
