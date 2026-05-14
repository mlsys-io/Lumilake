"""Verify the Lumilake server image for a release.

Resolves ``ghcr.io/mlsys-io/lumilake_server:<tag>``, queries it through
``docker buildx imagetools inspect``, and asserts:

* The published manifest is a multi-arch OCI / Docker index.
* The platform set (excluding buildx attestation manifests with
  ``platform.architecture=unknown``) covers ``linux/amd64`` and
  ``linux/arm64``.
* Every per-platform image config carries
  ``org.opencontainers.image.version == tag`` and
  ``org.opencontainers.image.revision == commit``.

Writes a Markdown digest table to ``--markdown-file`` and, when given,
appends ``is_release_tag=<bool>`` to ``--github-output`` for downstream
workflow gating. A tag is treated as a release (eligible for ``:latest``
retag) when it parses as PEP 440 and is neither a pre-release nor a dev
release. Post-releases (``vX.Y.Z.postN``) are eligible.
"""

import argparse
import json
import shutil
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

IMAGE_NAME = "lumilake_server"
REQUIRED_PLATFORMS = frozenset({"linux/amd64", "linux/arm64"})

# release-images may race image-publish; poll on missing-manifest only.
_MISSING_MANIFEST_PATTERNS = (
    "not found",
    "manifest unknown",
)

# Buildx emits one of either mediatype depending on driver/BuildKit.
ACCEPTED_INDEX_MEDIATYPES = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }
)


def _docker_bin() -> str:
    docker = shutil.which("docker")
    if not docker:
        raise RuntimeError("docker binary not found on PATH")
    return docker


def _imagetools_inspect(docker: str, ref: str, output_format: str) -> dict[str, Any]:
    result = subprocess.run(  # nosec B603
        [docker, "buildx", "imagetools", "inspect", ref, "--format", output_format],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"imagetools inspect {ref} (format={output_format!r}) failed: "
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def _wait_for_manifest(docker: str, ref: str, timeout_seconds: float) -> None:
    """Poll for ``ref``; re-raise non-missing inspect errors immediately."""
    if timeout_seconds <= 0:
        return
    deadline = time.monotonic() + timeout_seconds
    delay = 5.0
    while True:
        result = subprocess.run(  # nosec B603
            [docker, "buildx", "imagetools", "inspect", ref, "--raw"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return
        stderr = (result.stderr or "").lower()
        if not any(pat in stderr for pat in _MISSING_MANIFEST_PATTERNS):
            raise RuntimeError(
                f"imagetools inspect {ref} failed: {result.stderr.strip()}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Timed out waiting for {ref} to appear on the registry "
                f"after {timeout_seconds:.0f}s."
            )
        time.sleep(min(delay, remaining))
        delay = min(delay * 1.5, 30.0)


def _is_release(tag: str) -> bool:
    try:
        parsed = Version(tag.removeprefix("v"))
    except InvalidVersion:
        return False
    if parsed.is_prerelease or parsed.is_devrelease:
        return False
    if parsed.local is not None:
        return False
    return True


def _platforms(manifest: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for entry in manifest.get("manifests", []):
        platform = entry.get("platform") or {}
        arch = platform.get("architecture")
        os_name = platform.get("os")
        if arch == "unknown" or os_name == "unknown":
            continue
        if arch and os_name:
            out.add(f"{os_name}/{arch}")
    return out


def _per_platform_configs(image_meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Returns {"linux/amd64": image-config-dict, ...} from `imagetools
    inspect --format '{{json .Image}}'`."""
    out: dict[str, dict[str, Any]] = {}
    for key, value in image_meta.items():
        if not isinstance(value, dict):
            continue
        out[key] = value
    return out


def _config_labels(config: dict[str, Any]) -> dict[str, str]:
    cfg = config.get("config", {}) or {}
    labels = cfg.get("Labels") or {}
    return {str(k): str(v) for k, v in labels.items()}


def _check_target(
    docker: str,
    registry: str,
    tag: str,
    commit: str,
    wait_seconds: float,
) -> tuple[str, str]:
    """Verify the single image. Returns (image_ref, primary_digest)."""
    image_ref = f"{registry.rstrip('/')}/{IMAGE_NAME}:{tag}"
    _wait_for_manifest(docker, image_ref, wait_seconds)
    manifest = _imagetools_inspect(docker, image_ref, "{{json .Manifest}}")
    mediatype = manifest.get("mediaType")
    if mediatype not in ACCEPTED_INDEX_MEDIATYPES:
        raise RuntimeError(
            f"{image_ref} is not a multi-arch index " f"(got mediaType={mediatype!r})"
        )
    platforms = _platforms(manifest)
    missing = REQUIRED_PLATFORMS - platforms
    if missing:
        raise RuntimeError(
            f"{image_ref} is missing platforms: {sorted(missing)} "
            f"(found: {sorted(platforms)})"
        )

    per_platform = _per_platform_configs(
        _imagetools_inspect(docker, image_ref, "{{json .Image}}")
    )
    for platform in REQUIRED_PLATFORMS:
        config = per_platform.get(platform)
        if config is None:
            raise RuntimeError(
                f"{image_ref} missing per-platform config for {platform}"
            )
        labels = _config_labels(config)
        actual_version = labels.get("org.opencontainers.image.version", "")
        actual_revision = labels.get("org.opencontainers.image.revision", "")
        if actual_version != tag:
            raise RuntimeError(
                f"{image_ref} ({platform}) declares image.version "
                f"{actual_version!r}, expected {tag!r}"
            )
        if actual_revision != commit:
            raise RuntimeError(
                f"{image_ref} ({platform}) declares image.revision "
                f"{actual_revision!r}, expected {commit!r}"
            )

    digest = str(manifest.get("digest", "")) or "<unknown>"
    return image_ref, digest


def _render_markdown(image_ref: str, digest: str, tag: str, commit: str) -> str:
    return (
        f"### Published image for `{tag}`\n\n"
        f"| Image | Digest |\n"
        f"| --- | --- |\n"
        f"| `{image_ref}` | `{digest}` |\n\n"
        f"Built from commit `{commit}` (multi-arch: "
        f"{', '.join(sorted(REQUIRED_PLATFORMS))}).\n"
    )


def _write_github_output(path: Path | None, is_release: bool) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(f"is_release_tag={'true' if is_release else 'false'}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--registry", default="ghcr.io/mlsys-io", help="Registry prefix."
    )
    parser.add_argument(
        "--markdown-file",
        type=Path,
        required=True,
        help="Where to write the Markdown digest table.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="GITHUB_OUTPUT path to write is_release_tag=...",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=1800.0,
        help=(
            "Seconds to poll for the manifest; covers the "
            "release-images / image-publish race. Default 30 min."
        ),
    )
    args = parser.parse_args()

    docker = _docker_bin()
    try:
        image_ref, digest = _check_target(
            docker, args.registry, args.tag, args.commit, args.wait_seconds
        )
    except RuntimeError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    args.markdown_file.write_text(
        _render_markdown(image_ref, digest, args.tag, args.commit)
    )
    _write_github_output(args.github_output, _is_release(args.tag))
    print(f"Verified {image_ref} ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
