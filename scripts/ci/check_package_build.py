"""Validate built Lumilake distributions and umbrella extras.

Lumilake publishes one metapackage (``lumilake``) plus four workspace
wheels (``lumilake-sdk``, ``lumilake-cli``, ``lumilake-deploy``,
``lumilake-hook``). The server runtime (``lumilake_server``) is
image-only and must NOT appear in any wheel.
"""

import argparse
import hashlib
import subprocess  # nosec B404
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[2]

# Image-only modules forbidden in any wheel.
RUNTIME_TOP_LEVELS = {"lumilake_server"}

WORKSPACE_WHEELS: tuple[tuple[str, str, str], ...] = (
    ("lumilake_sdk-*-py3-none-any.whl", "lumilake", "lumilake"),
    ("lumilake_cli-*-py3-none-any.whl", "lumilake_cli", "lumilake_cli"),
    ("lumilake_deploy-*-py3-none-any.whl", "lumilake_deploy", "lumilake_deploy"),
    ("lumilake_hook-*-py3-none-any.whl", "lumilake_hook", "lumilake_hook"),
)


def _script_bin(env_dir: Path, name: str) -> Path:
    return env_dir / "bin" / name


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)  # nosec B603: fixed argv list, no shell.


def _single_file(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"Expected one {pattern!r}, found {len(matches)}")
    return matches[0]


def _wheel_top_levels(wheel: Path) -> set[str]:
    """Top-level entries: both ``pkg/...`` and flat ``pkg.py`` modules."""
    with ZipFile(wheel) as zf:
        names = zf.namelist()
    tops: set[str] = set()
    for name in names:
        head = name.split("/", 1)[0]
        if head.endswith(".py"):
            head = head[:-3]
        tops.add(head)
    return tops


def _check_no_runtime(wheel: Path) -> None:
    bad = sorted(_wheel_top_levels(wheel) & RUNTIME_TOP_LEVELS)
    if bad:
        raise SystemExit(
            f"{wheel.name} contains image-only runtime modules: {', '.join(bad)}"
        )


def _check_workspace_wheel(wheel: Path, top_level: str, license_sha: str) -> None:
    _check_no_runtime(wheel)
    with ZipFile(wheel) as zf:
        names = set(zf.namelist())
        top_levels = _wheel_top_levels(wheel)
        if top_level not in top_levels:
            raise SystemExit(f"{wheel.name} missing top-level package {top_level!r}")
        if any(name.startswith("tests/") for name in names):
            raise SystemExit(f"{wheel.name} should not contain tests/")
        license_entries = [
            n for n in names if n.endswith(".dist-info/licenses/LICENSE")
        ]
        if len(license_entries) != 1:
            raise SystemExit(
                f"{wheel.name} should ship exactly one dist-info licenses/LICENSE, "
                f"found {len(license_entries)}"
            )
        with zf.open(license_entries[0]) as f:
            wheel_sha = hashlib.sha256(f.read()).hexdigest()
        if wheel_sha != license_sha:
            raise SystemExit(
                f"{wheel.name} ships a LICENSE that does not match the repo root "
                f"LICENSE. Regenerate per-package copies from the root LICENSE."
            )


def _check_metapackage(wheel: Path) -> None:
    """Metapackage wheel: only ``.dist-info/`` metadata allowed."""
    with ZipFile(wheel) as zf:
        names = zf.namelist()
    non_distinfo = [name for name in names if ".dist-info/" not in name]
    if non_distinfo:
        raise SystemExit(
            "metapackage wheel contains non-metadata entries: "
            + ", ".join(non_distinfo[:5])
        )


def _smoke_metapackage_extra(
    dist_dir: Path, meta_wheel: Path, extra: str, import_name: str
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"lumilake-{extra}-smoke-") as tmp:
        env_dir = Path(tmp) / ".venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = _script_bin(env_dir, "python")
        _run(
            [
                python.as_posix(),
                "-m",
                "pip",
                "install",
                "--find-links",
                dist_dir.as_posix(),
                f"{meta_wheel.as_posix()}[{extra}]",
            ]
        )
        _run([python.as_posix(), "-c", f"import {import_name}"])


def _smoke_cli(dist_dir: Path, meta_wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="lumilake-cli-smoke-") as tmp:
        env_dir = Path(tmp) / ".venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = _script_bin(env_dir, "python")
        _run(
            [
                python.as_posix(),
                "-m",
                "pip",
                "install",
                "--find-links",
                dist_dir.as_posix(),
                f"{meta_wheel.as_posix()}[cli]",
            ]
        )
        _run([_script_bin(env_dir, "lumilake").as_posix(), "--help"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist",
        default="dist",
        type=Path,
        help="Directory containing distributions built by `uv build`.",
    )
    args = parser.parse_args()

    dist_dir = args.dist.resolve()
    if not dist_dir.is_dir():
        raise SystemExit(f"Distribution directory does not exist: {dist_dir}")

    meta_wheel = _single_file(dist_dir, "lumilake-*-py3-none-any.whl")
    _single_file(dist_dir, "lumilake-*.tar.gz")
    _check_metapackage(meta_wheel)

    license_sha = hashlib.sha256((REPO_ROOT / "LICENSE").read_bytes()).hexdigest()
    for pattern, top_level, _import in WORKSPACE_WHEELS:
        wheel = _single_file(dist_dir, pattern)
        _check_workspace_wheel(wheel, top_level, license_sha)

    _smoke_metapackage_extra(dist_dir, meta_wheel, "sdk", "lumilake")
    _smoke_metapackage_extra(dist_dir, meta_wheel, "hook", "lumilake_hook")
    _smoke_metapackage_extra(dist_dir, meta_wheel, "deploy", "lumilake_deploy")
    _smoke_cli(dist_dir, meta_wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
