"""Validate synchronized package versions for a Lumilake release."""

import argparse
import re
import tomllib
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PYPROJECTS: tuple[Path, ...] = (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "packages" / "sdk" / "pyproject.toml",
    REPO_ROOT / "packages" / "cli" / "pyproject.toml",
    REPO_ROOT / "packages" / "deploy" / "pyproject.toml",
    REPO_ROOT / "packages" / "hook" / "pyproject.toml",
)
SDK_INIT_MODULE = REPO_ROOT / "packages" / "sdk" / "src" / "lumilake" / "__init__.py"
SERVER_INIT_MODULE = REPO_ROOT / "src" / "lumilake_server" / "__init__.py"
FIRST_PARTY_DISTRIBUTIONS = {
    "lumilake",
    "lumilake-sdk",
    "lumilake-cli",
    "lumilake-deploy",
    "lumilake-hook",
}

_EXACT_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)(?:\[[^\]]+\])?==(?P<version>[^;,\s]+)"
)
_VERSION_LITERAL_RE = re.compile(r'(?m)^__version__ = "(?P<version>[^"]+)"$')


def _load_pyproject(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _release_version(tag: str | None) -> Version | None:
    if tag is None:
        return None
    try:
        return Version(tag.removeprefix("v"))
    except InvalidVersion as exc:
        raise SystemExit(f"Release tag is not PEP 440: {tag!r} ({exc}).")


def _project_dependencies(data: dict[str, Any]) -> list[str]:
    project = data.get("project", {})
    dependencies = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for specs in optional.values():
        dependencies.extend(specs)
    dependency_groups = data.get("dependency-groups", {})
    for specs in dependency_groups.values():
        dependencies.extend(spec for spec in specs if isinstance(spec, str))
    return dependencies


def _parse_pyproject_version(raw: str, rel: Path) -> Version:
    try:
        return Version(raw)
    except InvalidVersion as exc:
        raise SystemExit(
            f"{rel} declares version {raw!r} which is not PEP 440 ({exc})."
        )


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _read_literal_version(path: Path, expected: Version) -> None:
    matches = list(_VERSION_LITERAL_RE.finditer(path.read_text()))
    rel = _display_path(path)
    if len(matches) != 1:
        raise SystemExit(f"Expected one __version__ line in {rel}.")
    raw = matches[0].group("version")
    try:
        version = Version(raw)
    except InvalidVersion as exc:
        raise SystemExit(
            f"{rel} declares __version__ {raw!r} which is not PEP 440 ({exc})."
        )
    if version != expected:
        raise SystemExit(
            f"{rel} declares __version__ {version}, expected release {expected}."
        )


def _check_versions(expected: Version | None) -> Version:
    versions: dict[str, Version] = {}
    for path in PACKAGE_PYPROJECTS:
        data = _load_pyproject(path)
        project = data["project"]
        name = project["name"]
        rel = path.relative_to(REPO_ROOT)
        version = _parse_pyproject_version(project["version"], rel)
        versions[name] = version
        if expected is not None and version != expected:
            raise SystemExit(
                f"{rel} declares {name} {version}, expected release {expected}."
            )

    unique_versions = set(versions.values())
    if len(unique_versions) != 1:
        formatted = ", ".join(
            f"{name}={version}" for name, version in sorted(versions.items())
        )
        raise SystemExit(f"Published package versions differ: {formatted}.")
    return next(iter(unique_versions))


def _check_internal_pins(expected: Version) -> None:
    failures: list[str] = []
    for path in PACKAGE_PYPROJECTS:
        data = _load_pyproject(path)
        rel = path.relative_to(REPO_ROOT)
        for spec in _project_dependencies(data):
            match = _EXACT_PIN_RE.match(spec)
            if match is None:
                continue
            name = match.group("name").lower().replace("_", "-")
            if name not in FIRST_PARTY_DISTRIBUTIONS:
                continue
            raw_pin = match.group("version")
            try:
                pinned = Version(raw_pin)
            except InvalidVersion:
                failures.append(f"{rel}: {spec!r} pin is not PEP 440")
                continue
            if pinned != expected:
                failures.append(f"{rel}: {spec!r} should pin =={expected}")
    if failures:
        raise SystemExit("Stale internal package pins:\n" + "\n".join(failures))


def _check_runtime_versions(expected: Version) -> None:
    _read_literal_version(SDK_INIT_MODULE, expected)
    _read_literal_version(SERVER_INIT_MODULE, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        help="Release tag to validate, usually github.event.release.tag_name.",
    )
    args = parser.parse_args()

    expected = _release_version(args.tag)
    version = _check_versions(expected)
    _check_internal_pins(version)
    _check_runtime_versions(version)
    print(f"Release package versions are synchronized at {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
