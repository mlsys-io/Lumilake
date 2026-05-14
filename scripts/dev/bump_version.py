"""Update synchronized Lumilake package versions and internal pins."""

import argparse
import re
from pathlib import Path

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
# Sorted longest-first so e.g. ``lumilake-cli`` matches before ``lumilake``.
FIRST_PARTY_DISTRIBUTIONS: tuple[str, ...] = (
    "lumilake-deploy",
    "lumilake-hook",
    "lumilake-sdk",
    "lumilake-cli",
    "lumilake",
)

_VERSION_RE = re.compile(r'(?m)^version = "[^"]+"$')
_RUNTIME_VERSION_RE = re.compile(r'(?m)^__version__ = "[^"]+"$')
_PIN_RE = re.compile(
    r"(?P<name>\b(?:"
    + "|".join(re.escape(name) for name in FIRST_PARTY_DISTRIBUTIONS)
    + r")\b)(?P<extras>\[[^\]]+\])?==(?P<version>[^\"'\s,\]]+)"
)


def _normalize_version(raw: str) -> str:
    try:
        return str(Version(raw.removeprefix("v")))
    except InvalidVersion as exc:
        raise SystemExit(f"Version is not PEP 440: {raw!r} ({exc}).")


def _render_pyproject(text: str, version: str, path: Path) -> str:
    versioned, count = _VERSION_RE.subn(f'version = "{version}"', text, count=1)
    if count != 1:
        rel = path.relative_to(REPO_ROOT)
        raise SystemExit(f"Expected one project version line in {rel}.")
    return _PIN_RE.sub(
        lambda m: f"{m.group('name')}{m.group('extras') or ''}=={version}",
        versioned,
    )


def _render_runtime(text: str, version: str, path: Path) -> str:
    rendered, count = _RUNTIME_VERSION_RE.subn(
        f'__version__ = "{version}"', text, count=1
    )
    if count != 1:
        rel = path.relative_to(REPO_ROOT)
        raise SystemExit(f"Expected one __version__ line in {rel}.")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Synchronized release version, e.g. 0.1.1.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the package files are not already set to the version.",
    )
    args = parser.parse_args()

    version = _normalize_version(args.version)
    rendered: list[tuple[Path, str, str]] = []
    for path in PACKAGE_PYPROJECTS:
        current = path.read_text()
        rendered.append((path, current, _render_pyproject(current, version, path)))
    for path in (SDK_INIT_MODULE, SERVER_INIT_MODULE):
        current = path.read_text()
        rendered.append((path, current, _render_runtime(current, version, path)))

    changed = [path for path, current, updated in rendered if current != updated]
    if args.check:
        if changed:
            print("Package versions need updates:")
            for path in changed:
                print(f"- {path.relative_to(REPO_ROOT)}")
            return 1
        print(f"Package versions are already set to {version}.")
        return 0

    for path, current, updated in rendered:
        if current != updated:
            path.write_text(updated)

    if changed:
        print(f"Updated package versions and internal pins to {version}:")
        for path in changed:
            print(f"- {path.relative_to(REPO_ROOT)}")
    else:
        print(f"Package versions are already set to {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
