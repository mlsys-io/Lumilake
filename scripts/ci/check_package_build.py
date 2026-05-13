"""Validate built Lumilake distributions."""

import argparse
import subprocess
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile

REQUIRED_TOP_LEVELS = {"lumilake", "lumilake_hook"}


def _script_bin(env_dir: Path, name: str) -> Path:
    return env_dir / "bin" / name


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)  # nosec B603: fixed argv list, no shell.


def _single_file(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"Expected one {pattern!r}, found {len(matches)}")
    return matches[0]


def _check_wheel(wheel: Path) -> None:
    with ZipFile(wheel) as zf:
        names = set(zf.namelist())
    top_levels = {name.split("/", 1)[0] for name in names if "/" in name}
    missing = sorted(REQUIRED_TOP_LEVELS - top_levels)
    if missing:
        raise SystemExit("wheel missing top-level packages: " + ", ".join(missing))
    if "lumilake/py.typed" not in names:
        raise SystemExit("wheel missing lumilake/py.typed")
    entry_points = [
        name for name in names if name.endswith(".dist-info/entry_points.txt")
    ]
    if len(entry_points) != 1:
        raise SystemExit("wheel missing entry_points.txt")
    if any(name.startswith("tests/") for name in names):
        raise SystemExit("wheel should not contain tests/")


def _smoke_import(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="lumilake-package-smoke-") as tmp:
        env_dir = Path(tmp) / ".venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = _script_bin(env_dir, "python")
        _run([python.as_posix(), "-m", "pip", "install", "--no-deps", wheel.as_posix()])
        _run([python.as_posix(), "-c", "import lumilake; assert lumilake.__version__"])


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

    wheel = _single_file(dist_dir, "lumilake-*-py3-none-any.whl")
    _single_file(dist_dir, "lumilake-*.tar.gz")
    _check_wheel(wheel)
    _smoke_import(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
