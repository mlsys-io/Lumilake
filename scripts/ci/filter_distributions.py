"""Trim a built distribution directory down to a comma-separated pattern set.

Drives the release workflow's staggered PyPI Trusted Publisher onboarding:
``release.yml`` builds every distribution but only uploads the subset
whose filenames match one of the configured glob patterns. Removes
non-matching files in place, keeps everything when the pattern is
``*``, and exits non-zero if the pattern set selects nothing (a
misconfigured pattern would otherwise silently publish nothing).

Patterns are split on commas, so individual patterns cannot contain a
literal comma.
"""

import argparse
import fnmatch
import sys
from pathlib import Path


def _split_patterns(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _select_kept(
    filenames: list[str], patterns: list[str]
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for name in filenames:
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            kept.append(name)
        else:
            dropped.append(name)
    return kept, dropped


def filter_directory(dist_dir: Path, raw_pattern: str) -> int:
    if raw_pattern.strip() == "*":
        return 0

    patterns = _split_patterns(raw_pattern)
    if not patterns:
        print(
            f"::error::no usable patterns parsed from {raw_pattern!r}",
            file=sys.stderr,
        )
        return 1

    filenames = sorted(p.name for p in dist_dir.iterdir() if p.is_file())
    kept, dropped = _select_kept(filenames, patterns)

    for name in dropped:
        print(f"trim: {name}")
        (dist_dir / name).unlink()

    if not kept:
        print(
            f"::error::pattern {raw_pattern!r} matched no built distributions",
            file=sys.stderr,
        )
        return 1

    print(f"kept {len(kept)} distribution(s) after filter")
    for name in kept:
        print(f"keep: {name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", required=True, type=Path)
    parser.add_argument("--pattern", required=True)
    args = parser.parse_args()
    return filter_directory(args.dist, args.pattern)


if __name__ == "__main__":
    sys.exit(main())
