"""Verify committed Lumilake env examples match the deploy schema."""

import argparse
import re
from pathlib import Path

from lumilake_deploy import doctor

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

_ASSIGNMENT_RE = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)\s*=")


def _example_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text().splitlines():
        match = _ASSIGNMENT_RE.match(line)
        if match:
            keys.add(match.group(1))
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    if not ENV_EXAMPLE.is_file():
        raise SystemExit(f"Missing env example: {ENV_EXAMPLE.relative_to(REPO_ROOT)}")

    keys = _example_keys(ENV_EXAMPLE)
    required = set(doctor._ALWAYS_REQUIRED)
    known = set(
        doctor._ALWAYS_REQUIRED + doctor._RETRIEVAL_REQUIRED + doctor._OPTIONAL_KEYS
    )
    missing = sorted(required - keys)
    unknown = sorted(keys - known)

    failed = False
    if missing:
        failed = True
        print("Missing required env example keys:")
        for key in missing:
            print(f"- {key}")
    if unknown:
        failed = True
        print("Unknown env example keys:")
        for key in unknown:
            print(f"- {key}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
