"""Append the Lumilake image digest table to a GitHub Release body.

Reads the current Release body via ``gh release view``, strips any block
bounded by the Lumilake sentinel comments, then appends the table from
``--digest-file`` wrapped in fresh sentinel markers and writes the
result back via ``gh release edit --notes-file``. The sentinel-based
strip-and-replace makes re-runs idempotent — repeated invocations
replace the digest block in place instead of accumulating copies.
"""

import argparse
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

START_MARKER = "<!-- lumilake-image-digests:start -->"
END_MARKER = "<!-- lumilake-image-digests:end -->"


def _gh_bin() -> str:
    gh = shutil.which("gh")
    if not gh:
        raise RuntimeError("gh CLI not found on PATH")
    return gh


def _strip_existing_block(body: str) -> str:
    out: list[str] = []
    skip = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped == START_MARKER:
            skip = True
            continue
        if stripped == END_MARKER:
            skip = False
            continue
        if not skip:
            out.append(line)
    return "\n".join(out).rstrip()


def _read_current_body(gh: str, tag: str) -> str:
    result = subprocess.run(  # nosec B603
        [gh, "release", "view", tag, "--json", "body", "--jq", ".body"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"::error::gh release view {tag} failed: {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    body = result.stdout
    # ``gh release view --jq .body`` emits the literal string "null" for a
    # body-less Release; without normalization that token would land in the
    # published notes verbatim once the digest block is appended.
    if body.strip() in ("", "null"):
        return ""
    return body


def _write_release_body(gh: str, tag: str, body: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(body)
        notes_path = f.name
    try:
        result = subprocess.run(  # nosec B603
            [gh, "release", "edit", tag, "--notes-file", notes_path],
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"::error::gh release edit {tag} failed",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
    finally:
        Path(notes_path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--digest-file", required=True, type=Path)
    args = parser.parse_args()

    gh = _gh_bin()
    digest_table = args.digest_file.read_text().rstrip()
    current_body = _read_current_body(gh, args.tag)
    stripped = _strip_existing_block(current_body)

    separator = "\n\n" if stripped else ""
    new_body = f"{stripped}{separator}{START_MARKER}\n{digest_table}\n{END_MARKER}\n"

    _write_release_body(gh, args.tag, new_body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
