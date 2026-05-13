#!/usr/bin/env bash
# Verify that the commit message contains a DCO Signed-off-by line.
# Used as a commit-msg pre-commit hook.
set -eo pipefail

COMMIT_MSG_FILE="$1"

if [ -z "$COMMIT_MSG_FILE" ]; then
    echo "No commit message file provided."
    exit 1
fi

# Skip fixup/squash commits used during interactive rebase.
FIRST_LINE="$(head -1 "$COMMIT_MSG_FILE")"
case "$FIRST_LINE" in
    fixup!*|squash!*|amend!*) exit 0 ;;
esac

if ! grep -qE "^Signed-off-by: .+ <.+>" "$COMMIT_MSG_FILE"; then
    echo "Commit message must include a DCO sign-off line."
    echo "   Add it with:  git commit -s"
    echo "   Or append manually:"
    echo "     Signed-off-by: Your Name <your.email@example.com>"
    exit 1
fi

echo "DCO sign-off found."
