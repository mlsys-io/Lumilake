#!/usr/bin/env bash
# Auto-append DCO Signed-off-by line to commit messages.
# Used as a prepare-commit-msg pre-commit hook.
set -eo pipefail

COMMIT_MSG_FILE="$1"
COMMIT_SOURCE="${2:-}"

if [ -z "$COMMIT_MSG_FILE" ]; then
    echo "No commit message file provided."
    exit 1
fi

USER_NAME="$(git config user.name || true)"
USER_EMAIL="$(git config user.email || true)"

if [ -z "$USER_NAME" ] || [ -z "$USER_EMAIL" ]; then
    echo "git user.name or user.email is not set."
    echo "   Configure with:"
    echo "     git config user.name \"Your Name\""
    echo "     git config user.email \"your.email@example.com\""
    exit 1
fi

# Don't add if any Signed-off-by line is already present.
if grep -qE "^Signed-off-by: .+ <.+>" "$COMMIT_MSG_FILE"; then
    exit 0
fi

# Append sign-off with a blank line separator.
SOB="Signed-off-by: $USER_NAME <$USER_EMAIL>"
echo "" >> "$COMMIT_MSG_FILE"
echo "$SOB" >> "$COMMIT_MSG_FILE"
