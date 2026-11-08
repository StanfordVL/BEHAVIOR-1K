#!/usr/bin/env bash
# Determines the base commit to diff against for diff-scoped type checking,
# and writes it to $GITHUB_OUTPUT as `base_sha`.
set -euo pipefail

if [ "$EVENT_NAME" = "pull_request" ]; then
  git fetch --depth=1 origin "$BASE_REF"
  base_sha=$(git merge-base "origin/$BASE_REF" HEAD)
else
  base_sha="$BEFORE_SHA"
  if [ -z "$base_sha" ] || [ "$base_sha" = "0000000000000000000000000000000000000000" ]; then
    base_sha=$(git rev-parse HEAD~1)
  fi
fi

echo "base_sha=$base_sha" >> "$GITHUB_OUTPUT"
