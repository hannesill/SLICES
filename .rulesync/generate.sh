#!/usr/bin/env bash
set -euo pipefail

npx --yes rulesync generate \
  --targets claudecode,cursor,codexcli \
  --features rules \
  "$@"

npx --yes rulesync generate \
  --targets claudecode,cursor,agentsskills \
  --features skills \
  "$@"
