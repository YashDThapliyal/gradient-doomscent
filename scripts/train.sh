#!/usr/bin/env bash
# Thin wrapper: uv run doomscent train "$@"
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run doomscent train "$@"
