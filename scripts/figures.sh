#!/usr/bin/env bash
# Thin wrapper: uv run doomscent figures "$@"
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run doomscent figures "$@"
