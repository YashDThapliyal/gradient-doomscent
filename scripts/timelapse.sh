#!/usr/bin/env bash
# Thin wrapper: uv run doomscent timelapse "$@"
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run doomscent timelapse "$@"
