#!/usr/bin/env bash
# Step 10 assets once training and evaluation are complete: re-evaluate the smoke run,
# then figures, montages, and the seed-1 timelapse. Logs to runs/finalize_v0.log.
set -uo pipefail
cd "$(dirname "$0")/.."
step() { echo "[$(date '+%F %T')] $*"; }
step "smoke re-evaluation"
uv run doomscent eval-watcher --run v0_smoke >> runs/v0_smoke/logs/eval_watcher.out 2>&1
for s in 1 2 3; do
  step "figures s$s";  uv run doomscent figures --run "v0_defend_center_s$s" || step "figures s$s FAILED"
  step "montage s$s";  uv run doomscent montage --run "v0_defend_center_s$s" --ckpts auto || step "montage s$s FAILED"
done
step "figures smoke"; uv run doomscent figures --run v0_smoke || step "figures smoke FAILED"
step "timelapse s1"; uv run doomscent timelapse --run v0_defend_center_s1 --workers 6 || step "timelapse s1 FAILED"
step "finalize done"
touch runs/FINALIZE_DONE
