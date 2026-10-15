#!/usr/bin/env bash
# Step 10 assets once training and evaluation are complete. Logs to runs/queue_v0.log.
set -uo pipefail
cd "$(dirname "$0")/.."
step() { echo "[$(date '+%F %T')] $*"; }
for s in 1 2 3; do
  r="v0_defend_center_s$s"
  step "confirm best $r"
  uv run doomscent evaluate --run "$r" --ckpt best --episodes 100 --workers 4 --confirm-offset 1000 \
    > "runs/$r/logs/confirm_best.log" 2>&1 || step "confirm $r FAILED"
done
for s in 1 2 3; do
  r="v0_defend_center_s$s"
  step "figures $r";    uv run doomscent figures --run "$r" || step "figures $r FAILED"
  step "montage $r";    uv run doomscent montage --run "$r" --ckpts auto || step "montage $r FAILED"
  step "highlights $r"; uv run doomscent highlights --run "$r" --ckpt best || step "highlights $r FAILED"
  step "sidebyside $r"; uv run doomscent sidebyside --run "$r" --ckpts 0,best || step "sidebyside $r FAILED"
done
free_gb() { df -g /System/Volumes/Data | tail -1 | awk '{print $4}'; }
for s in 1 2 3; do
  step "timelapse s$s (free $(free_gb) GB)"
  uv run doomscent timelapse --run "v0_defend_center_s$s" --workers 4 || step "timelapse s$s FAILED"
done
if [ "$(free_gb)" -ge 4 ]; then
  step "fresh clone check"; scripts/check_fresh_clone.sh > runs/fresh_clone_check.log 2>&1 || step "fresh clone check FAILED"
else
  step "fresh clone check SKIPPED: only $(free_gb) GB free"
fi
step "finalize done"
touch runs/FINALIZE_DONE
