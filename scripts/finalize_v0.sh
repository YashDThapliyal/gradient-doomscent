#!/usr/bin/env bash
# Step 10 assets once training and evaluation are complete. Logs to runs/queue_v0.log.
set -uo pipefail
cd "$(dirname "$0")/.."
step() { echo "[$(date '+%F %T')] $*"; }
step "smoke re-evaluation"
uv run doomscent eval-watcher --run v0_smoke >> runs/v0_smoke/logs/eval_watcher.out 2>&1
for s in 1 2 3; do
  r="v0_defend_center_s$s"
  step "confirm best $r"
  uv run doomscent evaluate --run "$r" --ckpt best --episodes 100 --workers 6 --confirm-offset 1000 \
    > "runs/$r/logs/confirm_best.log" 2>&1 || step "confirm $r FAILED"
done
for s in 1 2 3; do
  r="v0_defend_center_s$s"
  step "figures $r";    uv run doomscent figures --run "$r" || step "figures $r FAILED"
  step "montage $r";    uv run doomscent montage --run "$r" --ckpts auto || step "montage $r FAILED"
  step "highlights $r"; uv run doomscent highlights --run "$r" --ckpt best || step "highlights $r FAILED"
  step "sidebyside $r"; uv run doomscent sidebyside --run "$r" --ckpts 0,best || step "sidebyside $r FAILED"
done
step "figures smoke"; uv run doomscent figures --run v0_smoke || step "figures smoke FAILED"
step "timelapse s1"; uv run doomscent timelapse --run v0_defend_center_s1 --workers 8 || step "timelapse s1 FAILED"
step "fresh clone check"; scripts/check_fresh_clone.sh > runs/fresh_clone_check.log 2>&1 || step "fresh clone check FAILED"
step "finalize done"
touch runs/FINALIZE_DONE
