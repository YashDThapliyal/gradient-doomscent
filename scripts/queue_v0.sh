#!/usr/bin/env bash
# Run v0 seeds one at a time so the Mac stays usable. Each run trains until its
# TRAINING_DONE.json exists; if its training session ends early (for example Sample
# Factory stopping after the Mac slept), it is resumed from its latest checkpoint.
# Finally re-evaluate the smoke run at low priority.
set -euo pipefail
cd "$(dirname "$0")/.."
run_to_completion() {
  local run="$1" config="$2"
  until [ -f "runs/$run/TRAINING_DONE.json" ]; do
    if ! tmux has-session -t "train_$run" 2>/dev/null; then
      echo "[$(date '+%F %T')] starting or resuming $run"
      tmux kill-session -t "watch_$run" 2>/dev/null || true
      scripts/launch_run.sh "$config"
    fi
    sleep 60
  done
  echo "[$(date '+%F %T')] $run finished"
}
for s in 1 2 3; do
  run_to_completion "v0_defend_center_s$s" "configs/v0_defend_center_s$s.yaml"
done
tmux new-session -d -s watch_v0_smoke \
  "nice -n 10 uv run doomscent eval-watcher --run v0_smoke >> runs/v0_smoke/logs/eval_watcher.out 2>&1"
echo "[$(date '+%F %T')] all seeds finished; smoke re-evaluation started"
