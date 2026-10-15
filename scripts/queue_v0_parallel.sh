#!/usr/bin/env bash
# Overnight v0: train every seed at the same time (the Mac is otherwise idle), resuming
# any run whose training session ends early (for example after a sleep), then wait for
# the eval watchers to finish their final clips and run scripts/finalize_v0.sh.
set -uo pipefail
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
  echo "[$(date '+%F %T')] $run finished training"
}
for s in 1 2 3; do
  run_to_completion "v0_defend_center_s$s" "configs/v0_defend_center_s$s.yaml" &
  sleep 30
done
wait
echo "[$(date '+%F %T')] all seeds finished training; waiting for eval watchers"
for s in 1 2 3; do
  while tmux has-session -t "watch_v0_defend_center_s$s" 2>/dev/null; do sleep 60; done
done
echo "[$(date '+%F %T')] eval watchers done; finalizing"
scripts/finalize_v0.sh
