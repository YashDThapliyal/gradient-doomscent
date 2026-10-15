#!/usr/bin/env bash
# Run v0 seeds one at a time so the Mac stays usable: wait for each run's training to
# finish (TRAINING_DONE.json), then start the next run and its eval watcher.
# Finally re-evaluate the smoke run at low priority.
set -euo pipefail
cd "$(dirname "$0")/.."
wait_done() { until [ -f "runs/$1/TRAINING_DONE.json" ]; do sleep 60; done; }
wait_done v0_defend_center_s1
scripts/launch_run.sh configs/v0_defend_center_s2.yaml
wait_done v0_defend_center_s2
scripts/launch_run.sh configs/v0_defend_center_s3.yaml
wait_done v0_defend_center_s3
tmux new-session -d -s watch_v0_smoke \
  "nice -n 10 uv run doomscent eval-watcher --run v0_smoke >> runs/v0_smoke/logs/eval_watcher.out 2>&1"
