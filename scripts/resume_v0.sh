#!/usr/bin/env bash
# Resume v0: the 3 seed runs (each continues from its latest checkpoint) with their eval
# watchers, plus the smoke run's re-evaluation at low priority.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 1 2 3; do
  scripts/launch_run.sh "configs/v0_defend_center_s$s.yaml"
done
tmux new-session -d -s watch_v0_smoke \
  "nice -n 10 uv run doomscent eval-watcher --run v0_smoke >> runs/v0_smoke/logs/eval_watcher.out 2>&1"
tmux ls
